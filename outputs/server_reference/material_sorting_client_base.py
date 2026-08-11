#!/usr/bin/env python3
"""物料分拣 baseline client 的共享基类。

驱动 MMK2：导航到货架 → 视觉锁定彩色盒 → 部署右臂抓取姿态 → 车体 creep 进货架 →
夹取抬起 → 倒车 → 导航到原料区桌子 → 放置 → 返回结束区。

坐标世界系(+X 东 / +Y 北)。抓取姿态在 footprint 系(相对基座,与世界朝向无关)，因此
货架在西墙、机器人面向西 creep 进去，与参考仓库(面向北)用同一套抓取几何。

子类只需实现：
  - select_target(dets) -> (world_xyz, meta) | None   从 /material/detections 选目标
  - place_target(client) -> bool                       放置逻辑(桌面 / 道具旁),完成返回 True
  - target_ready() 决定是否已拿到指令

控制向量 tc(19): [base_lin, base_ang, slide, head_yaw, head_pitch,
                  l_arm(6), l_grip, r_arm(6), r_grip]
"""
import json
import math
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from vision_msgs.msg import Detection3DArray

from discoverse.utils import step_func
from mmk2_kdl import MMK2Kdl

# ---- 抓取姿态(footprint 系，沿用参考仓库经 teleop 标定的水平夹取) ----
GRIP_OPEN, GRIP_CLOSE = 1.0, 0.10
GRASP_ROT = np.array([
    [0.93909533, -0.34348620, 0.01082593],
    [0.34365571, 0.93870412, -0.02711702],
    [-0.00084802, 0.02918586, 0.99957364],
])
INIT_ARM_L = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
INIT_ARM_R = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]
HEAD_PITCH = -0.5

# footprint 系抓取几何：forward=fp[0](朝货架), lateral=fp[1], up=fp[2]
DEPLOY_BACKOFF = 0.32     # 部署时夹爪在目标前方(朝机器人)后撤量 m
CREEP_STOP_FWD = 0.02     # creep 到夹爪距目标 forward 这么近时停(居中夹取)
GRASP_Z_OFFSET = 0.02     # 夹爪相对盒中心的竖直偏置
LIFT_AMOUNT = 0.06        # 夹住后减小 slide 抬起量
CREEP_SPEED = 0.06
RETREAT_SPEED = 0.20

# 相位
(NAV_SHELF, SCAN, DEPLOY, CREEP, CLOSE, LIFT, RETREAT,
 NAV_TABLE, PLACE, NAV_END, DONE) = range(11)
PHASE_NAME = {NAV_SHELF: "nav->shelf", SCAN: "scan", DEPLOY: "deploy", CREEP: "creep",
              CLOSE: "close", LIFT: "lift", RETREAT: "retreat", NAV_TABLE: "nav->table",
              PLACE: "place", NAV_END: "nav->end", DONE: "done"}


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# 场景常量(世界系，来自 layout/scene)
SHELF_FRONT_X = -2.50          # 货架前沿 x
APPROACH_STANDOFF_X = 0.90     # 取货时基座停在盒子东侧(+X)这么远
TABLE_TOP_Z = 0.747
END_XY = np.array([-0.70, 0.55])
YAW_WEST = math.pi             # 面向货架(-X)
YAW_NORTH = math.pi / 2.0      # 面向原料区桌子(+Y)

JOINT_NAMES = [
    "slide_joint", "head_yaw_joint", "head_pitch_joint",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4",
    "left_arm_joint5", "left_arm_joint6", "left_arm_eef_gripper_joint",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4",
    "right_arm_joint5", "right_arm_joint6", "right_arm_eef_gripper_joint",
]


class MaterialSortingClientBase(Node):
    """pick→place→return 状态机基类。子类实现 select_target / compute_place_world。"""

    def __init__(self, node_name="material_sorting_client"):
        super().__init__(node_name)
        self.kdl = MMK2Kdl()

        self.tc = np.zeros(19)
        self.tc[5:11] = INIT_ARM_L
        self.tc[11] = GRIP_OPEN
        self.tc[12:18] = INIT_ARM_R
        self.tc[18] = GRIP_OPEN
        self.action = self.tc.copy()
        self.joint_move_ratio = np.ones(19)
        self.tc_prev = self.tc.copy()
        self.joint_slew = 1.2

        # feedback
        self.base_xy = None
        self.base_yaw = 0.0
        self.jpos = None
        self.instructions = None       # 从 /material/instruction 读到的任务指令

        # perception
        self.det_buf = deque(maxlen=30)
        self.target_locked = False
        self.OBJECT_WORLD = None
        self.place_world = None

        # phase state
        self.phase = NAV_SHELF
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.state_t0 = self.now()
        self.deploy_set = False
        self.arm_target_set = False
        self.place_sub = 0

        # gains / ramps
        self.pos_tol, self.turn_tol = 0.06, 0.03
        self.max_lin, self.max_ang = 0.45, 1.2
        self.rate_hz = 50.0
        self.dt = 1.0 / self.rate_hz
        self.max_lin_acc, self.max_ang_acc = 0.8, 5.0
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0

        # io
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.spine_pub = self.create_publisher(Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.larm_pub = self.create_publisher(Float64MultiArray, "/left_arm_forward_position_controller/commands", 5)
        self.rarm_pub = self.create_publisher(Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom", self.odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.js_cb, 10)
        self.create_subscription(Detection3DArray, "/material/detections", self.det_cb, 10)
        self.create_subscription(String, "/material/instruction", self.instr_cb, 5)

        self.timer = self.create_timer(self.dt, self.tick)
        self.last_log = 0.0
        self.get_logger().info(f"{node_name} up; waiting for odom + joint_states + instruction...")

    # ---- hooks for subclasses ----
    def select_target(self, dets):
        """dets: list[(class_str, world_xyz, score)]. 返回 world_xyz 或 None。"""
        raise NotImplementedError

    def compute_place_world(self):
        """抓到盒后，返回世界系放置点(桌面上方)。"""
        raise NotImplementedError

    # ---- ros helpers ----
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_xy = np.array([p.x, p.y])
        self.base_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

    def js_cb(self, msg):
        self.jpos = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}

    def instr_cb(self, msg):
        try:
            self.instructions = json.loads(msg.data)
        except Exception:
            pass

    def det_cb(self, msg):
        if self.target_locked or self.phase not in (SCAN, DEPLOY):
            return
        for det in msg.detections:
            if not det.results:
                continue
            r = det.results[0]
            cls = r.hypothesis.class_id
            pos = r.pose.pose.position
            self.det_buf.append((cls, np.array([pos.x, pos.y, pos.z]), r.hypothesis.score))

    @property
    def slide_meas(self):
        return self.jpos.get("slide_joint", self.tc[2])

    @property
    def rarm_meas(self):
        return np.array([self.jpos.get(f"right_arm_joint{i+1}", self.tc[12 + i]) for i in range(6)])

    # ---- frames ----
    def world_to_footprint(self, p_world):
        d = np.array(p_world, dtype=float) - np.array([self.base_xy[0], self.base_xy[1], 0.0])
        c, s = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def footprint_to_world(self, fp):
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        return np.array([self.base_xy[0] + c * fp[0] - s * fp[1],
                         self.base_xy[1] + s * fp[0] + c * fp[1], fp[2]])

    def arm_to(self, world_pos, rot=GRASP_ROT):
        fp = self.world_to_footprint(world_pos)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = fp
        ref = np.zeros(7)
        ref[0] = float(self.tc[2])
        ref[1:] = self.rarm_meas
        sols = self.kdl.inverse_kinematics(T_left=None, T_right=T, ref_pos=ref, target_height=float(self.tc[2]))
        if sols:
            self.tc[12:18] = np.asarray(sols[0])[1:7]
            self.arm_target_set = True
            return True
        self.get_logger().warn(f"IK unreachable: world={np.round(world_pos,3)} fp={np.round(fp,3)} (arm holds)")
        return False

    def ee_world(self):
        _, T = self.kdl.forward_kinematics(np.concatenate([[float(self.slide_meas)], self.rarm_meas]), index="right")
        return self.footprint_to_world(T[:3, 3])

    # ---- smoothing / publish ----
    def smooth_step(self):
        if not np.allclose(self.tc[2:19], self.tc_prev[2:19]):
            dif = np.abs(self.action[2:19] - self.tc[2:19])
            self.joint_move_ratio[2:19] = dif / (np.max(dif) + 1e-6)
            self.joint_move_ratio[2] *= 0.3
            self.tc_prev[:] = self.tc
        step = self.joint_slew * self.dt
        for i in range(2, 19):
            self.action[i] = step_func(self.action[i], self.tc[i], self.joint_move_ratio[i] * step)

    def publish(self):
        tw = Twist()
        tw.linear.x = float(self.tc[0])
        tw.angular.z = float(self.tc[1])
        self.cmd_vel_pub.publish(tw)
        self.spine_pub.publish(Float64MultiArray(data=[float(self.action[2])]))
        self.head_pub.publish(Float64MultiArray(data=[float(self.action[3]), float(self.action[4])]))
        self.larm_pub.publish(Float64MultiArray(data=[float(x) for x in self.action[5:11]] + [float(self.action[11])]))
        self.rarm_pub.publish(Float64MultiArray(data=[float(x) for x in self.action[12:18]] + [float(self.action[18])]))

    # ---- navigation ----
    def set_twist(self, lin, ang):
        self.des_lin = float(np.clip(lin, -self.max_lin, self.max_lin))
        self.des_ang = float(np.clip(ang, -self.max_ang, self.max_ang))

    def ramp_twist(self):
        dl = np.clip(self.des_lin - self.cur_lin, -self.max_lin_acc * self.dt, self.max_lin_acc * self.dt)
        da = np.clip(self.des_ang - self.cur_ang, -self.max_ang_acc * self.dt, self.max_ang_acc * self.dt)
        self.cur_lin += dl
        self.cur_ang += da
        self.tc[0], self.tc[1] = self.cur_lin, self.cur_ang

    def follow_route(self, route, final_yaw):
        if self.nav_idx < len(route):
            target = np.array(route[self.nav_idx], dtype=float)
            delta = target - self.base_xy
            dist = float(np.linalg.norm(delta))
            yaw_err = wrap_to_pi(math.atan2(delta[1], delta[0]) - self.base_yaw)
            if self.nav_mode == "turn":
                self.set_twist(0.0, 2.2 * yaw_err)
                if abs(yaw_err) < self.turn_tol:
                    self.nav_mode = "drive"
            else:
                if dist < self.pos_tol:
                    self.nav_idx += 1
                    self.nav_mode = "turn"
                    self.set_twist(0.0, 0.0)
                else:
                    if abs(yaw_err) < 0.05 or dist < 0.25:
                        ang = 0.0
                    else:
                        ang = 2.2 * yaw_err
                    align = max(0.0, math.cos(yaw_err))
                    self.set_twist(1.0 * dist * align, ang)
            return False
        yaw_err = wrap_to_pi(final_yaw - self.base_yaw)
        self.set_twist(0.0, 1.8 * yaw_err)
        if abs(yaw_err) < self.turn_tol:
            self.set_twist(0.0, 0.0)
            return True
        return False

    def reset_nav(self):
        self.nav_idx = 0
        self.nav_mode = "turn"

    def action_done(self, dwell=0.4):
        if self.now() - self.state_t0 < dwell:
            return False
        slide_ok = abs(self.slide_meas - self.tc[2]) < 0.02
        arm_ok = (not self.arm_target_set) or np.max(np.abs(self.rarm_meas - self.tc[12:18])) < 0.05
        return slide_ok and arm_ok

    # ---- perception lock ----
    def _lock_target(self, want_color=None):
        """从 det_buf 里按颜色(可选)聚合出目标世界坐标。返回是否锁定。"""
        cands = [(c, p, s) for (c, p, s) in self.det_buf if want_color is None or c == want_color]
        if len(cands) < 3:
            return False
        pts = np.array([p for _, p, _ in cands])
        self.OBJECT_WORLD = np.median(pts, axis=0)
        self.target_locked = True
        self.get_logger().info(f"[perception] target locked color={want_color} world={np.round(self.OBJECT_WORLD,3)} n={len(cands)}")
        return True

    def slide_for_height(self, obj_z):
        """粗略把升降平台设到能水平够到目标高度的位置(参考: slide 0.11 时够到 z~0.92)。"""
        return float(np.clip(0.11 + (0.92 - obj_z), -0.04, 0.87))

    # ---- deploy/creep geometry (footprint frame) ----
    def _deploy_world(self):
        """把夹爪部署到目标前方(朝机器人后撤 DEPLOY_BACKOFF)的世界点。"""
        fp = self.world_to_footprint(self.OBJECT_WORLD)
        fp[0] -= DEPLOY_BACKOFF
        fp[2] += GRASP_Z_OFFSET
        return self.footprint_to_world(fp)

    def _creep_done(self):
        """夹爪 forward 距目标 <= CREEP_STOP_FWD 视为到位。"""
        ee_fp = self.world_to_footprint(self.ee_world())
        obj_fp = self.world_to_footprint(self.OBJECT_WORLD)
        return ee_fp[0] >= obj_fp[0] - CREEP_STOP_FWD

    # ---- routes ----
    def route_to_shelf(self):
        """停到目标盒东侧 APPROACH_STANDOFF_X、朝西。"""
        stop_x = self.OBJECT_WORLD[0] + APPROACH_STANDOFF_X if self.OBJECT_WORLD is not None else -1.75
        y = self.OBJECT_WORLD[1] if self.OBJECT_WORLD is not None else 0.72
        return [[stop_x, y]]

    # ---- main tick ----
    def tick(self):
        if self.base_xy is None or self.jpos is None or self.instructions is None:
            return

        if self.phase == NAV_SHELF:
            # 先大致开到货架前(未锁目标前用缺省停靠点)，同时抬头看货架积累检测
            self.tc[4] = HEAD_PITCH
            approach_x = (self.OBJECT_WORLD[0] + APPROACH_STANDOFF_X
                          if self.OBJECT_WORLD is not None else -1.60)
            if self.follow_route([[approach_x, 0.72]], YAW_WEST):
                self.phase = SCAN
                self.state_t0 = self.now()
        elif self.phase == SCAN:
            # 面向货架静止,积累 /material/detections,锁定目标盒
            self.set_twist(0.0, 0.0)
            self.tc[4] = HEAD_PITCH
            self.tc[2] = self.slide_for_height(
                self.OBJECT_WORLD[2] if self.OBJECT_WORLD is not None else 0.90)
            want = self.wanted_color()
            if self._lock_target(want) and self.now() - self.state_t0 > 1.0:
                # 锁定后微调基座到标准站位再部署
                self.tc[2] = self.slide_for_height(self.OBJECT_WORLD[2])
                self.reset_nav()
                self.phase = DEPLOY
                self.deploy_set = False
                self.state_t0 = self.now()
        elif self.phase == DEPLOY:
            # 站位对齐目标 y、朝西,把右臂摆成抓取姿态(开阔处,手臂不再动)
            if self.follow_route(self.route_to_shelf(), YAW_WEST):
                self.tc[18] = GRIP_OPEN
                if not self.deploy_set:
                    if self.arm_to(self._deploy_world()):
                        self.deploy_set = True
                        self.state_t0 = self.now()
                if self.deploy_set and self.action_done():
                    self.phase = CREEP
        elif self.phase == CREEP:
            # 保持手臂不动,车直着往货架开,把夹爪平移送到盒处
            if not self._creep_done():
                self.set_twist(CREEP_SPEED, 4.0 * wrap_to_pi(YAW_WEST - self.base_yaw))
            else:
                self.set_twist(0.0, 0.0)
                self.phase = CLOSE
                self.state_t0 = self.now()
        elif self.phase == CLOSE:
            self.set_twist(0.0, 0.0)
            self.tc[18] = GRIP_CLOSE
            if self.now() - self.state_t0 > 0.8:
                self.phase = LIFT
        elif self.phase == LIFT:
            self.set_twist(0.0, 0.0)
            self.tc[2] = self.slide_meas - LIFT_AMOUNT
            if abs(self.slide_meas - self.tc[2]) < 0.02:
                self.phase = RETREAT
                self.state_t0 = self.now()
        elif self.phase == RETREAT:
            # 倒车退出货架(保持抓取姿态),盒还夹在手里
            back_x = self.OBJECT_WORLD[0] + APPROACH_STANDOFF_X + 0.15
            if self.base_xy[0] < back_x:
                self.set_twist(-RETREAT_SPEED, 1.0 * wrap_to_pi(YAW_WEST - self.base_yaw))
            else:
                self.set_twist(0.0, 0.0)
                self.place_world = self.compute_place_world()
                self.reset_nav()
                self.phase = NAV_TABLE
        elif self.phase == NAV_TABLE:
            # 开到桌子前方(配送区),朝北
            tx = float(np.clip(self.place_world[0], -1.55, 0.30))
            if self.follow_route([[tx, 0.72], [tx, 1.55]], YAW_NORTH):
                self.phase = PLACE
                self.place_sub = 0
                self.state_t0 = self.now()
        elif self.phase == PLACE:
            self.set_twist(0.0, 0.0)
            if self.place_sub == 0:
                # 把夹爪伸到放置点上方,再下降
                above = self.place_world.copy()
                above[2] = max(self.place_world[2] + 0.12, TABLE_TOP_Z + 0.15)
                self.arm_to(above)
                if self.action_done():
                    self.place_sub = 1
                    self.state_t0 = self.now()
            elif self.place_sub == 1:
                self.arm_to(self.place_world)
                if self.action_done():
                    self.place_sub = 2
                    self.state_t0 = self.now()
            else:
                self.tc[18] = GRIP_OPEN
                if self.now() - self.state_t0 > 1.0:
                    self.reset_nav()
                    self.phase = NAV_END
        elif self.phase == NAV_END:
            # 收臂并返回结束区
            self.tc[12:18] = INIT_ARM_R
            if self.follow_route([[END_XY[0], 1.2], list(END_XY)], YAW_NORTH):
                self.phase = DONE
                self.get_logger().info("task cycle done: returned to end zone")
        else:
            self.set_twist(0.0, 0.0)

        self.ramp_twist()
        self.smooth_step()
        self.publish()

        if self.now() - self.last_log > 1.0:
            obj = self.OBJECT_WORLD
            self.get_logger().info(
                f"phase={PHASE_NAME[self.phase]} base=({self.base_xy[0]:.2f},{self.base_xy[1]:.2f}) "
                f"yaw={self.base_yaw:.2f} slide={self.slide_meas:.3f} locked={self.target_locked} "
                f"obj={None if obj is None else np.round(obj,3)}")
            self.last_log = self.now()

    def wanted_color(self):
        """子类可覆盖:锁定目标时要求的颜色(任务二按指令颜色)。默认 None=任意。"""
        return None

