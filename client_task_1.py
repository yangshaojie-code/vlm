"""任务一(40 分):在场景内找到长方体包装盒放到桌子上。

赛题(DG-202612)要求把该脚本命名为 client_task_1.py。

真实运行前提:
1. 已安装并配置好 DISCOVERSE(pip install -e . 且 import discoverse 成功)
2. 已拿到赛题方提供的场景资产(货架/包装盒/工具桶等 mjcf/3D 模型),
   并替换下面 `cfg.mjcf_file_path` 等标 TODO 的配置项
3. 已按赛题评分标准(见 PDF 第六节)实现 SimNode.check_success()

在没有安装 discoverse 的当前阶段,直接运行本脚本会自动切换到离线演示
模式:用 test_images/boxes.jpg 模拟一次完整的①-⑤流程(指令解析→目标
定位→抓取/放置位姿生成→IK 求解→状态机),只打印每一步的目标指令,
不做真实物理仿真,用来验证代码逻辑和坐标数值是否合理。
"""

import os

import numpy as np

from client_common import build_pick_and_place, build_pick_and_place_with_search, run_pick_place_loop
from motion_planning import get_ik_backend
from search_utils import get_obs, rotate_base
from sim_task_utils import env_name, object_is_placed_near, randomize_free_joints
from live_visualization import LiveVisualizer

TASK1_INSTRUCTION = "在场景内找到长方体包装盒放到桌子上"
ARM = "r"

try:
    from discoverse.robots_env.mmk2_base import MMK2Cfg
    from discoverse.task_base import MMK2TaskBase

    DISCOVERSE_AVAILABLE = True
except ImportError:
    DISCOVERSE_AVAILABLE = False


if DISCOVERSE_AVAILABLE:

    class SimNode(MMK2TaskBase):
        def domain_randomization(self):
            joint_names = [
                name.strip()
                for name in env_name("TASK1_RANDOM_JOINTS", "packing_box_freejoint").split(",")
                if name.strip()
            ]
            randomize_free_joints(self, joint_names)

        def check_success(self):
            return object_is_placed_near(
                self,
                env_name("TASK1_TARGET_BODY", "packing_box"),
                env_name("TASK1_TABLE_BODY", "table"),
                max_xy_distance=float(os.environ.get("TASK1_SUCCESS_XY", "0.65")),
                max_height_delta=float(os.environ.get("TASK1_SUCCESS_Z", "0.35")),
            )

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "mjcf/tasks_mmk2/pick_kiwi.xml"  #TODO 例如 mjcf/tasks_mmk2/xxx.xml
    cfg.obs_rgb_cam_id = [0]
    cfg.obs_depth_cam_id = [0]


def run_real():
    """真实环境:驱动 discoverse 仿真完成任务一。"""
    sim_node = SimNode(cfg)
    obs = sim_node.reset()
    ik_backend = get_ik_backend(prefer_real=True)
    visualizer = LiveVisualizer(enabled=os.environ.get("VLM_VISUALIZE", "1") != "0")

    cam_id = cfg.obs_rgb_cam_id[0]

    def _get_obs():
        return get_obs(sim_node, cam_id)

    def _rotate_base(delta_yaw):
        actual = rotate_base(sim_node, delta_yaw)
        print(f"[搜索] 旋转目标 {np.degrees(delta_yaw):.1f}°,实际 {np.degrees(actual):.1f}°")
        return actual

    _task, pick, place, info = build_pick_and_place_with_search(
        TASK1_INSTRUCTION,
        _get_obs,
        _rotate_base,
        arm=ARM,
        on_detection=visualizer.show_detection,
    )
    print(f"[搜索] 目标找到(旋转 {info['target_attempts']} 次)")

    def get_current_qpos():
        return sim_node.sensor_rgt_arm_qpos.copy() if ARM == "r" else sim_node.sensor_lft_arm_qpos.copy()

    def get_slide_pos():
        return float(sim_node.sensor_slide_qpos[0])

    execution_steps = {"count": 0}

    def apply_action(action, state):
        obs, _reward, _terminated, _truncated, _info = sim_node.step(action)
        execution_steps["count"] += 1
        images = obs.get("img")
        if execution_steps["count"] % 5 == 0 and images is not None and len(images) > cam_id:
            visualizer.show_execution(images[cam_id], state, sim_node.mj_data.time)
        if execution_steps["count"] % 30 == 0:
            print(f"[{state}] action 持续执行,mj_data.time={sim_node.mj_data.time:.2f}")

    try:
        run_pick_place_loop(
            pick, place, ik_backend, ARM,
            get_current_qpos, get_slide_pos, apply_action,
        )
    finally:
        visualizer.close()


def run_dry_run():
    """离线演示:未安装 discoverse 时,用合成测试图跑通①-⑤全链路。"""
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images", "boxes.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError("请先运行 python make_test_image.py 生成测试图")

    FAKE_DEPTH = 0.70  # 合成图不含真实几何,用固定占位深度(米)

    def depth_lookup(u, v):
        return FAKE_DEPTH

    task, pick, place = build_pick_and_place(TASK1_INSTRUCTION, image_path, depth_lookup, arm=ARM)
    print("① 指令解析结果:", task)
    print("③ 抓取位姿:", {k: v for k, v in pick.items() if k != "rotation"})
    print("③ 放置位姿:", {k: v for k, v in place.items() if k != "rotation"})

    ik_backend = get_ik_backend(prefer_real=False)  # 离线阶段强制使用 Mock,避免误报缺依赖
    q_ref = {"pos": np.zeros(6)}
    slide_pos = {"pos": 0.15}

    def apply_action(action, state):
        arm_slice = slice(12, 18) if ARM == "r" else slice(5, 11)
        gripper_idx = 18 if ARM == "r" else 11
        q_ref["pos"] = action[arm_slice].copy()  # Mock: simulate immediate joint feedback
        print(f"[{state}] 手臂关节角={np.round(action[arm_slice], 3)} 夹爪={action[gripper_idx]:.2f}")

    def get_current_qpos():
        return q_ref["pos"]

    def get_slide_pos():
        return slide_pos["pos"]

    task_fsm = run_pick_place_loop(pick, place, ik_backend, ARM, get_current_qpos, get_slide_pos, apply_action)
    assert task_fsm.is_done()
    print("[OK] 离线演示:①-⑤全链路跑通(未做真实物理仿真)")


if __name__ == "__main__":
    if DISCOVERSE_AVAILABLE:
        run_real()
    else:
        print("未检测到 discoverse,自动切换到离线演示模式(仅验证代码逻辑,不做真实仿真)\n")
        run_dry_run()
