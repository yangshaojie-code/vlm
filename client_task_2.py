"""任务二(60 分):找到一个[颜色]的包装盒,并将其放到[指定道具]的[指定方向]。

赛题(DG-202612)要求把该脚本命名为 client_task_2.py。示例指令:
"找到一个粉色长方体包装盒,放到圆形工具桶左边"。

真实运行前提同 client_task_1.py(见其文件头注释)。

!! TODO:指令来源 !!
赛题 PDF 描述"当前一轮任务被判定完成后,系统自动下发第二条指令",
说明具体指令是仿真平台服务端在运行时下发的,但公开的 DISCOVERSE 仓库里
没有找到这个下发接口的具体形式(可能是某个 sim_node 属性、消息队列或
文件)。拿到真实平台文档后,需要把 get_instruction() 里的占位逻辑替换成
真正的接收方式;现在用环境变量 TASK2_INSTRUCTION 或 PDF 里的示例指令
兜底,便于离线开发和调试。

在没有安装 discoverse 的当前阶段,直接运行本脚本会自动切换到离线演示
模式:用 test_images/boxes.jpg 模拟一次完整的①-⑤流程,只打印每一步的
目标指令,不做真实物理仿真。
"""

import os

import numpy as np

from client_common import build_pick_and_place, build_pick_and_place_with_search, run_pick_place_loop
from motion_planning import get_ik_backend
from search_utils import get_obs, rotate_base
from sim_task_utils import env_name, object_is_placed_near, randomize_free_joints
from live_visualization import LiveVisualizer

DEFAULT_INSTRUCTION = "找到一个粉色长方体包装盒,放到圆形工具桶左边"
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
                for name in env_name(
                    "TASK2_RANDOM_JOINTS",
                    "pink_box_freejoint,yellow_box_freejoint,brown_box_freejoint",
                ).split(",")
                if name.strip()
            ]
            randomize_free_joints(self, joint_names)

        def check_success(self):
            task = getattr(self, "current_task", None)
            if not task:
                return False
            target = task["target_object"]
            reference = task["reference_object"]
            color = target.get("color") or ""
            target_default = f"{color}_packing_box" if color else "packing_box"
            reference_default = reference["category"]
            return object_is_placed_near(
                self,
                env_name("TASK2_TARGET_BODY", target_default),
                env_name("TASK2_REFERENCE_BODY", reference_default),
                direction=task["direction"],
                max_xy_distance=float(os.environ.get("TASK2_SUCCESS_XY", "0.55")),
                min_direction_offset=float(os.environ.get("TASK2_DIRECTION_OFFSET", "0.05")),
                max_height_delta=float(os.environ.get("TASK2_SUCCESS_Z", "0.35")),
            )

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "TODO: 替换为赛题提供的场景 mjcf 路径"#TODO: 替换为赛题提供的场景 mjcf 路径
    cfg.obs_rgb_cam_id = [0]
    cfg.obs_depth_cam_id = [0]


def get_instruction() -> str:
    """获取当前这一轮的任务指令。

    TODO: 替换为仿真平台服务端真正的指令下发方式(见文件头说明),
    目前用环境变量兜底,方便离线开发时手动指定不同指令测试。
    """
    return os.environ.get("TASK2_INSTRUCTION", DEFAULT_INSTRUCTION)


def run_real():
    """真实环境:驱动 discoverse 仿真完成任务二。"""
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

    instruction = get_instruction()
    _task, pick, place, info = build_pick_and_place_with_search(
        instruction,
        _get_obs,
        _rotate_base,
        arm=ARM,
        on_detection=visualizer.show_detection,
    )
    sim_node.current_task = _task
    print(f"[搜索] 目标找到(旋转 {info['target_attempts']} 次),参照物(旋转 {info['reference_attempts']} 次)")

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

    instruction = get_instruction()
    print(f"本轮指令: {instruction}")
    task, pick, place = build_pick_and_place(instruction, image_path, depth_lookup, arm=ARM)
    print("① 指令解析结果:", task)
    print("③ 抓取位姿:", {k: v for k, v in pick.items() if k != "rotation"})
    print("③ 放置位姿:", {k: v for k, v in place.items() if k != "rotation"})

    ik_backend = get_ik_backend(prefer_real=False)
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
