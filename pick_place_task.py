"""④ 运动规划(状态机部分):抓取/放置位姿 -> IK -> 关节角/夹爪指令序列。

状态机产出的关节角/夹爪值,直接对应 discoverse.robots_env.mmk2_base.MMK2Base
的控制向量(action,长度 19)布局(见该文件 updateControl/getObservation 的
切片方式):
    index 0-1   底盘轮速(本项目不涉及,保持 0)
    index 2     升降关节 slide
    index 3-4   头部俯仰/偏航(本项目不涉及,保持 0)
    index 5-10  左臂 6 个关节
    index 11    左爪开合
    index 12-17 右臂 6 个关节
    index 18    右爪开合

拿到真实仿真环境后,只需把这里算出的 action 数组赋给 sim_node 对应的
tctr_lft_arm / tctr_rgt_arm / tctr_lft_gripper / tctr_rgt_gripper 切片
(或直接整体传给 sim_node.step(action)),状态机逻辑本身不需要改动。
"""

import numpy as np

ARM_JOINT_SLICE = {"l": slice(5, 11), "r": slice(12, 18)}
GRIPPER_INDEX = {"l": 11, "r": 18}
ACTION_DIM = 19

GRIPPER_OPEN = 1.0

# 抓取/放置目标"上方"与"物体高度"之间的垂直落差,与 grasp_pose.APPROACH_HEIGHT 对应
from grasp_pose import APPROACH_HEIGHT


class PickPlaceTask:
    """通用抓取-搬运-放置状态机,覆盖赛题任务一、任务二的共同流程。"""

    STATES = [
        "APPROACH_PICK",  # 移动到目标物体正上方,夹爪打开
        "DESCEND_GRASP",  # 下降到物体高度
        "CLOSE_GRIPPER",  # 合爪抓取
        "LIFT_AFTER_GRASP",  # 抬升离开货架/桌面
        "MOVE_TO_PLACE",  # 移动到放置目标正上方
        "DESCEND_PLACE",  # 下降到放置高度
        "OPEN_GRIPPER",  # 松爪释放
        "RETREAT",  # 抬升离开,任务结束
        "DONE",
    ]

    def __init__(self, pick_pose: dict, place_pose: dict, ik_backend, arm: str = "r", gripper_close_value: float = 0.5):
        if arm not in ("l", "r"):
            raise ValueError(f"arm 必须是 'l' 或 'r', 收到: {arm!r}")
        self.pick_pose = pick_pose
        self.place_pose = place_pose
        self.ik = ik_backend
        self.arm = arm
        self.gripper_close_value = gripper_close_value
        self.state_idx = 0
        self.history = []  # 记录每一步产生的目标,便于离线核验/调试

    @property
    def state(self) -> str:
        return self.STATES[self.state_idx]

    def is_done(self) -> bool:
        return self.state == "DONE"

    def advance(self) -> None:
        if self.state_idx < len(self.STATES) - 1:
            self.state_idx += 1

    def _target_for_state(self, state: str):
        """返回 (target_pos 或 None, target_rmat 或 None, gripper_value 或 None)。"""
        pick_above = self.pick_pose["position"]
        pick_object = self.pick_pose["object_position"]
        place_above = self.place_pose["position"]
        place_object = place_above - np.array([0.0, 0.0, APPROACH_HEIGHT])
        rmat = self.pick_pose["rotation"]

        table = {
            "APPROACH_PICK": (pick_above, rmat, GRIPPER_OPEN),
            "DESCEND_GRASP": (pick_object, rmat, None),
            "CLOSE_GRIPPER": (None, None, self.gripper_close_value),
            "LIFT_AFTER_GRASP": (pick_above, rmat, None),
            "MOVE_TO_PLACE": (place_above, rmat, None),
            "DESCEND_PLACE": (place_object, rmat, None),
            "OPEN_GRIPPER": (None, None, GRIPPER_OPEN),
            "RETREAT": (place_above, rmat, None),
        }
        if state not in table:
            raise ValueError(f"未知状态: {state}")
        return table[state]

    def step(self, action: np.ndarray, q_ref: np.ndarray, slide_pos: float) -> np.ndarray:
        """计算当前状态对应的关节角/夹爪指令,写入 action 并返回。

        action:    长度 19 的控制向量(原地修改后返回,便于连续调用)
        q_ref:     当前手臂 6 个关节角,作为 IK 迭代参考解
        slide_pos: 当前升降关节位置(米)
        """
        if self.is_done():
            return action

        target_pos, target_rmat, gripper_value = self._target_for_state(self.state)

        if target_pos is not None:
            jq = self.ik.solve(target_pos, target_rmat, self.arm, slide_pos, q_ref)
            action[ARM_JOINT_SLICE[self.arm]] = jq

        if gripper_value is not None:
            action[GRIPPER_INDEX[self.arm]] = gripper_value

        self.history.append(
            {"state": self.state, "target_pos": target_pos, "gripper": gripper_value}
        )
        return action


if __name__ == "__main__":
    import grasp_pose
    from motion_planning import MockIKBackend

    pick = grasp_pose.pick_pose((640, 400), depth=0.7, arm="r")
    place = grasp_pose.place_pose((900, 380), reference_depth=0.75, direction="left", reference_category="tool_bucket", arm="r")

    task = PickPlaceTask(pick, place, MockIKBackend(), arm="r", gripper_close_value=0.5)

    action = np.zeros(ACTION_DIM)
    q_ref = np.zeros(6)
    visited = []
    guard = 0
    while not task.is_done() and guard < 20:
        visited.append(task.state)
        action = task.step(action, q_ref, slide_pos=0.15)
        task.advance()
        guard += 1

    print("状态序列:", visited)
    assert visited == PickPlaceTask.STATES[:-1], "状态机未按预期顺序遍历全部状态"
    print("[OK] 状态机按预期顺序遍历全部状态")

    close_step = next(h for h in task.history if h["state"] == "CLOSE_GRIPPER")
    assert close_step["gripper"] == 0.5
    open_step = next(h for h in task.history if h["state"] == "OPEN_GRIPPER")
    assert open_step["gripper"] == GRIPPER_OPEN
    print("[OK] 夹爪开合指令在正确的状态被写入")

    approach = next(h for h in task.history if h["state"] == "APPROACH_PICK")
    assert np.allclose(approach["target_pos"], pick["position"])
    print("[OK] 目标位姿正确对应 grasp_pose 的输出")
