"""底盘旋转搜索工具。

当 VLM 在当前视角找不到目标物体时,通过差速驱动旋转底盘扫描四周,
每转一个角度重新拍照 + grounding,直到找到目标或转完一整圈。

action[0:2] 是左右轮速,差速驱动转向:
    左转(逆时针, yaw 增大): 左轮后退, 右轮前进  -> action[0] = -v, action[1] = +v
    右转(顺时针, yaw 减小): 左轮前进, 右轮后退  -> action[0] = +v, action[1] = -v

DISCOVERSE 的 base_orientation 传感器返回四元数 [w, x, y, z],
yaw 从中提取后用于闭环控制旋转角度。
"""

import numpy as np

from pick_place_task import ACTION_DIM


def quat_to_yaw(quat) -> float:
    """四元数 [w, x, y, z] -> 绕 z 轴的偏航角(弧度,范围 (-pi, pi])。"""
    w, x, y, z = np.asarray(quat, dtype=float).tolist()
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def angle_diff(target: float, current: float) -> float:
    """从 current 到 target 的最短有向角差,范围 (-pi, pi]。"""
    d = (target - current + np.pi) % (2 * np.pi) - np.pi
    return float(d)


def rotate_base(sim_node, delta_yaw: float, wheel_vel: float = 8.0,
                max_steps: int = 12000, tolerance: float = 0.05,
                settle_steps: int = 40) -> float:
    """闭环旋转底盘 delta_yaw 弧度,返回实际转过的角度。

    用 base_orientation 传感器做反馈,转到目标角度(容差 tolerance)或达到
    max_steps 上限后停止,再发 settle_steps 步零指令让底盘停稳。

    wheel_vel 会再被 updateControl 裁剪到 actuator_ctrlrange;默认给较大值,
    让差速转向尽量跑满量程。实测 0.3/1.5 都偏小,默认 8.0。
    """
    start_yaw = quat_to_yaw(sim_node.sensor_base_orientation)
    target_yaw = start_yaw + delta_yaw

    direction = 1.0 if delta_yaw >= 0 else -1.0
    turn_action = np.zeros(ACTION_DIM)

    for step_i in range(max_steps):
        current_yaw = quat_to_yaw(sim_node.sensor_base_orientation)
        remaining = angle_diff(target_yaw, current_yaw)
        if abs(remaining) < tolerance:
            break
        # 只在最后约 10° 内减速,避免全程被 scale 拖慢
        scale = 1.0 if abs(remaining) > 0.18 else max(0.35, abs(remaining) / 0.18)
        turn_action[0] = -wheel_vel * direction * scale   # 左轮
        turn_action[1] = +wheel_vel * direction * scale   # 右轮
        sim_node.step(turn_action)
    else:
        # 正常 for 跑完(未 break)说明超时,打印便于继续调参
        current_yaw = quat_to_yaw(sim_node.sensor_base_orientation)
        remaining = angle_diff(target_yaw, current_yaw)
        print(
            f"[搜索] 旋转超时: 已跑 {max_steps} 步,剩余 {np.degrees(remaining):.1f}°,"
            f" 可再增大 wheel_vel/max_steps"
        )

    # 停稳
    stop = np.zeros(ACTION_DIM)
    for _ in range(settle_steps):
        sim_node.step(stop)

    final_yaw = quat_to_yaw(sim_node.sensor_base_orientation)
    return angle_diff(final_yaw, start_yaw)


def get_obs(sim_node, cam_id: int):
    """取当前观测的 (image_rgb, depth_map),不推进仿真。"""
    obs = sim_node.getObservation()
    return obs["img"][cam_id], obs["depth"][cam_id]


if __name__ == "__main__":
    # 离线自检:四元数 -> yaw 往返
    tests = [
        ([1, 0, 0, 0], 0.0),
        ([0, 0, 0, 1], np.pi),
        ([0, 0, 0, -1], -np.pi),
        ([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], np.pi / 2),
        ([np.cos(-np.pi / 8), 0, 0, np.sin(-np.pi / 8)], -np.pi / 4),
    ]
    for quat, expected in tests:
        got = quat_to_yaw(quat)
        ok = abs(angle_diff(expected, got)) < 1e-6
        print(f"quat={quat} -> yaw={got:.4f}  expected={expected:.4f}  {'OK' if ok else 'FAIL'}")
        assert ok

    # angle_diff 包裹性
    assert abs(angle_diff(np.pi - 0.1, -np.pi + 0.1) - (-0.2)) < 1e-6
    assert abs(angle_diff(-np.pi + 0.1, np.pi - 0.1) - 0.2) < 1e-6
    print("[OK] search_utils 自检通过")
