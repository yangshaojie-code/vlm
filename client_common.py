"""client_task_1.py / client_task_2.py 共用逻辑:串联①-⑤全流程。

    ① 指令解析(task_parser)
    ② 目标定位(grounding)
    ③ 抓取/放置位姿生成(grasp_pose)
    ④ 运动规划:IK 求解 + 状态机(motion_planning + pick_place_task)
    ⑤ 应用到仿真:由调用方传入 apply_action / get_current_qpos 等回调,
      真实环境下这些回调读写 discoverse 的 sim_node,离线环境下用打印/
      合成数据模拟。

!! 重要 !!
- TABLE_PLACE_POSITION_BASE 是任务一"放到桌子上"时的目标位置占位值,
  赛题 PDF 只给了桌子尺寸(160×80×75cm)和"位置:场景左侧",并未给出
  机器人到桌子的精确几何关系。拿到真实场景后需要用桌面的真实检测结果
  或已知固定坐标替换这里的占位值。
- 深度获取方式(depth_lookup)由调用方决定:真实环境用 obs["depth"][v, u]
  (深度相机与 RGB 相机像素对齐的前提下),离线环境用固定占位深度。
"""

import numpy as np

from depth_utils import robust_depth_from_bbox
from grasp_pose import pick_pose, place_pose, place_pose_on_table
from grounding import locate_object, object_to_description
from pick_place_task import ACTION_DIM, ARM_JOINT_SLICE, PickPlaceTask
from task_parser import parse_instruction

DEFAULT_ARM = "r"

# TODO: 替换为真实场景桌面上的目标坐标(base_link 系,单位:米)
TABLE_PLACE_POSITION_BASE = np.array([0.75, 0.0, 0.80])

# 搜索参数:找不到目标时旋转底盘扫描四周
SEARCH_ROTATION_STEP_DEG = 120.0  # 每次旋转的角度(3 次 = 360°)
SEARCH_MAX_ATTEMPTS = 5           # 最多尝试次数(5 * 120° = 600°,覆盖一整圈有余)


def build_pick_and_place(instruction: str, image, depth_lookup, arm: str = DEFAULT_ARM):
    """①②③:指令解析 + 目标/参照物定位 + 抓取/放置位姿生成。

    depth_lookup(u, v) -> 深度值(米),由调用方提供。
    返回: (task_json, pick_pose_dict, place_pose_dict)
    """
    task = parse_instruction(instruction)

    target_desc = object_to_description(task["target_object"])
    target_result = locate_object(image, target_desc)
    if not target_result["found"]:
        raise RuntimeError(f"未在图像中找到目标物体: {target_desc}")
    u, v = target_result["center"]
    try:
        target_depth = depth_lookup(target_result["bbox"])
    except TypeError:
        # Backward compatibility for dry-run callbacks that accept (u, v).
        target_depth = depth_lookup(u, v)
    pick = pick_pose((u, v), target_depth, arm=arm)

    if task.get("reference_object"):
        ref_desc = object_to_description(task["reference_object"])
        ref_result = locate_object(image, ref_desc)
        if not ref_result["found"]:
            raise RuntimeError(f"未在图像中找到参照物: {ref_desc}")
        ru, rv = ref_result["center"]
        try:
            reference_depth = depth_lookup(ref_result["bbox"])
        except TypeError:
            reference_depth = depth_lookup(ru, rv)
        place = place_pose(
            (ru, rv),
            reference_depth,
            direction=task["direction"],
            reference_category=task["reference_object"]["category"],
            arm=arm,
        )
    else:
        # 任务一:无参照物/方向,直接放到桌面固定位置
        place = place_pose_on_table(TABLE_PLACE_POSITION_BASE, arm=arm)

    return task, pick, place


def _locate_with_search(description, get_obs, rotate_base, max_attempts, step_deg, on_detection=None):
    """在当前及旋转后的各视角中搜索 description 描述的物体。

    get_obs()       -> (image, depth_map),返回当前帧的 RGB + 深度
    rotate_base(d)  -> 旋转底盘 d 弧度(正=左转,负=右转)
    返回: (result, depth_map, attempts) 找到时 result 为 grounding 结果;
          找不到时抛 RuntimeError。
    """
    for attempt in range(max_attempts):
        image, depth_map = get_obs()
        result = locate_object(image, description)
        if result["found"]:
            if attempt > 0:
                print(f"[搜索] 第 {attempt} 次旋转后找到: {description}")
            if on_detection is not None:
                on_detection(image, result, description)
            return result, depth_map, attempt
        if attempt < max_attempts - 1:
            print(f"[搜索] 第 {attempt + 1}/{max_attempts} 次未找到 {description},旋转 {step_deg}° 继续搜索...")
            rotate_base(np.radians(step_deg))
    raise RuntimeError(f"已旋转搜索 {max_attempts} 次仍未找到: {description}")


def _rotate_pose_between_base_frames(pose, source_yaw, target_yaw):
    """Rotate a pose from the base frame at source_yaw into target_yaw's base frame."""
    delta = source_yaw - target_yaw
    c, s = np.cos(delta), np.sin(delta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    result = dict(pose)
    for key in ("position", "object_position", "reference_position"):
        if key in result:
            result[key] = rot @ np.asarray(result[key])
    if "rotation" in result:
        result["rotation"] = rot @ np.asarray(result["rotation"])
    return result


def build_pick_and_place_with_search(instruction, get_obs, rotate_base,
                                     arm=DEFAULT_ARM,
                                     max_attempts=SEARCH_MAX_ATTEMPTS,
                                     step_deg=SEARCH_ROTATION_STEP_DEG,
                                     on_detection=None):
    """①②③ + 搜索:找不到目标/参照物时旋转底盘重试。

    与 build_pick_and_place 的区别:不在当前单帧找不到就报错,而是旋转
    底盘扫描四周后再判。

    坐标一致性:pick/place 位姿都必须在同一个 base 系下。流程为
      1) 旋转搜索目标 -> 记录目标三维点及当时的累计偏航角
      2) 在同一视角找参照物;若找不到,继续旋转搜索参照物
      3) 将目标位姿从其观测时的 base 系旋转到最终 base 系,从而不要求
         目标和参照物必须同时出现在最后一帧
    任务一(放桌上)额外处理:跟踪累计偏航角,把 TABLE_PLACE_POSITION_BASE
    从初始 base 系旋转到当前 base 系。

    get_obs()      -> (image, depth_map)
    rotate_base(d) -> 旋转底盘 d 弧度,返回实际转过的角度
    返回: (task_json, pick_pose, place_pose, search_info)
        search_info = {"target_attempts": int, "reference_attempts": int}
    """
    task = parse_instruction(instruction)

    # 包装 rotate_base 以跟踪累计偏航角
    cum_yaw = [0.0]

    def _tracked_rotate(delta_yaw):
        actual = rotate_base(delta_yaw)
        if actual is None:
            # 调用方回调忘了 return 时兜底,用目标角度近似累计
            actual = float(delta_yaw)
        cum_yaw[0] += actual
        return actual

    target_desc = object_to_description(task["target_object"])
    target_result, depth_map, t_attempts = _locate_with_search(
        target_desc,
        get_obs,
        _tracked_rotate,
        max_attempts,
        step_deg,
        on_detection=(
            (lambda image, result, desc: on_detection(image, result, desc, "target", None))
            if on_detection else None
        ),
    )
    target_yaw = cum_yaw[0]

    def bbox_depth(dm, result):
        return robust_depth_from_bbox(dm, result["bbox"])

    u, v = target_result["center"]
    pick = pick_pose((u, v), bbox_depth(depth_map, target_result), arm=arm)

    ref_attempts = 0
    if task.get("reference_object"):
        ref_desc = object_to_description(task["reference_object"])
        # 先在当前(找到目标的)视角找参照物,不行再旋转搜索
        current_image, current_depth = get_obs()
        ref_result = locate_object(current_image, ref_desc)
        if not ref_result["found"]:
            print(f"[搜索] 当前视角未找到参照物 {ref_desc},开始旋转搜索...")
            ref_result, ref_depth, ref_attempts = _locate_with_search(
                ref_desc,
                get_obs,
                _tracked_rotate,
                max_attempts,
                step_deg,
                on_detection=(
                    (lambda image, result, desc: on_detection(
                        image, result, desc, "reference", task["direction"]
                    )) if on_detection else None
                ),
            )
            dm_for_ref = ref_depth
        else:
            dm_for_ref = current_depth
            if on_detection:
                on_detection(
                    current_image, ref_result, ref_desc, "reference", task["direction"]
                )

        # The reference is in the final base frame. Rotate the previously computed
        # pick pose from its observation frame into this same frame.
        pick = _rotate_pose_between_base_frames(pick, target_yaw, cum_yaw[0])
        ru, rv = ref_result["center"]
        place = place_pose(
            (ru, rv),
            bbox_depth(dm_for_ref, ref_result),
            direction=task["direction"],
            reference_category=task["reference_object"]["category"],
            arm=arm,
            task_to_base_yaw=-cum_yaw[0],
        )
    else:
        # 任务一:无参照物,放到桌面固定位置。
        # TABLE_PLACE_POSITION_BASE 定义在初始 base 系,底盘转过 cum_yaw 后,
        # 桌面在当前 base 系的位置 = Rz(-cum_yaw) @ 原位置
        yaw = cum_yaw[0]
        if abs(yaw) > 1e-3:
            c, s = np.cos(-yaw), np.sin(-yaw)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            table_pos_current = rot @ TABLE_PLACE_POSITION_BASE
            print(f"[搜索] 底盘累计转过 {np.degrees(yaw):.1f}°,桌面坐标已旋转到当前 base 系")
        else:
            table_pos_current = TABLE_PLACE_POSITION_BASE
        place = place_pose_on_table(table_pos_current, arm=arm)

    return task, pick, place, {"target_attempts": t_attempts, "reference_attempts": ref_attempts}


def run_pick_place_loop(
    pick,
    place,
    ik_backend,
    arm,
    get_current_qpos,
    get_slide_pos,
    apply_action,
    joint_tolerance=0.03,
    stable_steps=5,
    gripper_hold_steps=30,
    max_steps_per_state=600,
):
    """④⑤:持续控制直到关节到位，再切换状态。

    运动状态要求关节误差连续 stable_steps 次小于 joint_tolerance；夹爪状态
    保持 gripper_hold_steps 个仿真步。任一状态超时会抛出 RuntimeError。
    """
    task = PickPlaceTask(pick, place, ik_backend, arm=arm)
    action = np.zeros(ACTION_DIM)
    arm_slice = ARM_JOINT_SLICE[arm]

    while not task.is_done():
        state = task.state
        target_pos, _target_rmat, _gripper = task._target_for_state(state)
        required_stable = stable_steps if target_pos is not None else gripper_hold_steps
        consecutive = 0

        for _ in range(max_steps_per_state):
            action = task.step(action, get_current_qpos(), get_slide_pos())
            apply_action(action, state)

            if target_pos is None:
                consecutive += 1
            else:
                error = np.max(np.abs(get_current_qpos() - action[arm_slice]))
                consecutive = consecutive + 1 if error <= joint_tolerance else 0

            if consecutive >= required_stable:
                break
        else:
            raise RuntimeError(
                f"状态 {state} 在 {max_steps_per_state} 步内未收敛，"
                f"关节容差={joint_tolerance:.3f} rad"
            )

        task.advance()
    return task
