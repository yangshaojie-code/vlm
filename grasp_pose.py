"""③ 抓取点生成(Grasp Pose)。

输入:目标/参照物在图像中的像素中心 + 对应深度值(米)。
输出:机器人 base_link 坐标系下的抓取位姿或放置位姿
      {"position": np.ndarray(3,), "rotation": np.ndarray(3,3), "arm": "l"/"r"}

position/rotation 直接对应 DISCOVERSE 里 mocap_ik_mmk2.py 传给
MMK2IK().armIK_wrt_footprint(target_pos, target_rmat, arm, slide_pos, q_ref) 的
前两个参数,拿到仿真平台后无需再做坐标系转换。

物体尺寸取自赛题 PDF(DG-202612):
- 包装盒 24×16×19 cm(长×宽×高)
- 货架单层 203×80×28 cm,层间距 30cm
- 桌子 160×80×75 cm
工具桶/物料盒尺寸赛题未给出,用占位值,需在拿到真实场景后校正(见 TODO)。
"""

import numpy as np
from scipy.spatial.transform import Rotation

from camera_config import pixel_depth_to_base_point

# 包装盒尺寸(米),来自赛题 PDF
BOX_SIZE = {"length": 0.24, "width": 0.16, "height": 0.19}

# TODO: 工具桶/物料盒尺寸赛题未给出,拿到真实场景模型后替换为准确值
REFERENCE_OBJECT_RADIUS = {
    "tool_bucket": 0.15,
    "material_box": 0.12,
    "packing_box": max(BOX_SIZE["length"], BOX_SIZE["width"]) / 2,
}
DEFAULT_REFERENCE_RADIUS = 0.15

# 抓取时抬升到物体上方的安全高度(米),抓取后先到这个位置再下降合爪,
# 参考 discoverse 官方示例 kiwi_pick.py 的 approach 偏移量设计
APPROACH_HEIGHT = 0.10

# 放置时目标位置与参照物之间的额外安全间隙(米)
PLACEMENT_CLEARANCE = 0.05


def _top_down_rotation() -> np.ndarray:
    """默认俯视抓取姿态:夹爪 z 轴指向 -base_z(竖直向下)。"""
    return Rotation.from_euler("xyz", [0, 90, 0], degrees=True).as_matrix()


DEFAULT_TOP_DOWN_RMAT = _top_down_rotation()


def pick_pose(center_pixel: tuple, depth: float, arm: str = "r") -> dict:
    """根据目标物体的像素中心 + 深度,生成抓取位姿(位置为物体正上方 APPROACH_HEIGHT 处)。

    实际抓取时的状态机(见 pick_place_task.py)会先移动到该位姿(夹爪打开),
    再下降到物体高度合爪,与 discoverse 官方 kiwi_pick.py 的两段式抓取一致。
    """
    u, v = center_pixel
    object_base_point = pixel_depth_to_base_point(u, v, depth)
    grasp_position = object_base_point + np.array([0.0, 0.0, APPROACH_HEIGHT])
    return {
        "position": grasp_position,
        "object_position": object_base_point,
        "rotation": DEFAULT_TOP_DOWN_RMAT,
        "arm": arm,
    }


def place_pose(
    reference_center_pixel: tuple,
    reference_depth: float,
    direction: str,
    reference_category: str = None,
    arm: str = "r",
    task_to_base_yaw: float = 0.0,
) -> dict:
    """根据参照物的像素中心 + 深度 + 方向(left/right),生成放置位姿。

    方向定义在接收指令时的初始任务坐标系中:y 轴左正右负。
    task_to_base_yaw 将该任务方向旋转到当前 base 坐标系；底盘未旋转时为 0。
    偏移距离 = 参照物半径 + 包装盒半宽 + 安全间隙。
    """
    if direction not in ("left", "right"):
        raise ValueError(f"direction 必须是 'left' 或 'right',收到: {direction!r}")

    u, v = reference_center_pixel
    reference_base_point = pixel_depth_to_base_point(u, v, reference_depth)

    ref_radius = REFERENCE_OBJECT_RADIUS.get(reference_category, DEFAULT_REFERENCE_RADIUS)
    offset_distance = ref_radius + BOX_SIZE["width"] / 2 + PLACEMENT_CLEARANCE
    sign = 1.0 if direction == "left" else -1.0
    task_offset = np.array([0.0, sign * offset_distance, 0.0])
    c, s = np.cos(task_to_base_yaw), np.sin(task_to_base_yaw)
    task_to_base = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    offset = task_to_base @ task_offset

    place_position = reference_base_point + offset + np.array([0.0, 0.0, APPROACH_HEIGHT])
    return {
        "position": place_position,
        "reference_position": reference_base_point,
        "rotation": DEFAULT_TOP_DOWN_RMAT,
        "arm": arm,
    }


def place_pose_on_table(table_position_base: np.ndarray, arm: str = "r") -> dict:
    """任务一场景:无参照物/方向,直接放到桌面上某个固定/给定的 base 系坐标。"""
    place_position = np.asarray(table_position_base) + np.array([0.0, 0.0, APPROACH_HEIGHT])
    return {"position": place_position, "rotation": DEFAULT_TOP_DOWN_RMAT, "arm": arm}


if __name__ == "__main__":
    # 干跑自检:用 camera_config 的自检点反算一个抓取位姿,检查数值量级是否合理
    from camera_config import project_point_to_pixel

    fake_object_point = np.array([0.85, -0.10, 0.78])  # base 系:前方 0.85m,右偏 0.10m,高 0.78m
    u, v, depth = project_point_to_pixel(fake_object_point)
    pose = pick_pose((u, v), depth, arm="r")
    print(f"目标物体 base 坐标(还原): {pose['object_position']}")
    assert np.allclose(pose["object_position"], fake_object_point, atol=1e-6)
    assert np.allclose(pose["position"], fake_object_point + [0, 0, APPROACH_HEIGHT])
    print(f"抓取位姿(物体上方 {APPROACH_HEIGHT}m): {pose['position']}")
    print("[OK] pick_pose 坐标还原与偏移正确")

    fake_ref_point = np.array([0.90, 0.20, 0.78])
    ru, rv, rdepth = project_point_to_pixel(fake_ref_point)
    place = place_pose((ru, rv), rdepth, direction="left", reference_category="tool_bucket", arm="r")
    print(f"参照物 base 坐标(还原): {place['reference_position']}")
    assert np.allclose(place["reference_position"], fake_ref_point, atol=1e-6)
    assert place["position"][1] > fake_ref_point[1], "left 方向应向 +y 偏移"
    print(f"放置位姿(左边): {place['position']}")
    print("[OK] place_pose 方向偏移正确")
