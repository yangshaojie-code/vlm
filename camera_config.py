"""相机内外参配置(③ 三维坐标转换用)。

!! 重要 !!
本文件中的数值均为占位估计值(视场角、安装位置/俯仰角),用于在没有真实
DISCOVERSE 场景前打通"像素+深度 -> 机器人 base_link 坐标系三维坐标"的算法链路。
拿到仿真平台的真实场景后,必须用平台提供的相机标定参数(或从 MJCF 里读取
相机的 fovy、位置、四元数)替换 INTRINSICS / CAMERA_POSITION_WRT_BASE /
CAMERA_ROTATION_WRT_BASE 这三个值,grasp_pose.py 的算法本身不需要改。

坐标系约定:
- 机器人 base_link 坐标系(与 discoverse.robots_env.mmk2_base 的
  sensor_base_position/orientation 一致):x 前,y 左,z 上。
- 相机坐标系(OpenCV/常见针孔相机约定):x 右,y 下,z 前(光轴方向)。
"""

import numpy as np

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FOV_Y_DEG = 60.0  # 占位:垂直视场角,需替换为真实相机参数


def _build_intrinsics() -> dict:
    fy = IMAGE_HEIGHT / (2 * np.tan(np.radians(FOV_Y_DEG) / 2))
    fx = fy  # 假设像素为正方形,无畸变
    return {"fx": fx, "fy": fy, "cx": IMAGE_WIDTH / 2, "cy": IMAGE_HEIGHT / 2}


INTRINSICS = _build_intrinsics()

# 相机安装在机器人头部,相对 base_link 的位置(单位:米,占位值)
CAMERA_POSITION_WRT_BASE = np.array([0.30, 0.0, 1.20])

# 相机俯仰角(向下看的角度,占位值)
CAMERA_PITCH_DEG = 30.0


def _build_camera_rotation(pitch_deg: float) -> np.ndarray:
    """构造相机坐标系到 base_link 坐标系的旋转矩阵。

    相机水平(pitch=0)时:相机 x(右)= -base_y,相机 y(下)= -base_z,
    相机 z(前)= base_x。俯仰角 pitch 表示相机在自身坐标系下绕 x 轴(右)
    向下旋转,使光轴逐渐指向 base 的 -z 方向。
    """
    r0 = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    theta = np.radians(pitch_deg)
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), np.sin(theta)],
            [0.0, -np.sin(theta), np.cos(theta)],
        ]
    )
    return r0 @ r_pitch


CAMERA_ROTATION_WRT_BASE = _build_camera_rotation(CAMERA_PITCH_DEG)


def pixel_depth_to_base_point(u: float, v: float, depth: float) -> np.ndarray:
    """像素坐标 (u, v) + 深度值(米) -> 机器人 base_link 坐标系下的三维坐标。

    depth 定义为该像素点到相机的直线距离沿相机光轴(z)方向的分量,
    即标准针孔相机的深度图约定(与 obs["depth"] 的语义一致)。
    """
    fx, fy, cx, cy = (INTRINSICS[k] for k in ("fx", "fy", "cx", "cy"))
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    point_cam = np.array([x_cam, y_cam, z_cam])
    return CAMERA_ROTATION_WRT_BASE @ point_cam + CAMERA_POSITION_WRT_BASE


def project_point_to_pixel(point_base: np.ndarray) -> tuple:
    """三维坐标(base_link 系) -> 像素坐标 + 深度。仅用于自检/单元测试(正向投影)。"""
    point_cam = CAMERA_ROTATION_WRT_BASE.T @ (np.asarray(point_base) - CAMERA_POSITION_WRT_BASE)
    x_cam, y_cam, z_cam = point_cam
    fx, fy, cx, cy = (INTRINSICS[k] for k in ("fx", "fy", "cx", "cy"))
    u = x_cam * fx / z_cam + cx
    v = y_cam * fy / z_cam + cy
    return u, v, z_cam


if __name__ == "__main__":
    # 自检:任取一个 base 系三维点,正向投影再反向还原,应能拿回原始坐标
    test_point = np.array([0.9, -0.15, 0.85])
    u, v, depth = project_point_to_pixel(test_point)
    recovered = pixel_depth_to_base_point(u, v, depth)
    print(f"原始点: {test_point}")
    print(f"投影像素: ({u:.1f}, {v:.1f}), 深度: {depth:.3f}")
    print(f"反向还原: {recovered}")
    assert np.allclose(test_point, recovered, atol=1e-6), "往返转换不一致!"
    print("[OK] 正向投影 / 反向还原 往返一致")
