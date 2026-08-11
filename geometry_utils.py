"""Pure NumPy geometry helpers for ROS TF and placement calculations."""

import math
from typing import Iterable

import numpy as np


def transform_point(matrix, point) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    point = np.asarray(point, dtype=float)
    if matrix.shape != (4, 4) or point.shape != (3,):
        raise ValueError("matrix 必须为 4x4，point 必须为长度 3")
    return (matrix @ np.r_[point, 1.0])[:3]


def inverse_transform(matrix) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("matrix 必须为 4x4")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def shelf_surface_height(layer: int) -> float:
    heights = {1: 0.403, 2: 0.732, 3: 1.061}
    try:
        return heights[int(layer)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("正式货架层数必须是从下往上的 1、2、3 层") from exc


def box_center_height(layer: int, box_height: float = 0.19) -> float:
    return shelf_surface_height(layer) + float(box_height) / 2.0

