"""Robust depth sampling for VLM/detector bounding boxes."""

import numpy as np


class DepthSamplingError(ValueError):
    """Raised when a bounding box contains no reliable depth samples."""


def robust_depth_from_bbox(
    depth_map,
    bbox,
    center_ratio=0.5,
    min_depth=0.05,
    max_depth=10.0,
    min_samples=9,
):
    """Estimate object depth from the central region of a bounding box.

    Invalid values are removed first. A median/MAD filter then rejects isolated
    foreground/background values. The bbox center is used because VLM boxes often
    include substantial background around their edges.
    """
    depth = np.asarray(depth_map, dtype=float)
    if depth.ndim != 2:
        raise DepthSamplingError(f"depth_map 必须是二维数组，收到 shape={depth.shape}")
    if len(bbox) != 4:
        raise DepthSamplingError(f"bbox 必须包含 4 个值，收到 {bbox!r}")

    height, width = depth.shape
    x1, y1, x2, y2 = (float(value) for value in bbox)
    x1, x2 = sorted((np.clip(x1, 0, width), np.clip(x2, 0, width)))
    y1, y2 = sorted((np.clip(y1, 0, height), np.clip(y2, 0, height)))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise DepthSamplingError(f"bbox 裁剪后为空: {bbox!r}")

    ratio = float(np.clip(center_ratio, 0.1, 1.0))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * ratio / 2, (y2 - y1) * ratio / 2
    ix1, ix2 = int(np.floor(cx - half_w)), int(np.ceil(cx + half_w))
    iy1, iy2 = int(np.floor(cy - half_h)), int(np.ceil(cy + half_h))
    region = depth[max(0, iy1):min(height, iy2), max(0, ix1):min(width, ix2)]

    valid = region[np.isfinite(region) & (region > min_depth) & (region < max_depth)]
    if valid.size < min_samples:
        raise DepthSamplingError(
            f"bbox 中有效深度不足: {valid.size} < {min_samples}, bbox={bbox!r}"
        )

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    if mad > 1e-6:
        filtered = valid[np.abs(valid - median) <= 3.5 * 1.4826 * mad]
        if filtered.size >= min_samples:
            valid = filtered

    return float(np.median(valid))
