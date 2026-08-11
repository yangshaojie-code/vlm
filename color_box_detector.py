"""Small dependency-free baseline detector for the three colored boxes.

It is intended for offline development and as a deterministic fallback. The
formal container can later replace it with a YOLO node without changing the
mission protocol or motion code.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from mission_protocol import normalize_color


@dataclass(frozen=True)
class BoxDetection:
    color: str
    bbox: tuple
    center: tuple
    area: int
    confidence: float

    def as_dict(self) -> dict:
        return {
            "color": self.color,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "area": self.area,
            "confidence": self.confidence,
        }


def _rgb_array(image) -> np.ndarray:
    if isinstance(image, (str, bytes)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"image 必须是 HxWx3 RGB 数组，收到 shape={array.shape}")
    return array[..., :3].astype(np.uint8)


def _mask(rgb: np.ndarray, color: str) -> np.ndarray:
    r, g, b = [rgb[..., i].astype(np.int16) for i in range(3)]
    if color == "pink":
        return (r > 150) & (b > 90) & (r > g + 35) & (b > g - 20)
    if color == "yellow":
        return (r > 160) & (g > 125) & (b < 150) & (r - b > 60) & (g - b > 35)
    if color == "brown":
        return (r > 45) & (r > g + 20) & (g > b + 12) & (b < 135) & (r < 210)
    raise ValueError(f"不支持的颜色: {color}")


def _components(mask: np.ndarray, min_area: int) -> Iterable[tuple]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        count = 0
        x_min = x_max = int(x)
        y_min = y_max = int(y)
        while stack:
            cy, cx = stack.pop()
            count += 1
            x_min, x_max = min(x_min, cx), max(x_max, cx)
            y_min, y_max = min(y_min, cy), max(y_max, cy)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if count >= min_area:
            yield x_min, y_min, x_max + 1, y_max + 1, count


def detect_colored_boxes(image, color: Optional[str] = None, min_area: int = 80) -> List[BoxDetection]:
    rgb = _rgb_array(image)
    colors = [normalize_color(color)] if color is not None else ["pink", "yellow", "brown"]
    detections = []
    for name in colors:
        for x1, y1, x2, y2, area in _components(_mask(rgb, name), min_area):
            box_area = max(1, (x2 - x1) * (y2 - y1))
            confidence = min(1.0, area / box_area * 2.0)
            detections.append(BoxDetection(name, (x1, y1, x2, y2), ((x1 + x2) // 2, (y1 + y2) // 2), area, confidence))
    return sorted(detections, key=lambda item: (item.color, -item.area))
