"""Small dependency-free detector for the three colored competition boxes.

It is a deterministic fallback, not a replacement for a trained detector.
The filters intentionally reject scene-scale regions: the wooden tabletop is
brown/yellow enough to pass broad RGB thresholds but cannot be one box.
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
    """Return an HSV mask tuned to the rendered competition materials."""
    values = rgb.astype(np.float32) / 255.0
    r, g, b = values[..., 0], values[..., 1], values[..., 2]
    maximum = np.max(values, axis=2)
    minimum = np.min(values, axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    valid = delta > 1e-6
    red = valid & (maximum == r)
    green = valid & (maximum == g)
    blue = valid & (maximum == b)
    hue[red] = ((g[red] - b[red]) / delta[red]) % 6.0
    hue[green] = (b[green] - r[green]) / delta[green] + 2.0
    hue[blue] = (r[blue] - g[blue]) / delta[blue] + 4.0
    hue *= 60.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-6)
    if color == "pink":
        return (hue >= 320.0) & (hue <= 355.0) & (saturation >= 0.30) & (maximum >= 0.45)
    if color == "yellow":
        # The table is orange-brown (about 25-35 degrees), while the yellow
        # box is near 50 degrees.  The hue lower bound is the critical split.
        return (hue >= 42.0) & (hue <= 70.0) & (saturation >= 0.42) & (maximum >= 0.55)
    if color == "brown":
        # Require stronger chroma than the wooden tabletop.  The fixed-layout
        # shelf box has a distinctly more saturated brown front face.
        return (hue >= 12.0) & (hue <= 40.0) & (saturation >= 0.53) & (maximum >= 0.22) & (maximum <= 0.82)
    if color == "white":
        # Packaging box: bright, low-chroma.  Shelf lighting is dimmer than
        # a synthetic 240-gray patch, so keep a looser value floor.
        return (saturation <= 0.32) & (maximum >= 0.55)
    raise ValueError(f"不支持的颜色: {color}")


def _components(mask: np.ndarray, min_area: int, max_area: int, max_bbox_area: int, min_span: int) -> Iterable[tuple]:
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
        bbox_area = (x_max + 1 - x_min) * (y_max + 1 - y_min)
        width, height = x_max + 1 - x_min, y_max + 1 - y_min
        if min_area <= count <= max_area and bbox_area <= max_bbox_area and width >= min_span and height >= min_span:
            yield x_min, y_min, x_max + 1, y_max + 1, count


def detect_colored_boxes(
    image,
    color: Optional[str] = None,
    min_area: int = 80,
    max_area_frac: float = 0.06,
    max_bbox_frac: float = 0.10,
) -> List[BoxDetection]:
    rgb = _rgb_array(image)
    colors = [normalize_color(color)] if color is not None else ["pink", "yellow", "brown"]
    image_area = rgb.shape[0] * rgb.shape[1]
    # A rendered box occupies a compact region.  The values are intentionally
    # generous for close views but eliminate table/wall-sized false positives.
    max_area = max(min_area, int(image_area * float(max_area_frac)))
    max_bbox_area = max(min_area, int(image_area * float(max_bbox_frac)))
    min_span = 12
    detections = []
    for name in colors:
        for x1, y1, x2, y2, area in _components(_mask(rgb, name), min_area, max_area, max_bbox_area, min_span):
            box_area = max(1, (x2 - x1) * (y2 - y1))
            confidence = min(1.0, area / box_area * 2.0)
            detections.append(BoxDetection(name, (x1, y1, x2, y2), ((x1 + x2) // 2, (y1 + y2) // 2), area, confidence))
    return sorted(detections, key=lambda item: (item.color, -item.area))
