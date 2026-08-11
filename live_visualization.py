"""Optional X11/OpenCV visualization for the native DISCOVERSE workflow."""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class LiveVisualizer:
    """Show perception and execution frames without affecting simulation control."""

    COLORS = {"target": (220, 48, 74), "reference": (20, 120, 210)}

    def __init__(self, enabled=True, output_dir=None):
        self.enabled = bool(enabled and os.environ.get("DISPLAY"))
        self.output_dir = Path(output_dir or Path(__file__).resolve().parent / "outputs" / "live")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        self._cv2 = None
        if self.enabled:
            try:
                import cv2

                self._cv2 = cv2
                cv2.namedWindow("VLM Perception", cv2.WINDOW_NORMAL)
                cv2.namedWindow("Robot Execution", cv2.WINDOW_NORMAL)
            except Exception as exc:
                self.enabled = False
                print(f"[可视化] OpenCV 窗口不可用，将只保存标注帧: {exc}")

    @staticmethod
    def _font(size):
        for name in (
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "msyh.ttc",
        ):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _as_rgb(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    def show_detection(self, image, result, description, role="target", direction=None):
        frame = self._as_rgb(image)
        draw = ImageDraw.Draw(frame)
        color = self.COLORS.get(role, (20, 155, 90))
        width = max(3, frame.width // 280)
        x1, y1, x2, y2 = result["bbox"]
        cx, cy = result["center"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        radius = width * 2
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
        prefix = "目标" if role == "target" else "参照物"
        label = f"{prefix}: {description}"
        font = self._font(max(16, frame.width // 55))
        draw.text((x1, max(2, y1 - 28)), label, fill=color, font=font, stroke_width=1, stroke_fill="white")

        if role == "reference" and direction in ("left", "right"):
            sign = -1 if direction == "left" else 1
            end_x = int(np.clip(cx + sign * max(70, (x2 - x1) // 2 + 40), 0, frame.width - 1))
            draw.line([cx, cy, end_x, cy], fill=(20, 155, 90), width=width)
            head = width * 4
            draw.polygon(
                [(end_x, cy), (end_x - sign * head, cy - head // 2), (end_x - sign * head, cy + head // 2)],
                fill=(20, 155, 90),
            )

        self._counter += 1
        path = self.output_dir / f"detection_{self._counter:02d}_{role}.png"
        frame.save(path)
        print(f"[可视化] 检测帧已保存: {path}")
        self._show("VLM Perception", frame)

    def show_execution(self, image, state, sim_time=None):
        if not self.enabled:
            return
        frame = self._as_rgb(image)
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, frame.width, 46], fill=(25, 28, 31))
        suffix = f" | t={sim_time:.2f}s" if sim_time is not None else ""
        draw.text((14, 11), f"State: {state}{suffix}", fill="white", font=self._font(20))
        self._show("Robot Execution", frame)

    def _show(self, title, frame):
        if not self.enabled:
            return
        array = np.asarray(frame)
        self._cv2.imshow(title, self._cv2.cvtColor(array, self._cv2.COLOR_RGB2BGR))
        self._cv2.waitKey(1)

    def close(self):
        if self.enabled and self._cv2 is not None:
            self._cv2.destroyAllWindows()
