"""Generate deterministic grounding cases and ground-truth annotations."""

import json
import os

from PIL import Image, ImageDraw

W, H = 960, 540
COLORS = {
    "pink": ((238, 125, 171), (190, 78, 126)),
    "yellow": ((240, 202, 60), (190, 151, 32)),
    "brown": ((145, 94, 55), (102, 62, 34)),
}


def draw_box(draw, bbox, color_name):
    x1, y1, x2, y2 = bbox
    depth = max(12, (x2 - x1) // 7)
    face, dark = COLORS[color_name]
    draw.polygon([(x1, y1), (x1 + depth, y1 - depth), (x2 + depth, y1 - depth), (x2, y1)], fill=dark)
    draw.polygon([(x2, y1), (x2 + depth, y1 - depth), (x2 + depth, y2 - depth), (x2, y2)], fill=dark)
    draw.rectangle(bbox, fill=face, outline=(45, 45, 45), width=3)


def draw_bucket(draw, bbox):
    x1, y1, x2, y2 = bbox
    lip = max(10, (y2 - y1) // 7)
    draw.rectangle([x1, y1 + lip // 2, x2, y2 - lip // 2], fill=(115, 125, 135))
    draw.ellipse([x1, y1, x2, y1 + lip], fill=(165, 174, 181), outline=(65, 70, 75), width=3)
    draw.ellipse([x1, y2 - lip, x2, y2], fill=(95, 105, 115))


def background():
    image = Image.new("RGB", (W, H), (220, 225, 229))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 300, W, H], fill=(137, 105, 75))
    draw.rectangle([0, 300, W, 318], fill=(102, 76, 54))
    return image, draw


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "test_images", "grounding_cases")
    os.makedirs(out_dir, exist_ok=True)
    cases = []

    layouts = [
        ("clear_three_boxes", [("pink", [80, 340, 230, 455]), ("yellow", [365, 350, 520, 465]), ("brown", [650, 335, 810, 455])], [790, 305, 910, 485]),
        ("target_at_edge", [("pink", [8, 350, 125, 445]), ("yellow", [430, 345, 590, 465]), ("brown", [690, 350, 845, 465])], [230, 315, 355, 490]),
        ("small_distant_target", [("pink", [420, 245, 485, 295]), ("yellow", [170, 355, 330, 470]), ("brown", [620, 345, 785, 470])], [805, 320, 915, 485]),
        ("similar_distractors", [("pink", [110, 350, 255, 465]), ("pink", [385, 335, 535, 460]), ("yellow", [665, 350, 825, 465])], [15, 315, 100, 485]),
        ("partial_occlusion", [("pink", [190, 340, 360, 465]), ("yellow", [520, 345, 680, 465]), ("brown", [735, 350, 880, 465])], [330, 310, 470, 490]),
    ]

    for name, boxes, bucket in layouts:
        image, draw = background()
        for color, bbox in boxes:
            draw_box(draw, bbox, color)
        draw_bucket(draw, bucket)
        if name == "partial_occlusion":
            draw.rectangle([300, 395, 405, 490], fill=(92, 100, 108))
        path = os.path.join(out_dir, f"{name}.png")
        image.save(path)
        for color, bbox in boxes:
            cases.append({"image": os.path.basename(path), "description": f"{ {'pink':'粉色','yellow':'黄色','brown':'棕色'}[color] }的长方体包装盒", "bbox": bbox})
        cases.append({"image": os.path.basename(path), "description": "圆形工具桶", "bbox": bucket})

    with open(os.path.join(out_dir, "annotations.json"), "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)
    print(f"Generated {len(layouts)} images and {len(cases)} cases in {out_dir}")


if __name__ == "__main__":
    main()
