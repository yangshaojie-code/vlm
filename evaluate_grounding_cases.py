"""Run Qwen grounding over generated cases and report detection metrics."""

import json
import os
from collections import defaultdict

from PIL import Image

from grounding import draw_result, locate_object


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def main():
    root = os.path.join(os.path.dirname(__file__), "test_images", "grounding_cases")
    annotation_path = os.path.join(root, "annotations.json")
    if not os.path.exists(annotation_path):
        raise FileNotFoundError("请先运行 python make_grounding_cases.py")

    with open(annotation_path, encoding="utf-8") as file:
        annotations = json.load(file)

    grouped = defaultdict(list)
    for item in annotations:
        grouped[(item["image"], item["description"])].append(item["bbox"])

    output_dir = os.path.join(os.path.dirname(__file__), "outputs", "grounding_eval")
    rows = []
    for index, ((image_name, description), truth_boxes) in enumerate(grouped.items(), 1):
        image_path = os.path.join(root, image_name)
        prediction = locate_object(image_path, description)
        iou = 0.0
        if prediction["found"]:
            iou = max(bbox_iou(prediction["bbox"], truth) for truth in truth_boxes)
            draw_result(
                Image.open(image_path),
                prediction,
                description,
                output_dir=output_dir,
                name_hint=f"case_{index:02d}",
            )
        rows.append({
            "image": image_name,
            "description": description,
            "found": prediction["found"],
            "predicted_bbox": prediction["bbox"],
            "truth_bboxes": truth_boxes,
            "max_iou": round(iou, 4),
        })
        print(f"[{index:02d}/{len(grouped)}] found={prediction['found']} IoU={iou:.3f} {image_name} | {description}")

    found_rate = sum(row["found"] for row in rows) / len(rows)
    mean_iou = sum(row["max_iou"] for row in rows) / len(rows)
    iou50_rate = sum(row["max_iou"] >= 0.5 for row in rows) / len(rows)
    summary = {
        "case_count": len(rows),
        "found_rate": round(found_rate, 4),
        "mean_iou": round(mean_iou, 4),
        "iou_at_0.5_rate": round(iou50_rate, 4),
        "cases": rows,
    }
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
