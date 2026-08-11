"""Generate and optionally serve a visual report for the VLM pick/place pipeline.

Examples:
  python3 visualize_pipeline.py
  python3 visualize_pipeline.py --image test_images/grounding_cases/clear_three_boxes.png
  python3 visualize_pipeline.py --depth-map frame_depth.npy --serve --port 8000
"""

import argparse
import base64
import html
import io
import json
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from client_common import TABLE_PLACE_POSITION_BASE
from depth_utils import robust_depth_from_bbox
from grasp_pose import pick_pose, place_pose, place_pose_on_table
from grounding import locate_object, object_to_description
from pick_place_task import PickPlaceTask
from task_parser import parse_instruction

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "test_images" / "boxes.jpg"
DEFAULT_OUTPUT = ROOT / "outputs" / "pipeline_visualization"
DEFAULT_INSTRUCTION = "找到一个粉色长方体包装盒,放到圆形工具桶左边"


def _font(size):
    for name in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _draw_box(draw, result, label, color, width):
    if not result or not result["found"]:
        return
    x1, y1, x2, y2 = result["bbox"]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    cx, cy = result["center"]
    radius = max(4, width * 2)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    font = _font(max(15, width * 5))
    text_y = max(2, y1 - font.size - 7) if hasattr(font, "size") else max(2, y1 - 22)
    draw.text((x1, text_y), label, fill=color, font=font, stroke_width=1, stroke_fill="white")


def _draw_direction(draw, reference_result, direction, color, width):
    if not reference_result or not reference_result["found"] or direction not in ("left", "right"):
        return
    cx, cy = reference_result["center"]
    distance = max(70, (reference_result["bbox"][2] - reference_result["bbox"][0]) // 2 + 45)
    sign = -1 if direction == "left" else 1  # image x: left is negative
    end_x = cx + sign * distance
    draw.line([cx, cy, end_x, cy], fill=color, width=width)
    head = max(10, width * 3)
    draw.polygon([(end_x, cy), (end_x - sign * head, cy - head // 2), (end_x - sign * head, cy + head // 2)], fill=color)
    label = "放置:左边" if direction == "left" else "放置:右边"
    draw.text((min(cx, end_x), cy + 10), label, fill=color, font=_font(max(15, width * 5)), stroke_width=1, stroke_fill="white")


def _load_depth(path, shape, fallback_depth):
    if path:
        depth = np.load(path)
        if depth.shape != shape:
            raise ValueError(f"深度图 shape={depth.shape} 与图片 shape={shape} 不一致")
        return depth, f"深度图: {path}"
    return np.full(shape, fallback_depth, dtype=float), f"演示固定深度: {fallback_depth:.3f} m"


def build_report(instruction, image_path, output_dir, depth_map_path=None, fallback_depth=0.7):
    output_dir.mkdir(parents=True, exist_ok=True)
    original = Image.open(image_path).convert("RGB")
    depth_map, depth_source = _load_depth(depth_map_path, (original.height, original.width), fallback_depth)

    print("[1/5] 解析指令...")
    task = parse_instruction(instruction)
    target_desc = object_to_description(task["target_object"])

    print(f"[2/5] 定位目标: {target_desc}")
    target_result = locate_object(original, target_desc)
    if not target_result["found"]:
        raise RuntimeError(f"未找到目标: {target_desc}")

    reference_result = None
    reference_desc = None
    if task.get("reference_object"):
        reference_desc = object_to_description(task["reference_object"])
        print(f"[3/5] 定位参照物: {reference_desc}")
        reference_result = locate_object(original, reference_desc)
        if not reference_result["found"]:
            raise RuntimeError(f"未找到参照物: {reference_desc}")
    else:
        print("[3/5] 任务无参照物，使用桌面放置点")

    target_depth = robust_depth_from_bbox(depth_map, target_result["bbox"])
    pick = pick_pose(target_result["center"], target_depth)
    reference_depth = None
    if reference_result:
        reference_depth = robust_depth_from_bbox(depth_map, reference_result["bbox"])
        place = place_pose(
            reference_result["center"],
            reference_depth,
            task["direction"],
            reference_category=task["reference_object"]["category"],
        )
    else:
        place = place_pose_on_table(TABLE_PLACE_POSITION_BASE)

    print("[4/5] 绘制定位和放置语义...")
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    line_width = max(3, original.width // 260)
    _draw_box(draw, target_result, f"目标: {target_desc}", (218, 48, 74), line_width)
    _draw_box(draw, reference_result, f"参照: {reference_desc}", (20, 120, 210), line_width)
    _draw_direction(draw, reference_result, task.get("direction"), (20, 155, 90), line_width)
    annotated_path = output_dir / "annotated_scene.png"
    annotated.save(annotated_path)

    plan = {
        "target_description": target_desc,
        "target_grounding": target_result,
        "target_depth_m": target_depth,
        "pick_pose": pick,
        "reference_description": reference_desc,
        "reference_grounding": reference_result,
        "reference_depth_m": reference_depth,
        "place_pose": place,
        "states": PickPlaceTask.STATES,
        "depth_source": depth_source,
        "warning": "相机内外参仍为占位值，XYZ 仅用于流程演示" if not depth_map_path else "请确认深度与 RGB 对齐，且相机内外参已替换",
    }
    serializable_plan = _jsonable(plan)
    with open(output_dir / "plan.json", "w", encoding="utf-8") as file:
        json.dump({"instruction": instruction, "task": task, "plan": serializable_plan}, file, ensure_ascii=False, indent=2)

    stage_names = ["指令解析", "目标定位", "深度反投影", "抓取/放置位姿", "状态机执行"]
    stage_html = "".join(f'<div class="stage"><span>{i}</span><b>{html.escape(name)}</b></div>' for i, name in enumerate(stage_names, 1))
    task_json = html.escape(json.dumps(task, ensure_ascii=False, indent=2))
    plan_json = html.escape(json.dumps(serializable_plan, ensure_ascii=False, indent=2))
    original_url, annotated_url = _data_url(original), _data_url(annotated)
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>文旅机器人视觉流程</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#f4f5f6;color:#202326;font:14px Arial,"Microsoft YaHei",sans-serif;letter-spacing:0}}
header{{background:#202326;color:white;padding:18px 28px;border-bottom:4px solid #d9304f}} header h1{{margin:0 0 6px;font-size:22px}} header p{{margin:0;color:#cdd2d6}}
main{{max-width:1440px;margin:auto;padding:20px}} .pipeline{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:16px}}
.stage{{background:white;border:1px solid #d7dadd;padding:12px;display:flex;align-items:center;gap:9px;border-radius:6px}} .stage span{{display:grid;place-items:center;width:25px;height:25px;border-radius:50%;background:#202326;color:white}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} section{{background:white;border:1px solid #d7dadd;border-radius:6px;padding:15px}} h2{{font-size:16px;margin:0 0 12px}} img{{width:100%;height:auto;display:block;background:#ddd}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:12px 0}} .metric{{border-left:3px solid #16865c;background:#f5f8f7;padding:10px}} .metric b{{display:block;font-size:18px;margin-top:4px}}
pre{{white-space:pre-wrap;word-break:break-word;background:#f4f5f6;border:1px solid #e1e3e5;padding:12px;max-height:420px;overflow:auto;font-size:12px}} .warning{{background:#fff6df;border-left:4px solid #d99b16;padding:10px;margin-top:12px}}
@media(max-width:800px){{.pipeline,.grid,.metrics{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>文旅机器人视觉抓放流程</h1><p>{html.escape(instruction)}</p></header><main>
<div class="pipeline">{stage_html}</div><div class="grid">
<section><h2>原始场景</h2><img src="{original_url}" alt="原始场景"></section>
<section><h2>Grounding 与放置方向</h2><img src="{annotated_url}" alt="定位结果"></section>
<section><h2>结构化任务</h2><pre>{task_json}</pre></section>
<section><h2>三维规划结果</h2><div class="metrics"><div class="metric">目标深度<b>{target_depth:.3f} m</b></div><div class="metric">抓取 XYZ<b>{', '.join(f'{v:.3f}' for v in pick['object_position'])}</b></div><div class="metric">放置 XYZ<b>{', '.join(f'{v:.3f}' for v in place['position'])}</b></div></div><pre>{plan_json}</pre><div class="warning">{html.escape(serializable_plan['warning'])}<br>{html.escape(depth_source)}</div></section>
</div></main></body></html>"""
    report_path = output_dir / "index.html"
    report_path.write_text(report, encoding="utf-8")
    print(f"[5/5] 可视化报告: {report_path}")
    return report_path


def serve(directory, host, port, open_browser):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}/"
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving {directory} at {url} (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="生成 VLM 抓放流程可视化报告")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--depth-map", type=Path, help="与 RGB 对齐的二维 .npy 深度图")
    parser.add_argument("--fallback-depth", type=float, default=0.7, help="无深度图时的演示深度")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--serve", action="store_true", help="生成后启动 HTTP 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", help="服务启动后尝试打开浏览器")
    args = parser.parse_args()
    if not args.image.exists():
        parser.error(f"图片不存在: {args.image}")
    report_path = build_report(args.instruction, args.image, args.output, args.depth_map, args.fallback_depth)
    if args.serve:
        serve(report_path.parent, args.host, args.port, args.open)


if __name__ == "__main__":
    main()
