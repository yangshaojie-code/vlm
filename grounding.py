"""② 目标定位:图片 + 目标描述 -> 像素边界框和中心点。

利用 Qwen2.5-VL 原生的 grounding 能力,直接让模型输出目标在图片中的
像素坐标边界框 [x1, y1, x2, y2],无需额外的检测模型。
中心像素坐标是后续结合深度图反投影出三维坐标的输入。
"""

import base64
import io
import json
import mimetypes
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from task_parser import extract_json

# 结构化字段 -> 中文描述,用于拼接 grounding 提示词
COLOR_ZH = {"pink": "粉色", "yellow": "黄色", "brown": "棕色"}
SHAPE_ZH = {"cuboid": "长方体", "square": "方形", "round": "圆形"}
CATEGORY_ZH = {
    "packing_box": "包装盒",
    "material_box": "物料盒",
    "tool_bucket": "工具桶",
    "table": "桌子",
    "shelf": "货架",
}


def object_to_description(obj: dict) -> str:
    """把任务 JSON 中的物体字段转成中文描述,如 {'category':'packing_box','color':'pink','shape':'cuboid'} -> '粉色的长方体包装盒'。"""
    parts = []
    if obj.get("color"):
        parts.append(COLOR_ZH.get(obj["color"], obj["color"]) + "的")
    if obj.get("shape"):
        parts.append(SHAPE_ZH.get(obj["shape"], obj["shape"]))
    parts.append(CATEGORY_ZH.get(obj.get("category", ""), obj.get("category", "物体")))
    return "".join(parts)


def _load_image(image) -> Image.Image:
    """把 image_path(str)/PIL.Image/numpy 数组统一转成 PIL.Image。

    numpy 输入用于真实仿真场景:sim_node.step() 返回的 obs["img"] 是
    (H, W, 3) 的 RGB numpy 数组,不需要先存盘再读取。
    """
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype(np.uint8))
    return Image.open(image)


def _image_to_data_url(image) -> str:
    if isinstance(image, str):
        mime = mimetypes.guess_type(image)[0] or "image/jpeg"
        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"

    buf = io.BytesIO()
    _load_image(image).convert("RGB").save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def norm_bbox_to_pixel(bbox: list, width: int, height: int) -> list:
    """把 Qwen3 系列输出的归一化坐标(范围 [0, 999])换算成原图像素坐标。

    官方说明:Qwen3-VL/Qwen3.5/3.6/3.7 系列返回归一化到 [0, 999] 的相对坐标
    (Qwen2.5-VL 才是像素绝对坐标),需按 值/1000*边长 换算。
    """
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((x1 / 1000 * width, x2 / 1000 * width))
    y1, y2 = sorted((y1 / 1000 * height, y2 / 1000 * height))
    # 裁剪到图片范围内,防止越界
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    return [round(x1), round(y1), round(x2), round(y2)]


def locate_object(image, description: str) -> dict:
    """在图片中定位目标物体。

    image: 文件路径(str)、PIL.Image 或 numpy 数组((H,W,3) RGB,如真实仿真
           环境返回的 obs["img"])均可。

    返回: {"found": bool, "bbox": [x1,y1,x2,y2] 或 None, "center": [cx,cy] 或 None}
    坐标为原图像素坐标(已从模型的归一化坐标换算)。
    """
    width, height = _load_image(image).size

    # 改进的 prompt:明确归一化坐标的含义,并允许忽略未指定的颜色
    prompt = (
        f"请在图中找出:{description}。\n"
        f"注意:如果描述中没有指定颜色(如\"长方体包装盒\"),请忽略颜色,直接找符合形状和类别的物体。\n"
        "输出该物体的边界框,坐标使用相对坐标,范围 0-999(0=最左/最上,999=最右/最下)。\n"
        "只输出 JSON,不要输出任何其他文字。格式:\n"
        '{"found": true, "bbox": [x1, y1, x2, y2]}\n'
        '如果图中不存在该物体,输出 {"found": false, "bbox": null}'
    )

    client = config.get_client()
    resp = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.0,
        extra_body=config.EXTRA_BODY,
    )
    raw_reply = resp.choices[0].message.content
    result = extract_json(raw_reply)

    # 调试:打印 VLM 原始回复
    print(f"[Grounding] 描述: {description}, VLM 回复: {raw_reply[:200]}...")

    if not result.get("found") or not result.get("bbox"):
        return {"found": False, "bbox": None, "center": None}

    bbox = norm_bbox_to_pixel(result["bbox"], width, height)
    center = [round((bbox[0] + bbox[2]) / 2), round((bbox[1] + bbox[3]) / 2)]
    return {"found": True, "bbox": bbox, "center": center}


def _load_font(size: int):
    """优先加载支持中文的系统字体,失败则退回默认字体。"""
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_result(image, result: dict, label: str, output_dir: str = "outputs", name_hint: str = "frame") -> str:
    """在原图上画出边界框和中心点,保存到 outputs 目录,返回保存路径。

    image 同 locate_object,支持路径/PIL.Image/numpy 数组;非路径输入时用
    name_hint 作为输出文件名前缀(真实仿真环境没有现成文件名)。
    """
    os.makedirs(output_dir, exist_ok=True)
    img = _load_image(image).convert("RGB")
    draw = ImageDraw.Draw(img)

    if result["found"]:
        x1, y1, x2, y2 = result["bbox"]
        cx, cy = result["center"]
        line_w = max(2, img.width // 300)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_w)
        r = line_w * 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red")
        font_size = max(16, img.width // 50)
        draw.text((x1, max(0, y1 - font_size - 6)), label, fill="red", font=_load_font(font_size))

    stem = os.path.splitext(os.path.basename(image))[0] if isinstance(image, str) else name_hint
    out_path = os.path.join(output_dir, f"{stem}_annotated.png")
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys

    image = sys.argv[1] if len(sys.argv) > 1 else os.path.join("test_images", "boxes.jpg")
    desc = sys.argv[2] if len(sys.argv) > 2 else "粉色的长方体包装盒"
    res = locate_object(image, desc)
    print(json.dumps(res, ensure_ascii=False))
    if res["found"]:
        print("标注图:", draw_result(image, res, desc))
