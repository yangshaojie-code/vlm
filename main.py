"""流水线演示:指令解析(①) + 目标定位(②)。

用法(两个参数均可省略,默认用示例指令和合成测试图):
    python main.py --instruction "找到一个粉色长方体包装盒,放到圆形工具桶左边" --image test_images\\boxes.jpg
"""

import argparse
import json
import os

from grounding import draw_result, locate_object, object_to_description
from task_parser import parse_instruction

DEFAULT_INSTRUCTION = "找到一个粉色长方体包装盒,放到圆形工具桶左边"
DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images", "boxes.jpg")


def main():
    parser = argparse.ArgumentParser(description="Qwen-VL 视觉链路验证(指令解析 + 目标定位)")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="自然语言指令")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="场景图片路径")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        parser.error(f"图片不存在: {args.image}\n请先运行 python make_test_image.py 生成测试图,或用 --image 指定图片")

    print(f"指令: {args.instruction}")
    print(f"图片: {args.image}")

    print("=" * 60)
    print("① 指令解析")
    print("=" * 60)
    task = parse_instruction(args.instruction)
    print(json.dumps(task, ensure_ascii=False, indent=2))

    print()
    print("=" * 60)
    print("② 目标定位")
    print("=" * 60)
    desc = object_to_description(task["target_object"])
    print(f"定位目标: {desc}")
    result = locate_object(args.image, desc)
    print(json.dumps(result, ensure_ascii=False))

    if result["found"]:
        out_path = draw_result(args.image, result, desc)
        print(f"标注图已保存: {out_path}")
        print(f"中心像素坐标 {result['center']} 可用于后续结合深度图计算三维坐标")
    else:
        print("图中未找到目标物体")

    # 若指令包含参照物,顺带定位参照物,验证放置位置也能找到
    if task.get("reference_object"):
        ref_desc = object_to_description(task["reference_object"])
        print()
        print(f"参照物定位: {ref_desc} (方向: {task.get('direction')})")
        ref_result = locate_object(args.image, ref_desc)
        print(json.dumps(ref_result, ensure_ascii=False))


if __name__ == "__main__":
    main()
