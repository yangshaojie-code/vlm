"""① 指令解析:自然语言指令 -> 结构化任务 JSON。

对应赛题两类指令:
- 任务一:"在场景内找到长方体包装盒放到桌子上"(无颜色、无参照物)
- 任务二:"找到一个[颜色]的包装盒,并将其放到[指定道具]的[指定方向]"
"""

import json
import re

import config

# 统一的任务 schema,见 few-shot 示例。
# 按官方文档建议,qwen3.7 系列在通用场景不设 System Message,指令随 User Message 传入。
PARSER_PROMPT = """你是一个机器人任务解析器。将用户的自然语言指令解析为 JSON,只输出 JSON,不要输出任何其他文字。

JSON 格式如下:
{
  "task_type": "pick_and_place",
  "target_object": {"category": <物体类别>, "color": <颜色或null>, "shape": <形状或null>},
  "reference_object": {"category": <参照物类别>} 或 null,
  "direction": "left" / "right" / null,
  "place_target": <放置位置类别或null>,
  "raw_instruction": <原始指令>
}

字段取值约定:
- category 可选: "packing_box"(包装盒), "material_box"(物料盒), "tool_bucket"(工具桶), "table"(桌子), "shelf"(货架)
- color 可选: "pink"(粉色), "yellow"(黄色), "brown"(棕色), 未指定时为 null
- shape 可选: "cuboid"(长方体), "square"(方形), "round"(圆形), 未指定时为 null
- direction 定义在接收指令时的初始任务坐标系中,不随机器人后续搜索转向改变; "left"=初始左边, "right"=初始右边,未指定时为 null
- reference_object: 放置时的参照道具, 没有参照物时为 null
- place_target: 直接放置的目标位置(如桌子), 有参照物时为 null

示例 1:
指令: 在场景内找到长方体包装盒放到桌子上
输出:
{"task_type": "pick_and_place", "target_object": {"category": "packing_box", "color": null, "shape": "cuboid"}, "reference_object": null, "direction": null, "place_target": "table", "raw_instruction": "在场景内找到长方体包装盒放到桌子上"}

示例 2:
指令: 找到一个粉色长方体包装盒,放到圆形工具桶左边
输出:
{"task_type": "pick_and_place", "target_object": {"category": "packing_box", "color": "pink", "shape": "cuboid"}, "reference_object": {"category": "tool_bucket"}, "direction": "left", "place_target": null, "raw_instruction": "找到一个粉色长方体包装盒,放到圆形工具桶左边"}

示例 3:
指令: 把黄色包装盒放到方形物料盒右边
输出:
{"task_type": "pick_and_place", "target_object": {"category": "packing_box", "color": "yellow", "shape": null}, "reference_object": {"category": "material_box"}, "direction": "right", "place_target": null, "raw_instruction": "把黄色包装盒放到方形物料盒右边"}
"""


def extract_json(text: str) -> dict:
    """从模型回复中提取 JSON(容忍 ```json ... ``` 代码块包裹)。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # 截取第一个 { 到最后一个 } 之间的内容
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"模型回复中未找到 JSON: {text!r}")
    return json.loads(text[start : end + 1])


def parse_instruction(instruction: str) -> dict:
    """调用 Qwen-VL 将指令解析为结构化任务 JSON。"""
    client = config.get_client()
    resp = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "user", "content": f"{PARSER_PROMPT}\n\n指令: {instruction}"},
        ],
        temperature=0.0,
        extra_body=config.EXTRA_BODY,
    )
    return extract_json(resp.choices[0].message.content)


if __name__ == "__main__":
    import sys

    instruction = sys.argv[1] if len(sys.argv) > 1 else "找到一个粉色长方体包装盒,放到圆形工具桶左边"
    task = parse_instruction(instruction)
    print(json.dumps(task, ensure_ascii=False, indent=2))
