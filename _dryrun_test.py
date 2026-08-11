"""无 Key 干跑检查:验证纯逻辑部分正确,API 调用处报出清晰错误。"""

import os
import sys

os.environ.pop("DASHSCOPE_API_KEY", None)

from grounding import draw_result, norm_bbox_to_pixel, object_to_description
from task_parser import extract_json

# 1. JSON 提取
assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
assert extract_json('前缀 {"found": true, "bbox": [1, 2, 3, 4]} 后缀') == {
    "found": True,
    "bbox": [1, 2, 3, 4],
}
print("[OK] extract_json")

# 2. 物体描述映射
assert object_to_description({"category": "packing_box", "color": "pink", "shape": "cuboid"}) == "粉色的长方体包装盒"
assert object_to_description({"category": "tool_bucket"}) == "工具桶"
assert object_to_description({"category": "material_box", "shape": "square"}) == "方形物料盒"
print("[OK] object_to_description")

# 3. 归一化坐标 -> 像素坐标换算(qwen3.7 系列输出 [0, 999] 相对坐标)
assert norm_bbox_to_pixel([0, 0, 999, 999], 1280, 720) == [0, 0, 1279, 719]
assert norm_bbox_to_pixel([100, 500, 250, 800], 1000, 500) == [100, 250, 250, 400]
assert norm_bbox_to_pixel([250, 800, 100, 500], 1000, 500) == [100, 250, 250, 400]  # 容忍坐标乱序
print("[OK] norm_bbox_to_pixel")

# 4. 画框保存
img_path = os.path.join("test_images", "boxes.jpg")
result = {"found": True, "bbox": [110, 365, 320, 490], "center": [215, 427]}
out = draw_result(img_path, result, "粉色的长方体包装盒")
assert os.path.exists(out)
print(f"[OK] draw_result -> {out}")

# 5. 无 Key 时 API 调用应给出清晰报错
import config

try:
    config.get_client()
    print("[FAIL] 未设置 Key 却没有报错")
    sys.exit(1)
except RuntimeError as e:
    assert "DASHSCOPE_API_KEY" in str(e)
    print("[OK] 无 Key 报错清晰,错误发生在 API 调用处而非代码逻辑")

print("\n全部干跑检查通过")
