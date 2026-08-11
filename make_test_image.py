"""生成合成测试图:桌面上摆放粉色/黄色/棕色包装盒和一个圆形工具桶。

拿到仿真平台后,应换成机器人相机截图重新验证。
运行: python make_test_image.py
"""

import os

from PIL import Image, ImageDraw

W, H = 1280, 720


def draw_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, depth: int, color, dark):
    """画一个带简单透视的长方体:正面 + 顶面 + 侧面。"""
    draw.polygon([(x, y), (x + depth, y - depth), (x + w + depth, y - depth), (x + w, y)], fill=dark)
    draw.polygon(
        [(x + w, y), (x + w + depth, y - depth), (x + w + depth, y - depth + h), (x + w, y + h)],
        fill=dark,
    )
    draw.rectangle([x, y, x + w, y + h], fill=color, outline=(60, 60, 60), width=2)


def main():
    img = Image.new("RGB", (W, H), (225, 228, 232))  # 墙面
    draw = ImageDraw.Draw(img)

    # 桌面
    draw.rectangle([0, 420, W, H], fill=(150, 111, 71))
    draw.rectangle([0, 420, W, 445], fill=(126, 90, 55))

    # 粉色包装盒
    draw_box(draw, 140, 460, 220, 150, 40, (240, 130, 170), (205, 95, 140))
    # 黄色包装盒
    draw_box(draw, 480, 490, 200, 140, 36, (245, 205, 65), (205, 165, 40))
    # 棕色包装盒
    draw_box(draw, 790, 470, 210, 145, 38, (150, 100, 60), (115, 72, 40))

    # 圆形工具桶(灰色圆柱)
    bx, by, bw, bh = 1060, 430, 150, 210
    draw.rectangle([bx, by + 30, bx + bw, by + bh], fill=(120, 125, 135))
    draw.ellipse([bx, by + bh - 30, bx + bw, by + bh + 30], fill=(105, 110, 120))
    draw.ellipse([bx, by, bx + bw, by + 60], fill=(160, 165, 175), outline=(80, 85, 95), width=3)

    os.makedirs("test_images", exist_ok=True)
    out = os.path.join("test_images", "boxes.jpg")
    img.save(out, quality=92)
    print(f"已生成测试图: {out}")


if __name__ == "__main__":
    main()
