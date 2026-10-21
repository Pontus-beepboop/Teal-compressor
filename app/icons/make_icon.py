"""Draws the app icon (a blue squircle with a 2x2 Bayer quad being squeezed).

    python3 app/qt/make_icon.py app/qt/icons   ->  AppIcon.iconset/, AppIcon.ico (and AppIcon.icns on macOS)
"""
import os
import sys

from PIL import Image, ImageDraw


def draw(size):
    s = size / 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Background: vertical blue gradient inside a rounded square.
    grad = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / max(1, size - 1)
        gd.line([(0, y), (size, y)], fill=(int(30 + 20 * t), int(110 - 50 * t), int(235 - 60 * t), 255))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([100 * s, 100 * s, 924 * s, 924 * s], radius=190 * s, fill=255)
    img.paste(grad, (0, 0), mask)
    # Bayer quad (G R / B G), slightly narrowed to suggest compression.
    cell, gap = 170 * s, 22 * s
    x0, y0 = 512 * s - cell - gap / 2, 512 * s - cell - gap / 2
    colors = [(92, 214, 120), (240, 90, 80), (80, 140, 250), (92, 214, 120)]
    for i, c in enumerate(colors):
        cx, cy = x0 + (i % 2) * (cell + gap), y0 + (i // 2) * (cell + gap)
        d.rounded_rectangle([cx, cy, cx + cell, cy + cell], radius=28 * s, fill=c + (255,),
                            outline=(255, 255, 255, 230), width=max(1, int(10 * s)))
    # Inward arrows on both sides.
    for side in (-1, 1):
        tip = 512 * s + side * (cell + gap / 2 + 40 * s)
        base = tip + side * 150 * s
        d.polygon([(tip, 512 * s), (base, 430 * s), (base, 594 * s)], fill=(255, 255, 255, 235))
    return img


def main(out):
    iconset = os.path.join(out, "AppIcon.iconset")
    os.makedirs(iconset, exist_ok=True)
    for pt in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = pt * scale
            name = f"icon_{pt}x{pt}{'@2x' if scale == 2 else ''}.png"
            draw(px).save(os.path.join(iconset, name))
    draw(256).save(os.path.join(out, "AppIcon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    draw(512).save(os.path.join(out, "AppIcon.png"))
    if sys.platform == "darwin":
        os.system(f'iconutil -c icns "{iconset}" -o "{os.path.join(out, "AppIcon.icns")}"')


if __name__ == "__main__":
    main(sys.argv[1])
