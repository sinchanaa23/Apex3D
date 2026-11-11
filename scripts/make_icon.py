"""Draws the Apex3D icon (the logo's orange curve on a dark square) to scripts/apex3d.ico.

    python scripts/make_icon.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 256
SCALE = 4  # draw big, then shrink, for smooth edges
OUT = Path(__file__).with_name("apex3d.ico")


def bezier(points, steps=60):
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    for i in range(steps + 1):
        t = i / steps
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t**2, t**3
        yield (a * x0 + b * x1 + c * x2 + d * x3, a * y0 + b * y1 + c * y2 + d * y3)


def main() -> None:
    big = SIZE * SCALE
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([0, 0, big - 1, big - 1], radius=big // 5, fill=(14, 14, 15, 255))
    unit = big / 32  # the logo is drawn on a 32 by 32 grid: M5 25 C13 25 13 7 18 7 s4 9 9 11
    points = [(x * unit, y * unit) for x, y in bezier([(5, 25), (13, 25), (13, 7), (18, 7)], 240)] + [(x * unit, y * unit) for x, y in bezier([(18, 7), (22, 7), (22, 16), (27, 18)], 240)]
    r = 1.5 * unit  # stamp round dots along the curve: a smooth, round-ended line
    for x, y in points:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 106, 43, 255))
    image = image.resize((SIZE, SIZE), Image.LANCZOS)
    image.save(OUT, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
