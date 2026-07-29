#!/usr/bin/env python3
"""Generate the original CVio pixel-shrimp animation and source sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

NAVY = (7, 26, 43)
CYAN = (20, 184, 196)
TEAL = (15, 157, 138)
CORAL = (255, 107, 107)
SAND = (245, 233, 211)


def draw_shrimp(frame: Image.Image, x: int, y: int, contour: bool) -> None:
    draw = ImageDraw.Draw(frame)
    pixels = [
        (1, 2),
        (2, 1),
        (3, 1),
        (4, 2),
        (5, 2),
        (6, 3),
        (7, 3),
        (8, 4),
        (7, 5),
        (6, 5),
        (5, 6),
        (4, 6),
        (3, 5),
        (2, 5),
        (1, 4),
        (0, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (5, 4),
        (6, 4),
    ]
    scale = 5
    if contour:
        for px, py in pixels:
            draw.rectangle(
                (
                    x + px * scale - 2,
                    y + py * scale - 2,
                    x + (px + 1) * scale + 1,
                    y + (py + 1) * scale + 1,
                ),
                fill=CYAN,
            )
    for px, py in pixels:
        color = CORAL if (px + py) % 3 else SAND
        draw.rectangle(
            (
                x + px * scale,
                y + py * scale,
                x + (px + 1) * scale - 1,
                y + (py + 1) * scale - 1,
            ),
            fill=color,
        )
    draw.rectangle((x + 14, y + 9, x + 17, y + 12), fill=NAVY)
    draw.line((x + 39, y + 18, x + 49, y + 10), fill=CORAL, width=2)
    draw.line((x + 39, y + 20, x + 51, y + 22), fill=CORAL, width=2)


def build_frames() -> list[Image.Image]:
    frames: list[Image.Image] = []
    for index in range(36):
        frame = Image.new("RGB", (420, 120), NAVY)
        draw = ImageDraw.Draw(frame)
        for bubble in range(8):
            bx = (bubble * 61 + index * (1 + bubble % 2)) % 430
            by = 16 + (bubble * 29) % 88
            draw.rectangle((bx, by, bx + 2, by + 2), fill=(12, 89, 110))
        x = -15 + index * 15
        jump = -max(0, 18 - abs(index - 22) * 4)
        contour = 10 <= index <= 20
        labels = [("CLS", 245), ("SEG", 305), ("MOBILE", 356)]
        for label, lx in labels:
            fill = TEAL if x + 45 >= lx else (11, 79, 108)
            draw.rounded_rectangle(
                (lx, 79, lx + (52 if label != "MOBILE" else 61), 101), 4, fill=fill
            )
            draw.text((lx + 8, 84), label, fill=SAND)
        draw_shrimp(frame, x, 34 + jump, contour)
        status = TEAL if index >= 27 else CORAL
        draw.rectangle((389, 18, 402, 31), fill=status)
        if index >= 27:
            draw.line((392, 24, 396, 28), fill=SAND, width=2)
            draw.line((396, 28, 401, 20), fill=SAND, width=2)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=64))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/hero/shrimp-pixel-loop.gif")
    )
    parser.add_argument(
        "--source-sheet",
        type=Path,
        default=Path("docs/assets/hero/shrimp-pixel-source/sprite-sheet.png"),
    )
    args = parser.parse_args()
    frames = build_frames()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.source_sheet.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=True,
        disposal=2,
    )
    sheet = Image.new("RGB", (420 * 6, 120 * 6), NAVY)
    for index, frame in enumerate(frames):
        sheet.paste(frame.convert("RGB"), ((index % 6) * 420, (index // 6) * 120))
    sheet.save(args.source_sheet, optimize=True)
    print(f"Generated {args.output} and {args.source_sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
