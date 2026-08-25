from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "reports" / "final_report_xelatex" / "validation" / "rendered_pages"
OUT = ROOT / "reports" / "final_report_xelatex" / "validation" / "contact_sheets"
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(PAGES.glob("page-*.jpg"))
per_sheet = 8
columns = 4
thumb_w = 300
gap = 22
label_h = 30

for sheet_index in range(math.ceil(len(files) / per_sheet)):
    group = files[sheet_index * per_sheet:(sheet_index + 1) * per_sheet]
    thumbs = []
    for path in group:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, 450), Image.Resampling.LANCZOS)
        thumbs.append((path, image.copy()))
    rows = math.ceil(len(thumbs) / columns)
    cell_h = max(image.height for _, image in thumbs) + label_h
    canvas = Image.new("RGB", (columns * (thumb_w + gap) + gap, rows * (cell_h + gap) + gap), "#d9dde2")
    draw = ImageDraw.Draw(canvas)
    for idx, (path, image) in enumerate(thumbs):
        col = idx % columns
        row = idx // columns
        x = gap + col * (thumb_w + gap) + (thumb_w - image.width) // 2
        y = gap + row * (cell_h + gap) + label_h
        draw.rectangle((x - 2, y - 2, x + image.width + 2, y + image.height + 2), fill="white", outline="#777777", width=1)
        canvas.paste(image, (x, y))
        draw.text((gap + col * (thumb_w + gap), gap + row * (cell_h + gap)), path.stem, fill="black")
    canvas.save(OUT / f"contact_{sheet_index + 1:02d}.jpg", quality=92)
    print(OUT / f"contact_{sheet_index + 1:02d}.jpg")
