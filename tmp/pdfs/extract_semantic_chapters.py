from __future__ import annotations

import json
import re
from pathlib import Path

import fitz


ROOT = Path(r"C:\Users\Admin\workspace\CVio_Shrimp_Disease_Classification_Capstone_SU26")
SRC = ROOT / "CVio_Codex_Final_Report_Handoff_2026-08-11" / "CVio_Final_Report_Split_2026-08-13" / "CVio_Final_Report_Split_2026-08-13"
OUT = ROOT / "reports" / "final_report_xelatex"
ASSETS = OUT / "assets" / "chapters_1_2"
DATA = OUT / "source_data"
ASSETS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

FILES = {
    "chapter1": SRC / "01_chapter_I_project_introduction.pdf",
    "chapter2": SRC / "02_chapter_II_project_management_plan.pdf",
}


def clean_text(value: str) -> str:
    value = value.replace("\u037e", ";").replace("\u00ad", "")
    value = value.replace("\ufb01", "fi").replace("\ufb02", "fl")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def rect_overlap(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    return max(0.0, inter.get_area())


def yellow_rects(page: fitz.Page) -> list[fitz.Rect]:
    found = []
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        rect = drawing.get("rect")
        if fill and rect:
            r, g, b = fill
            if r > 0.8 and g > 0.75 and b < 0.35:
                found.append(fitz.Rect(rect))
    return found


result = {}
real_table_indices = {
    ("chapter1", 1): [0],
    ("chapter1", 2): [0, 1],
    ("chapter1", 7): [0],
    ("chapter1", 8): [0],
    ("chapter2", 1): [1],
    ("chapter2", 2): [0],
    ("chapter2", 3): [0],
    ("chapter2", 4): [0],
    ("chapter2", 5): [0],
    ("chapter2", 7): [0],
    ("chapter2", 9): [2],
    ("chapter2", 10): [0],
    ("chapter2", 11): [0],
    ("chapter2", 12): [0],
    ("chapter2", 13): [0],
    ("chapter2", 15): [1],
    ("chapter2", 16): [0],
    ("chapter2", 18): [1],
    ("chapter2", 19): [0],
    ("chapter2", 22): [0],
    ("chapter2", 24): [1],
}
figure_map = {
    ("chapter1", 3): "figure_1_1_scope_evolution",
    ("chapter1", 4): "figure_1_2_technology_stack",
    ("chapter2", 2): "figure_2_1_team_structure",
    ("chapter2", 7): "figure_2_2_crisp_dm",
    ("chapter2", 8): "figure_2_3_weekly_cycle",
    ("chapter2", 12): "figure_2_4_gantt",
    ("chapter2", 20): "figure_2_5_artifact_flow",
    ("chapter2", 23): "figure_2_6_change_process",
}

for chapter, path in FILES.items():
    doc = fitz.open(path)
    chapter_data = []
    for pno, page in enumerate(doc, 1):
        highlights = yellow_rects(page)
        all_tables = page.find_tables().tables
        tables = [all_tables[index] for index in real_table_indices.get((chapter, pno), [])]
        table_rects = [fitz.Rect(table.bbox) for table in tables]
        images = page.get_image_info(xrefs=True)
        image_rects = [fitz.Rect(info["bbox"]) for info in images]

        if (chapter, pno) in figure_map and images:
            info = max(images, key=lambda item: item["width"] * item["height"])
            payload = doc.extract_image(info["xref"])
            asset_path = ASSETS / f"{figure_map[(chapter, pno)]}.{payload['ext']}"
            asset_path.write_bytes(payload["image"])

        blocks = []
        for block in page.get_text("dict", flags=fitz.TEXTFLAGS_DICT).get("blocks", []):
            if "lines" not in block:
                continue
            bbox = fitz.Rect(block["bbox"])
            if bbox.y0 > 785:
                continue
            if any(rect_overlap(bbox, table_rect) > 0.5 * bbox.get_area() for table_rect in table_rects):
                continue
            if any(rect_overlap(bbox, image_rect) > 0.5 * bbox.get_area() for image_rect in image_rects):
                continue
            spans = []
            for line in block["lines"]:
                for span in line["spans"]:
                    text = clean_text(span["text"])
                    if not text:
                        continue
                    span_rect = fitz.Rect(span["bbox"])
                    highlighted = any(rect_overlap(span_rect, hr) > 0.15 * span_rect.get_area() for hr in highlights)
                    spans.append({
                        "text": text,
                        "font": span["font"],
                        "size": round(span["size"], 2),
                        "bold": "Bold" in span["font"],
                        "highlighted": highlighted,
                        "bbox": [round(v, 2) for v in span["bbox"]],
                    })
            text = clean_text(" ".join(span["text"] for span in spans))
            if text and not re.fullmatch(r"\d+", text):
                blocks.append({
                    "bbox": [round(v, 2) for v in bbox],
                    "text": text,
                    "spans": spans,
                    "highlight_ratio": round(sum(len(s["text"]) for s in spans if s["highlighted"]) / max(1, sum(len(s["text"]) for s in spans)), 3),
                })

        table_data = []
        for table in tables:
            extracted = [[clean_text(cell or "") for cell in row] for row in table.extract()]
            table_data.append({
                "bbox": [round(v, 2) for v in table.bbox],
                "rows": extracted,
                "row_count": table.row_count,
                "col_count": table.col_count,
            })
        chapter_data.append({
            "page": pno,
            "blocks": blocks,
            "tables": table_data,
            "highlight_rect_count": len(highlights),
        })
    result[chapter] = chapter_data

(DATA / "chapters_1_2_semantic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print("assets:")
for path in sorted(ASSETS.iterdir()):
    print(path.name, path.stat().st_size)
print("pages:", {key: len(value) for key, value in result.items()})
for key, pages in result.items():
    highlighted = sum(sum(1 for block in page["blocks"] if block["highlight_ratio"] > 0.5) for page in pages)
    total = sum(len(page["blocks"]) for page in pages)
    print(key, "highlighted blocks", highlighted, "/", total)
