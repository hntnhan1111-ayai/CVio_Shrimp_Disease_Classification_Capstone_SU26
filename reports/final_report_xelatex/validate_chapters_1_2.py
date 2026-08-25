from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "final_report_xelatex"
PDF = REPORT / "final" / "chapters_1_2_review.pdf"
LOG = REPORT / "validation" / "chapters_1_2_review.log"
OUT = REPORT / "validation"
OUT.mkdir(parents=True, exist_ok=True)


doc = fitz.open(PDF)
issues: list[str] = []
page_summary: list[dict] = []

expected_tokens = [
    "I. Project Introduction",
    "II. Project Management Plan",
    "2.3 Quality Management",
    "Experimental and Model-Comparison Quality",
    "Table 2.5. Detailed Work Breakdown Structure and estimated effort",
    "Figure 2.5. Document and experiment artifact flow.",
    "Figure 2.6. Controlled change-management process.",
    "Times New Roman typography",
]
all_text = "\n".join(page.get_text() for page in doc)
for token in expected_tokens:
    if token not in all_text:
        issues.append(f"Missing required token: {token}")

for token in ["Tinos typography", "revised navy text", "Figure 2.6. Document and experiment artifact flow"]:
    if token in all_text:
        issues.append(f"Obsolete token remains: {token}")

font_names: Counter[str] = Counter()
for pno, page in enumerate(doc, 1):
    page_rect = page.rect
    text_blocks = page.get_text("dict").get("blocks", [])
    visible_numbers = []
    yellow_drawings = 0
    for drawing in page.get_drawings():
        fill = drawing.get("fill")
        if fill and fill[0] > 0.8 and fill[1] > 0.75 and fill[2] < 0.35:
            yellow_drawings += 1
    for block in text_blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_names[span.get("font", "")] += len(span.get("text", ""))
                bbox = fitz.Rect(span["bbox"])
                if bbox.x0 < -0.5 or bbox.y0 < -0.5 or bbox.x1 > page_rect.x1 + 0.5 or bbox.y1 > page_rect.y1 + 0.5:
                    issues.append(f"Page {pno}: text outside page bounds: {span.get('text', '')[:60]}")
                value = span.get("text", "").strip()
                if bbox.y0 > page_rect.y1 - 45 and re.fullmatch(r"\d+", value):
                    visible_numbers.append(value)
    if visible_numbers != [str(pno)]:
        issues.append(f"Page {pno}: footer page numbers={visible_numbers}, expected={[str(pno)]}")
    page_summary.append({
        "page": pno,
        "text_chars": len(page.get_text()),
        "images": len(page.get_images(full=True)),
        "yellow_drawings": yellow_drawings,
        "footer_numbers": visible_numbers,
    })

if not any("TimesNewRoman" in name.replace("-", "") for name in font_names):
    issues.append(f"Times New Roman not detected; fonts={dict(font_names)}")
non_times = {name: count for name, count in font_names.items() if "TimesNewRoman" not in name.replace("-", "") and count > 100}
if non_times:
    issues.append(f"Unexpected high-volume fonts: {non_times}")

log_text = LOG.read_text(encoding="utf-8", errors="replace")
fatal_patterns = ["Undefined control sequence", "Emergency stop", "Fatal error", "! LaTeX Error"]
for pattern in fatal_patterns:
    if pattern in log_text:
        issues.append(f"Compile log contains: {pattern}")

report = {
    "pdf": str(PDF),
    "pages": len(doc),
    "font_char_counts": dict(font_names),
    "page_summary": page_summary,
    "issues": issues,
    "status": "PASS" if not issues else "FAIL",
}
(OUT / "chapters_1_2_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(OUT / "chapters_1_2_validation.md").write_text(
    "# Chapters I-II PDF Validation\n\n"
    f"- Status: **{report['status']}**\n"
    f"- Pages: {len(doc)}\n"
    f"- PDF: `{PDF}`\n"
    f"- Fonts: `{dict(font_names)}`\n"
    f"- Issues: {len(issues)}\n\n"
    + ("## Issues\n\n" + "\n".join(f"- {item}" for item in issues) + "\n" if issues else "All automated hard gates passed.\n"),
    encoding="utf-8",
)
print(json.dumps({"status": report["status"], "pages": len(doc), "issues": issues, "fonts": dict(font_names)}, indent=2))
