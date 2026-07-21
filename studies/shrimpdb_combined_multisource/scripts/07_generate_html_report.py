#!/usr/bin/env python3
"""Generate self-contained HTML academic report."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

STUDY_CONFIG = Path("configs/study.yaml")
REPORT_PATH = Path("artifacts/reports/CVio_Final_Academic_Report_ShrimpDB_Combined_seed42.html")


def _img_to_base64(path: Path) -> str:
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _csv_to_html(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return ""
    header, *body = rows
    th = "".join(f"<th>{c}</th>" for c in header)
    trs = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in body
    )
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


def generate_report(
    figures_dir: Path = Path("artifacts/figures"),
    tables_dir: Path = Path("artifacts/tables"),
    output: Path = REPORT_PATH,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)

    study = json.loads(STUDY_CONFIG.read_text(encoding="utf-8"))
    sections: list[str] = []
    sections.append("<h1>CVio ShrimpDB Combined Research Report (seed=42)</h1>")
    sections.append(f"<p>Study: {study.get('study', {}).get('name', '')}</p>")
    sections.append(f"<p>Model: {study.get('training', {}).get('model', '')} | Method: {study.get('training', {}).get('method', '')}</p>")

    for table_name in ["experiment_overview_percent.csv", "dataset_source_split_distribution.csv"]:
        table_path = tables_dir / table_name
        if table_path.is_file():
            sections.append(f"<h2>{table_name}</h2>")
            sections.append(_csv_to_html(table_path))

    for exp in ("shrimpdb3", "combined4"):
        for suffix in ["best_confusion_count.png", "best_confusion_normalized.png"]:
            fig_path = figures_dir / f"{exp}_{suffix}"
            if fig_path.is_file():
                sections.append(f"<h2>{fig_path.name}</h2>")
                sections.append(f'<img src="{_img_to_base64(fig_path)}" style="max-width:100%;">')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CVio Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
th {{ background: #f4f4f4; }}
img {{ max-width: 100%; height: auto; margin-bottom: 2rem; }}
</style></head><body>{"".join(sections)}</body></html>"""
    output.write_text(html, encoding="utf-8")
    LOGGER.info("Wrote HTML report to %s", output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HTML report")
    parser.add_argument("--figures-dir", default="artifacts/figures")
    parser.add_argument("--tables-dir", default="artifacts/tables")
    parser.add_argument("--output", default=str(REPORT_PATH))
    args = parser.parse_args()
    try:
        return generate_report(Path(args.figures_dir), Path(args.tables_dir), Path(args.output))
    except Exception as exc:
        LOGGER.error("Report generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
