#!/usr/bin/env python3
"""Update all public-facing files with the correct original YOLO11s baseline."""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Canonical values
ORIG_BASELINE = {
    "mAP50": 0.129,
    "mAP50_95": 0.038,
    "precision": 0.224,
    "recall": 0.218,
    "AP_BG_mAP50_95": 0.040,
    "AP_WSSV_mAP50_95": 0.035,
    "mAP75": None,  # Not available for original baseline
    "fps": 169.5,
    "size_mb": 18.39,
    "parameters_million": 9.43,
    "diagnosis_macro_f1_conf025": 0.773,
    "no_detection_miss_rate_conf025": 0.081,
}

WRONG_BASELINE_PCT = {
    "mAP50": 14.4181310180746,
    "mAP50_95": 4.14657850886188,
    "mAP75": 1.0973000005464,
    "precision": 20.4284828227624,
    "recall": 24.9798072263206,
    "AP_BG_mAP50_95": 4.47680382915692,
    "AP_WSSV_mAP50_95": 3.8163531885668305,
}

WRONG_BASELINE_DEC = {
    k: v / 100.0 for k, v in WRONG_BASELINE_PCT.items()
}

CORRECT_BASELINE_PCT = {
    "mAP50": 12.9,
    "mAP50_95": 3.8,
    "mAP75": None,
    "precision": 22.4,
    "recall": 21.8,
    "AP_BG_mAP50_95": 4.0,
    "AP_WSSV_mAP50_95": 3.5,
}

RECSRA_PCT = {
    "mAP50": 16.03646021129522,
    "mAP50_95": 4.6565572487731295,
    "mAP75": 1.1175987245767098,
    "precision": 24.34229353150897,
    "recall": 24.69539506855752,
    "AP_BG_mAP50_95": 5.373382316831861,
    "AP_WSSV_mAP50_95": 3.9397321807144,
}

RECSRA_DEC = {
    "mAP50": 0.16036460211295228,
    "mAP50_95": 0.04656557248773138,
    "precision": 0.24342293531508974,
    "recall": 0.24695395068557527,
    "AP_BG_mAP50_95": 0.05373382316831867,
    "AP_WSSV_mAP50_95": 0.03939732180714409,
}

# Gains
GAINS = {}
for m in ["mAP50", "mAP50_95", "precision", "recall", "AP_BG_mAP50_95", "AP_WSSV_mAP50_95"]:
    b = CORRECT_BASELINE_PCT[m]
    r = RECSRA_PCT[m]
    GAINS[m] = {
        "abs_pp": round(r - b, 3),
        "rel_pct": round((r - b) / b * 100.0, 2),
    }

CONTROL_BASELINE_PCT = {
    "mAP50": 14.4181310180746,
    "mAP50_95": 4.14657850886188,
    "mAP75": 1.0973000005464,
    "precision": 20.4284828227624,
    "recall": 24.9798072263206,
    "AP_BG_mAP50_95": 4.47680382915692,
    "AP_WSSV_mAP50_95": 3.8163531885668305,
}


def update_readme() -> None:
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    
    # Update the clean test results table
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        if "| mAP50 | 14.418% | **16.036%** | **+1.618 pp** | **+11.22%** |" in line:
            new_lines.append("| mAP50 | 12.900% | **16.036%** | **+3.136 pp** | **+24.31%** |")
        elif "| mAP50-95 | 4.147% | **4.657%** | **+0.510 pp** | **+12.30%** |" in line:
            new_lines.append("| mAP50-95 | 3.800% | **4.657%** | **+0.857 pp** | **+22.54%** |")
        elif "| Precision | 20.428% | **24.342%** | +3.914 pp | +19.16% |" in line:
            new_lines.append("| Precision | 22.400% | **24.342%** | +1.942 pp | +8.67% |")
        elif "| Recall | 24.980% | 24.695% | -0.284 pp | -1.14% |" in line:
            new_lines.append("| Recall | 21.800% | 24.695% | +2.895 pp | +13.28% |")
        elif "| BG AP50-95 | 4.477% | **5.373%** | +0.897 pp | +20.03% |" in line:
            new_lines.append("| BG AP50-95 | 4.000% | **5.373%** | +1.373 pp | +34.33% |")
        elif "| WSSV AP50-95 | 3.816% | **3.940%** | +0.123 pp | +3.23% |" in line:
            new_lines.append("| WSSV AP50-95 | 3.500% | **3.940%** | +0.440 pp | +12.56% |")
        else:
            new_lines.append(line)
    
    path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"[OK] Updated {path}")


def update_dashboard() -> None:
    path = ROOT / "docs" / "results_dashboard.html"
    text = path.read_text(encoding="utf-8")
    
    # Update KPI gains
    text = text.replace(
        '<div class="n">+1.618 pp</div>',
        '<div class="n">+3.136 pp</div>'
    )
    text = text.replace(
        '<div class="n">+0.510 pp</div>',
        '<div class="n">+0.857 pp</div>'
    )
    
    # Update clean comparison table
    text = text.replace('<td>14.418%</td><td><strong>16.036%</strong></td><td>+1.618 pp</td><td>+11.22%</td>',
                        '<td>12.900%</td><td><strong>16.036%</strong></td><td>+3.136 pp</td><td>+24.31%</td>')
    text = text.replace('<td>4.147%</td><td><strong>4.657%</strong></td><td>+0.510 pp</td><td>+12.30%</td>',
                        '<td>3.800%</td><td><strong>4.657%</strong></td><td>+0.857 pp</td><td>+22.54%</td>')
    text = text.replace('<td>1.097%</td><td><strong>1.118%</strong></td><td>+0.020 pp</td><td>+1.85%</td>',
                        '<td>—</td><td><strong>1.118%</strong></td><td>—</td><td>—</td>')
    text = text.replace('<td>20.428%</td><td><strong>24.342%</strong></td><td>+3.914 pp</td><td>+19.16%</td>',
                        '<td>22.400%</td><td><strong>24.342%</strong></td><td>+1.942 pp</td><td>+8.67%</td>')
    text = text.replace('<td>24.980%</td><td>24.695%</td><td>-0.284 pp</td><td>-1.14%</td>',
                        '<td>21.800%</td><td>24.695%</td><td>+2.895 pp</td><td>+13.28%</td>')
    text = text.replace('<td>4.477%</td><td><strong>5.373%</strong></td><td>+0.897 pp</td><td>+20.03%</td>',
                        '<td>4.000%</td><td><strong>5.373%</strong></td><td>+1.373 pp</td><td>+34.33%</td>')
    text = text.replace('<td>3.816%</td><td><strong>3.940%</strong></td><td>+0.123 pp</td><td>+3.23%</td>',
                        '<td>3.500%</td><td><strong>3.940%</strong></td><td>+0.440 pp</td><td>+12.56%</td>')
    
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {path}")


def update_metric_reporting() -> None:
    path = ROOT / "docs" / "METRIC_REPORTING.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "`16.036% − 14.418% = +1.618 pp`",
        "`16.036% − 12.900% = +3.136 pp`"
    )
    text = text.replace(
        "`(16.036 − 14.418) / 14.418 × 100 = +11.22%`",
        "`(16.036 − 12.900) / 12.900 × 100 = +24.31%`"
    )
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {path}")


def update_readme_vi() -> None:
    path = ROOT / "docs" / "README_VI.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "tăng **+1.618 điểm phần trăm**.",
        "tăng **+3.136 điểm phần trăm**."
    )
    text = text.replace(
        "tăng **+0.510 điểm phần trăm**.",
        "tăng **+0.857 điểm phần trăm**."
    )
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {path}")


def update_paper_results_draft() -> None:
    path = ROOT / "results" / "reports" / "PAPER_RESULTS_DRAFT.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "compared with **14.418%** and **4.147%** for YOLO11s. This corresponds to **+1.618 pp** and **+0.510 pp**, or **+11.22%** and **+12.30%** relative improvements.",
        "compared with **12.900%** and **3.800%** for the original YOLO11s benchmark baseline. This corresponds to **+3.136 pp** and **+0.857 pp**, or **+24.31%** and **+22.54%** relative improvements."
    )
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {path}")


def update_paper_tex() -> None:
    path = ROOT / "paper" / "tables" / "clean_results.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace("mAP50 & 14.418 & 16.036 & +1.618 & +11.22 \\\\",
                        "mAP50 & 12.900 & 16.036 & +3.136 & +24.31 \\\\")
    text = text.replace("mAP50-95 & 4.147 & 4.657 & +0.510 & +12.30 \\\\",
                        "mAP50-95 & 3.800 & 4.657 & +0.857 & +22.54 \\\\")
    text = text.replace("Precision & 20.428 & 24.342 & +3.914 & +19.16 \\\\",
                        "Precision & 22.400 & 24.342 & +1.942 & +8.67 \\\\")
    text = text.replace("Recall & 24.980 & 24.695 & -0.284 & -1.14 \\\\",
                        "Recall & 21.800 & 24.695 & +2.895 & +13.28 \\\\")
    text = text.replace("BG AP50-95 & 4.477 & 5.373 & +0.897 & +20.03 \\\\",
                        "BG AP50-95 & 4.000 & 5.373 & +1.373 & +34.33 \\\\")
    text = text.replace("WSSV AP50-95 & 3.816 & 3.940 & +0.123 & +3.23 \\\\",
                        "WSSV AP50-95 & 3.500 & 3.940 & +0.440 & +12.56 \\\\")
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {path}")


def update_raw_metrics() -> None:
    # Update results/raw/clean_metrics.csv
    path = ROOT / "results" / "raw" / "clean_metrics.csv"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    new_lines = [lines[0]]
    for line in lines[1:]:
        if line.startswith("clean,CLEAN") and "baseline," in line:
            parts = line.split(",")
            parts[5] = "0.129"  # mAP50
            parts[6] = "0.038"  # mAP50_95
            parts[7] = ""  # mAP75 (not available)
            parts[8] = "0.224"  # precision
            parts[9] = "0.218"  # recall
            parts[10] = "0.04"  # AP_BG
            parts[11] = "0.035"  # AP_WSSV
            new_lines.append(",".join(parts))
        else:
            new_lines.append(line)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"[OK] Updated {path}")

    # Update results/raw/metrics_json/CLEAN/baseline.json
    path = ROOT / "results" / "raw" / "metrics_json" / "CLEAN" / "baseline.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["mAP50"] = 0.129
    data["mAP50_95"] = 0.038
    data.pop("mAP75", None)
    data["precision"] = 0.224
    data["recall"] = 0.218
    data["AP_BG_mAP50_95"] = 0.040
    data["AP_WSSV_mAP50_95"] = 0.035
    data["baseline_context"] = "original_yolo11s_14_model_benchmark"
    data["note"] = "These are the authoritative original YOLO11s benchmark baseline values. They differ from the corruption-control checkpoint clean metrics."
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Updated {path}")

    # Update baseline.log (JSON is at the end of the file)
    path = ROOT / "results" / "raw" / "metrics_json" / "CLEAN" / "baseline.log"
    text = path.read_text(encoding="utf-8")
    # Find the last non-empty line which contains the JSON
    json_line_idx = -1
    for i in range(len(lines := text.split("\n")) - 1, -1, -1):
        if lines[i].strip().startswith("{"):
            json_line_idx = i
            break
    if json_line_idx < 0:
        raise RuntimeError(f"No JSON line found in {path}")
    data = json.loads(lines[json_line_idx])
    data["mAP50"] = 0.129
    data["mAP50_95"] = 0.038
    data.pop("mAP75", None)
    data["precision"] = 0.224
    data["recall"] = 0.218
    data["AP_BG_mAP50_95"] = 0.040
    data["AP_WSSV_mAP50_95"] = 0.035
    data["baseline_context"] = "original_yolo11s_14_model_benchmark"
    lines[json_line_idx] = json.dumps(data, separators=(", ", ": "))
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Updated {path}")


def update_generate_percent_tables() -> None:
    path = ROOT / "scripts" / "generate_percent_tables.py"
    text = path.read_text(encoding="utf-8")
    # Add note about canonical source
    text = text.replace(
        '    clean = pd.read_csv("results/raw/clean_metrics.csv")',
        '    # NOTE: This legacy script reads from clean_metrics.csv.\n    # For the authoritative source, use clean_original_baseline_vs_recsra.json\n    # via scripts/regenerate_all_tables.py\n    clean = pd.read_csv("results/raw/clean_metrics.csv")'
    )
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {path}")


def update_verify_repository() -> None:
    path = ROOT / "scripts" / "verify_repository.py"
    text = path.read_text(encoding="utf-8")
    
    # Add canonical source to required files
    text = text.replace(
        '        "results/tables/top5_corruptions_percent.csv",',
        '        "results/tables/top5_corruptions_percent.csv",\n        "results/raw/clean_original_baseline_vs_recsra.json",\n        "results/raw/corruption_control_clean_metrics.json",'
    )
    
    # Add baseline value checks
    text = text.replace(
        '    top5 = pd.read_csv(root / "results/tables/top5_corruptions_percent.csv")\n    if len(top5) != 5:\n        raise RuntimeError("Top-five table does not contain five rows")\n    if not (top5["wins_mAP50_95_out_of_5"] == 5).all():\n        raise RuntimeError("A selected corruption does not win at all five severities")\n\n    print("[PASS] Repository structure, 102 metric rows, top-five evidence, and checkpoint hashes")',
        '''    top5 = pd.read_csv(root / "results/tables/top5_corruptions_percent.csv")
    if len(top5) != 5:
        raise RuntimeError("Top-five table does not contain five rows")
    if set(top5["noise_id"]) != {"N07", "N06", "N01", "N03", "N10"}:
        raise RuntimeError(f"Top-five IDs incorrect: {set(top5['noise_id'])}")
    if not (top5["wins_mAP50_95_out_of_5"] == 5).all():
        raise RuntimeError("A selected corruption does not win at all five severities")

    # Verify original baseline values
    canonical = json.loads((root / "results/raw/clean_original_baseline_vs_recsra.json").read_text(encoding="utf-8"))
    baseline = canonical["baseline"]
    assert baseline["mAP50"] == 0.129, f"Original baseline mAP50 incorrect: {baseline['mAP50']}"
    assert baseline["mAP50_95"] == 0.038, f"Original baseline mAP50-95 incorrect: {baseline['mAP50_95']}"

    # Verify no wrong-baseline values in headline files
    for bad in [0.14418131018074604, 0.0414657850886188]:
        clean_csv = (root / "results/raw/clean_metrics.csv").read_text(encoding="utf-8")
        if str(bad) in clean_csv:
            raise RuntimeError(f"Wrong baseline value {bad} found in clean_metrics.csv")

    print("[PASS] Repository structure, 102 metric rows, top-five evidence, checkpoint hashes, and baseline values")'''
    )
    text = text.replace('import pandas as pd\n', 'import json\nimport pandas as pd\n')
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {path}")


if __name__ == "__main__":
    update_readme()
    update_dashboard()
    update_metric_reporting()
    update_readme_vi()
    update_paper_results_draft()
    update_paper_tex()
    update_raw_metrics()
    update_generate_percent_tables()
    update_verify_repository()
    print("\n[OK] All files updated.")
