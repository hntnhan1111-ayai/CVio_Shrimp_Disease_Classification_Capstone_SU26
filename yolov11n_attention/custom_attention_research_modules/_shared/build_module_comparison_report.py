from __future__ import annotations

import io
import math
import re
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(".")
OUT_DIR = Path("custom_attention_research_modules/results")
CSV_PATH = OUT_DIR / "module_comparison_summary.csv"
MD_PATH = OUT_DIR / "module_comparison_report.md"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def clean_text(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "\n")


def output_text(output) -> str:
    chunks: list[str] = []
    if "text" in output:
        value = output["text"]
        chunks.append("".join(value) if isinstance(value, list) else str(value))
    data = output.get("data", {})
    value = data.get("text/plain")
    if value is not None:
        chunks.append("".join(value) if isinstance(value, list) else str(value))
    return clean_text("\n".join(chunks))


def notebook_text(nb) -> str:
    chunks: list[str] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            chunks.append(output_text(output))
    return "\n".join(chunks)


def html_tables(nb):
    for cell_index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for output_index, output in enumerate(cell.get("outputs", [])):
            html = output.get("data", {}).get("text/html") if isinstance(output, dict) else None
            if isinstance(html, list):
                html = "".join(html)
            if isinstance(html, str) and "<table" in html:
                try:
                    for df in pd.read_html(io.StringIO(html)):
                        yield cell_index, output_index, df
                except Exception:
                    continue


def rx_float(text: str, label: str) -> float | None:
    match = re.search(re.escape(label) + r"\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def yolo_all_values(text: str, images: int) -> list[float] | None:
    pattern = (
        rf"\ball\s+{images}\s+119\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+"
        r"([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)"
    )
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return [float(value) for value in matches[-1]]


def pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{100 * value:.1f}%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}"


def get_first(row: dict, *keys: str):
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def parse_standard(label: str, path: str, augmentation: str, family: str) -> dict:
    nb = nbformat.read(str(ROOT / path), as_version=4)
    summary: dict = {}
    compact: dict = {}

    for _, _, df in html_tables(nb):
        cols = set(map(str, df.columns))
        if "seg_loss_gap_val_minus_train" in cols:
            summary = df.iloc[0].to_dict()
        if "healthy_aware_labeled_test_mask_map50" in cols:
            compact = df.iloc[0].to_dict()

    text = notebook_text(nb)
    full = yolo_all_values(text, 129)
    labeled = yolo_all_values(text, 88)

    row = {
        "label": label,
        "family": family,
        "augmentation": augmentation,
        "path": path,
    }
    row.update(summary)
    row.update(compact)
    row["full_test_mask_map50"] = rx_float(text, "Full test mask mAP50:")
    row["labeled_test_mask_map50"] = rx_float(text, "Labeled-only diseased test mask mAP50:")
    row["healthy_test_mask_fp_rate"] = rx_float(text, "Healthy test mask false-positive rate:")
    row["healthy_fp_masks_per_image"] = rx_float(text, "Healthy test false-positive masks per image:")

    if full:
        row["full_box_p"] = full[0]
        row["full_box_r"] = full[1]
        row["full_box_map50"] = full[2]
        row["full_box_map50_95"] = full[3]
        row["full_mask_p"] = full[4]
        row["full_mask_r"] = full[5]
        row["full_mask_map50_from_line"] = full[6]
        row["full_mask_map50_95"] = full[7]
    if labeled:
        row["labeled_box_p"] = labeled[0]
        row["labeled_box_r"] = labeled[1]
        row["labeled_box_map50"] = labeled[2]
        row["labeled_box_map50_95"] = labeled[3]
        row["labeled_mask_p"] = labeled[4]
        row["labeled_mask_r"] = labeled[5]
        row["labeled_mask_map50_from_line"] = labeled[6]
        row["labeled_mask_map50_95"] = labeled[7]

    return row


def parse_simam_rows() -> list[dict]:
    path = "yolov11n_simam_NhomA/augmentation/yolov11n_simam_augmentation.ipynb"
    nb = nbformat.read(str(ROOT / path), as_version=4)

    summaries: dict[str, dict] = {}
    for _, _, df in html_tables(nb):
        cols = set(map(str, df.columns))
        if {"experiment", "best_val_mask_map50", "seg_loss_gap_val_minus_train"}.issubset(cols):
            for _, row in df.iterrows():
                summaries[str(row["experiment"])] = row.to_dict()

    rows: list[dict] = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            text = output_text(output)
            if "Full test mask mAP50:" not in text:
                continue
            if "simam-ca-head" in text.lower():
                key = "simam_ca"
                label = "SimAM_CA strong"
                family = "previous_best"
            elif "YOLO11n-seg summary" in text:
                key = "baseline"
                label = "Baseline strong"
                family = "baseline"
            else:
                continue

            full = yolo_all_values(text, 129)
            labeled = yolo_all_values(text, 88)
            row = {
                "label": label,
                "family": family,
                "augmentation": "strong",
                "path": path,
            }
            row.update(summaries.get(key, {}))
            row["full_test_mask_map50"] = rx_float(text, "Full test mask mAP50:")
            row["labeled_test_mask_map50"] = rx_float(text, "Labeled-only diseased test mask mAP50:")
            row["healthy_test_mask_fp_rate"] = rx_float(text, "Healthy test mask false-positive rate:")
            row["healthy_fp_masks_per_image"] = rx_float(text, "Healthy test false-positive masks per image:")
            if full:
                row["full_mask_p"] = full[4]
                row["full_mask_r"] = full[5]
                row["full_mask_map50_from_line"] = full[6]
                row["full_mask_map50_95"] = full[7]
            if labeled:
                row["labeled_mask_p"] = labeled[4]
                row["labeled_mask_r"] = labeled[5]
                row["labeled_mask_map50_from_line"] = labeled[6]
                row["labeled_mask_map50_95"] = labeled[7]
            rows.append(row)
    return rows


def build_rows() -> pd.DataFrame:
    rows = [
        parse_standard(
            "Baseline clean",
            "yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb",
            "clean_light",
            "baseline",
        ),
    ]
    rows.extend(parse_simam_rows())
    rows.extend(
        [
            parse_standard("CoTE", "custom_attention_research_modules/results/cote-gate.ipynb", "clean_light", "research"),
            parse_standard("LPSC", "custom_attention_research_modules/results/lpsc-gate.ipynb", "clean_light", "research"),
            parse_standard("SCSG", "custom_attention_research_modules/results/scsg-gate.ipynb", "clean_light", "research"),
            parse_standard(
                "CoTE strong",
                "custom_attention_research_modules/results/cote-gate-strong-augmentation.ipynb",
                "strong",
                "research",
            ),
            parse_standard(
                "LPSC strong",
                "custom_attention_research_modules/results/lpsc-gate-strong-augmentation.ipynb",
                "strong",
                "research",
            ),
            parse_standard(
                "CoLPSC",
                "custom_attention_research_modules/colpsc_gate/colpsc-gate.ipynb",
                "clean_light",
                "research_hybrid",
            ),
            parse_standard(
                "CoLPSC strong",
                "custom_attention_research_modules/colpsc_gate/colpsc-gate-strong-augmentation.ipynb",
                "strong",
                "research_hybrid",
            ),
        ]
    )
    df = pd.DataFrame(rows)

    for col in ["full_test_mask_map50", "labeled_test_mask_map50"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    order = [
        "Baseline clean",
        "Baseline strong",
        "SimAM_CA strong",
        "CoTE",
        "LPSC",
        "SCSG",
        "CoTE strong",
        "LPSC strong",
        "CoLPSC",
        "CoLPSC strong",
    ]
    df["order"] = df["label"].map({name: i for i, name in enumerate(order)})
    return df.sort_values("order").reset_index(drop=True)


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(title for title, _, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        values = []
        for _, key, kind in columns:
            value = row.get(key)
            if kind == "pct":
                values.append(pct(value))
            elif kind == "num":
                values.append(num(value))
            elif kind == "num2":
                values.append(num(value, 2))
            else:
                values.append("-" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def best_label(df: pd.DataFrame, col: str, maximize: bool = True) -> str:
    series = pd.to_numeric(df[col], errors="coerce")
    idx = series.idxmax() if maximize else series.idxmin()
    return f"{df.loc[idx, 'label']} ({num(series.loc[idx])})"


def write_report(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = df[df["label"] == "Baseline clean"].iloc[0]
    simam = df[df["label"] == "SimAM_CA strong"].iloc[0]
    cote_strong = df[df["label"] == "CoTE strong"].iloc[0]
    lpsc_strong = df[df["label"] == "LPSC strong"].iloc[0]

    main_cols = [
        ("Model", "label", "text"),
        ("Aug", "augmentation", "text"),
        ("Full mAP50", "full_test_mask_map50", "num"),
        ("Diseased mAP50", "labeled_test_mask_map50", "num"),
        ("Full mAP50-95", "full_mask_map50_95", "num"),
        ("Diseased mAP50-95", "labeled_mask_map50_95", "num"),
        ("Healthy FP", "healthy_test_mask_fp_rate", "pct"),
        ("FP masks/img", "healthy_fp_masks_per_image", "num"),
        ("Disease miss", "labeled_test_disease_box_miss_rate", "pct"),
        ("Healthy-aware", "healthy_aware_labeled_test_mask_map50", "num"),
    ]

    diag_cols = [
        ("Model", "label", "text"),
        ("Val mAP50", "best_val_mask_map50", "num"),
        ("Val mAP50-95", "best_val_mask_map50_95", "num"),
        ("Labeled val mAP50", "labeled_val_mask_map50", "num"),
        ("Val FP", "healthy_val_mask_fp_rate", "pct"),
        ("Val healthy-aware", "healthy_aware_labeled_val_mask_map50", "num"),
        ("Epochs", "epochs_ran", "num2"),
        ("Best epoch", "best_epoch_by_mask_map50", "num2"),
        ("Seg loss gap", "seg_loss_gap_val_minus_train", "num"),
    ]

    def delta(row, col: str) -> float | None:
        value = get_first(row, col)
        b = get_first(base, col)
        if value is None or b is None or pd.isna(value) or pd.isna(b):
            return None
        return float(value) - float(b)

    delta_lines = []
    for row in [simam, cote_strong, lpsc_strong]:
        delta_lines.append(
            f"- {row['label']}: full mAP50 {num(delta(row, 'full_test_mask_map50'))}, "
            f"diseased mAP50 {num(delta(row, 'labeled_test_mask_map50'))}, "
            f"healthy FP {pct(delta(row, 'healthy_test_mask_fp_rate'))} vs Baseline clean."
        )

    md = f"""# Module Comparison Report

Generated from completed notebooks in this workspace.

## Main Test Metrics

{markdown_table(df, main_cols)}

## Validation And Stability Diagnostics

{markdown_table(df, diag_cols)}

## Quick Ranking

- Best overall full test mask mAP50: {best_label(df, 'full_test_mask_map50')}
- Best diseased-only mask mAP50: {best_label(df, 'labeled_test_mask_map50')}
- Lowest healthy FP-rate: {best_label(df, 'healthy_test_mask_fp_rate', maximize=False)}
- Lowest FP masks per healthy image: {best_label(df, 'healthy_fp_masks_per_image', maximize=False)}
- Best healthy-aware test score among rows with this metric: {best_label(df.dropna(subset=['healthy_aware_labeled_test_mask_map50']), 'healthy_aware_labeled_test_mask_map50')}

## Deltas Versus Baseline Clean

{chr(10).join(delta_lines)}

## Interpretation

- SimAM_CA strong remains the best overall model by full test and diseased-only mask mAP50.
- CoTE strong is the best new research module for a balanced objective: it improves diseased-only mAP50 and healthy-aware score over the clean baseline while reducing healthy FP-rate.
- LPSC strong is the best new research module for low false positives: it has the lowest healthy FP-rate and FP masks per image among the custom attention experiments, but it raises disease miss rate.
- SCSG and CoLPSC are not recommended as primary candidates in the current form because their test mAP and healthy-aware scores trail the better alternatives.
- The gap between clean/light and strong augmentation shows augmentation is a major factor; architecture conclusions should be made against models trained under the same augmentation policy.
"""

    df.to_csv(CSV_PATH, index=False)
    MD_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")


def main() -> None:
    df = build_rows()
    write_report(df)
    print(df[[
        "label",
        "augmentation",
        "full_test_mask_map50",
        "labeled_test_mask_map50",
        "full_mask_map50_95",
        "healthy_test_mask_fp_rate",
        "healthy_fp_masks_per_image",
        "labeled_test_disease_box_miss_rate",
        "healthy_aware_labeled_test_mask_map50",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
