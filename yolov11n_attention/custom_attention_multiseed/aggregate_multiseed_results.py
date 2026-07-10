"""Aggregate top-4 custom attention multiseed and threshold-sweep results."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

if __name__ == "__main__" and str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_top4_multiseed import (  # noqa: E402
    TOP4_KEYS,
    load_project_config,
    load_seeds,
    repo_path,
    selected_modules,
)


DEFAULT_METRICS = [
    "healthy_aware_score",
    "labeled_test_mask_map50",
    "labeled_test_mask_map50_95",
    "full_test_mask_map50",
    "full_test_mask_map50_95",
    "healthy_test_healthy_mask_fp_rate",
    "healthy_test_healthy_fp_masks_per_image",
    "labeled_test_disease_box_miss_rate",
    "labeled_test_mask_count_mae",
    "full_test_fps",
    "labeled_test_fps",
    "params",
]

SWEEP_OUTPUT_COLUMNS = [
    "module",
    "display_name",
    "seed",
    "checkpoint",
    "dataset_dir",
    "data_yaml",
    "conf",
    "iou",
    "params",
    "eval_time_sec",
    "val_box_precision",
    "val_box_recall",
    "val_box_map50",
    "val_box_map50_95",
    "val_mask_precision",
    "val_mask_recall",
    "val_mask_map50",
    "val_mask_map50_95",
    "full_test_box_precision",
    "full_test_box_recall",
    "full_test_box_map50",
    "full_test_box_map50_95",
    "full_test_mask_precision",
    "full_test_mask_recall",
    "full_test_mask_map50",
    "full_test_mask_map50_95",
    "labeled_test_box_precision",
    "labeled_test_box_recall",
    "labeled_test_box_map50",
    "labeled_test_box_map50_95",
    "labeled_test_mask_precision",
    "labeled_test_mask_recall",
    "labeled_test_mask_map50",
    "labeled_test_mask_map50_95",
    "labeled_test_mask_count_mae",
    "labeled_test_disease_box_miss_rate",
    "healthy_test_healthy_mask_fp_rate",
    "healthy_test_healthy_fp_masks_per_image",
    "healthy_aware_score",
    "primary_default_threshold",
]

RANKING_COLUMNS = [
    "module",
    "display_name",
    "n_seeds",
    "missing_seeds",
    "healthy_aware_score_mean",
    "healthy_aware_score_std",
    "labeled_test_mask_map50_mean",
    "labeled_test_mask_map50_std",
    "full_test_mask_map50_mean",
    "full_test_mask_map50_std",
    "labeled_test_mask_map50_95_mean",
    "healthy_test_healthy_mask_fp_rate_mean",
    "healthy_test_healthy_mask_fp_rate_std",
    "labeled_test_disease_box_miss_rate_mean",
    "labeled_test_disease_box_miss_rate_std",
    "labeled_test_mask_count_mae_mean",
    "full_test_fps_mean",
    "params_mean",
    "stability_std_sum",
    "deployment_rank_sum",
    "complete_3seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to top4_modules.yaml.")
    parser.add_argument("--seeds", default=None, help="Path to seeds.yaml.")
    return parser.parse_args()


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def ensure_empty_csv(path: Path, columns: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def round4(value: Any) -> float:
    try:
        return round(float(value), 4)
    except Exception:
        return float("nan")


def fmt_float(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def fmt_mean_std(mean_value: Any, std_value: Any, digits: int = 4) -> str:
    mean_text = fmt_float(mean_value, digits)
    std_text = fmt_float(std_value, digits)
    if not mean_text:
        return ""
    if not std_text:
        return mean_text
    return f"{mean_text} +/- {std_text}"


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, [column for column in columns if column in df.columns]].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: fmt_float(value))
    return view.to_markdown(index=False)


def dedupe_sweep(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    required = ["module", "seed", "conf", "iou"]
    if not all(column in df.columns for column in required):
        return df
    df = numeric(df, ["seed", "conf", "iou"])
    return df.drop_duplicates(subset=required, keep="last").reset_index(drop=True)


def default_threshold_rows(df: pd.DataFrame, default_conf: float, default_iou: float) -> pd.DataFrame:
    if df.empty or "conf" not in df.columns or "iou" not in df.columns:
        return pd.DataFrame()
    df = numeric(df.copy(), ["conf", "iou", "seed"])
    mask = (df["conf"].round(4) == round4(default_conf)) & (df["iou"].round(4) == round4(default_iou))
    return df.loc[mask].copy()


def summarize_default_rows(
    default_df: pd.DataFrame,
    modules: list[dict[str, Any]],
    seeds: list[int],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    default_df = numeric(default_df.copy(), ["seed", *DEFAULT_METRICS])
    for module in modules:
        module_key = module["key"]
        module_df = default_df.loc[default_df.get("module", pd.Series(dtype=str)) == module_key].copy()
        present_seeds = sorted(int(seed) for seed in module_df["seed"].dropna().unique()) if "seed" in module_df else []
        missing_seeds = [seed for seed in seeds if seed not in present_seeds]
        record: dict[str, Any] = {
            "module": module_key,
            "display_name": module.get("display_name", module_key),
            "n_seeds": len(present_seeds),
            "expected_seeds": len(seeds),
            "present_seeds": ",".join(map(str, present_seeds)),
            "missing_seeds": ",".join(map(str, missing_seeds)),
            "complete_3seed": len(present_seeds) == len(seeds),
        }
        for metric in DEFAULT_METRICS:
            if metric in module_df.columns and not module_df.empty:
                record[f"{metric}_mean"] = module_df[metric].mean()
                record[f"{metric}_std"] = module_df[metric].std(ddof=1) if len(module_df) > 1 else float("nan")
                record[f"{metric}_min"] = module_df[metric].min()
                record[f"{metric}_max"] = module_df[metric].max()
            else:
                record[f"{metric}_mean"] = float("nan")
                record[f"{metric}_std"] = float("nan")
                record[f"{metric}_min"] = float("nan")
                record[f"{metric}_max"] = float("nan")
        records.append(record)
    summary = pd.DataFrame(records)
    stability_parts = [
        "healthy_aware_score_std",
        "labeled_test_mask_map50_std",
        "healthy_test_healthy_mask_fp_rate_std",
        "labeled_test_disease_box_miss_rate_std",
    ]
    summary["stability_std_sum"] = summary[stability_parts].sum(axis=1, min_count=1)
    add_rank_columns(summary)
    return summary


def add_rank_columns(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    rank_specs = [
        ("rank_healthy_aware_score", "healthy_aware_score_mean", False),
        ("rank_labeled_test_mask_map50", "labeled_test_mask_map50_mean", False),
        ("rank_full_test_mask_map50", "full_test_mask_map50_mean", False),
        ("rank_healthy_fp_low", "healthy_test_healthy_mask_fp_rate_mean", True),
        ("rank_miss_low", "labeled_test_disease_box_miss_rate_mean", True),
        ("rank_stability", "stability_std_sum", True),
        ("rank_params_low", "params_mean", True),
        ("rank_fps_high", "full_test_fps_mean", False),
    ]
    for rank_col, metric, ascending in rank_specs:
        if metric in summary.columns:
            summary[rank_col] = summary[metric].rank(ascending=ascending, method="min", na_option="bottom")
            summary.loc[summary[metric].isna(), rank_col] = float("nan")
    deployment_parts = [
        "rank_healthy_aware_score",
        "rank_healthy_fp_low",
        "rank_miss_low",
        "rank_params_low",
        "rank_fps_high",
    ]
    existing = [column for column in deployment_parts if column in summary.columns]
    summary["deployment_rank_sum"] = summary[existing].sum(axis=1, min_count=1)


def threshold_best_by_module(df: pd.DataFrame, modules: list[dict[str, Any]], seeds: list[int]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "module",
                "display_name",
                "conf",
                "iou",
                "n_seeds",
                "complete_3seed",
                "healthy_aware_score_mean",
                "labeled_test_mask_map50_mean",
                "full_test_mask_map50_mean",
                "healthy_test_healthy_mask_fp_rate_mean",
                "labeled_test_disease_box_miss_rate_mean",
                "labeled_test_mask_count_mae_mean",
            ]
        )
    metrics = [
        "healthy_aware_score",
        "labeled_test_mask_map50",
        "full_test_mask_map50",
        "healthy_test_healthy_mask_fp_rate",
        "labeled_test_disease_box_miss_rate",
        "labeled_test_mask_count_mae",
    ]
    df = numeric(df.copy(), ["seed", "conf", "iou", *metrics])
    grouped = (
        df.groupby(["module", "conf", "iou"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            healthy_aware_score_mean=("healthy_aware_score", "mean"),
            healthy_aware_score_std=("healthy_aware_score", "std"),
            labeled_test_mask_map50_mean=("labeled_test_mask_map50", "mean"),
            full_test_mask_map50_mean=("full_test_mask_map50", "mean"),
            healthy_test_healthy_mask_fp_rate_mean=("healthy_test_healthy_mask_fp_rate", "mean"),
            labeled_test_disease_box_miss_rate_mean=("labeled_test_disease_box_miss_rate", "mean"),
            labeled_test_mask_count_mae_mean=("labeled_test_mask_count_mae", "mean"),
        )
        .reset_index()
    )
    display_names = {module["key"]: module.get("display_name", module["key"]) for module in modules}
    grouped["display_name"] = grouped["module"].map(display_names).fillna(grouped["module"])
    grouped["complete_3seed"] = grouped["n_seeds"] == len(seeds)

    best_rows: list[pd.Series] = []
    for module in [module["key"] for module in modules]:
        module_df = grouped.loc[grouped["module"] == module].copy()
        if module_df.empty:
            continue
        complete = module_df.loc[module_df["complete_3seed"]]
        candidates = complete if not complete.empty else module_df
        candidates = candidates.sort_values(
            by=[
                "healthy_aware_score_mean",
                "labeled_test_mask_map50_mean",
                "healthy_test_healthy_mask_fp_rate_mean",
                "labeled_test_disease_box_miss_rate_mean",
            ],
            ascending=[False, False, True, True],
        )
        best_rows.append(candidates.iloc[0])
    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows).reset_index(drop=True)


def choose_if_complete(summary: pd.DataFrame, expected_seed_count: int, sort_by: list[str], ascending: list[bool]) -> str:
    if summary.empty:
        return "Not available: no evaluation rows found."
    if not bool((summary["n_seeds"] >= expected_seed_count).all()):
        return "Not claimed yet: at least one module is missing one or more of the three seeds."
    candidates = summary.sort_values(by=sort_by, ascending=ascending)
    if candidates.empty:
        return "Not available: no complete candidate rows."
    row = candidates.iloc[0]
    return f"{row['display_name']} (`{row['module']}`)"


def write_final_ranking(
    path: Path,
    summary: pd.DataFrame,
    best_thresholds: pd.DataFrame,
    errors: pd.DataFrame,
    train_runs: pd.DataFrame,
    config: dict[str, Any],
    seeds: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_seed_count = len(seeds)
    default_conf = config["evaluation"]["default_conf"]
    default_iou = config["evaluation"]["default_iou"]

    lines: list[str] = []
    lines.append("# Top-4 Multiseed Final Ranking")
    lines.append("")
    lines.append("This report is generated from threshold-sweep CSV outputs. It does not train or evaluate models by itself.")
    lines.append("")
    lines.append(f"- Required seeds: `{', '.join(map(str, seeds))}`")
    lines.append(f"- Default threshold for multiseed summary: `conf={default_conf}`, `iou={default_iou}`")
    lines.append("- Final winners are not claimed unless every top-4 module has all three default-threshold seed rows.")
    lines.append("")

    if summary.empty or summary["n_seeds"].sum() == 0:
        lines.append("## Data Completeness")
        lines.append("")
        lines.append("No completed sweep rows were found yet. Run training and evaluation first.")
        lines.append("")
        lines.append("```bash")
        lines.append("python custom_attention_multiseed/run_top4_multiseed.py")
        lines.append("python custom_attention_multiseed/evaluate_top4_threshold_sweep.py")
        lines.append("python custom_attention_multiseed/aggregate_multiseed_results.py")
        lines.append("```")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    completeness_cols = ["module", "display_name", "n_seeds", "expected_seeds", "present_seeds", "missing_seeds", "complete_3seed"]
    lines.append("## Data Completeness")
    lines.append("")
    lines.append(markdown_table(summary, completeness_cols))
    lines.append("")

    perf = summary.sort_values(
        by=["healthy_aware_score_mean", "labeled_test_mask_map50_mean", "full_test_mask_map50_mean"],
        ascending=[False, False, False],
    )
    stable = summary.sort_values(
        by=["stability_std_sum", "healthy_test_healthy_mask_fp_rate_mean", "labeled_test_disease_box_miss_rate_mean"],
        ascending=[True, True, True],
    )
    deploy = summary.sort_values(
        by=["deployment_rank_sum", "healthy_aware_score_mean"],
        ascending=[True, False],
    )

    lines.append("## Default-Threshold Mean +/- Std")
    lines.append("")
    compact = summary.copy()
    compact["healthy_aware_score"] = compact.apply(lambda r: fmt_mean_std(r["healthy_aware_score_mean"], r["healthy_aware_score_std"]), axis=1)
    compact["test_mask_mAP50"] = compact.apply(lambda r: fmt_mean_std(r["labeled_test_mask_map50_mean"], r["labeled_test_mask_map50_std"]), axis=1)
    compact["full_test_mask_mAP50"] = compact.apply(lambda r: fmt_mean_std(r["full_test_mask_map50_mean"], r["full_test_mask_map50_std"]), axis=1)
    compact["healthy_FP"] = compact.apply(
        lambda r: fmt_mean_std(r["healthy_test_healthy_mask_fp_rate_mean"], r["healthy_test_healthy_mask_fp_rate_std"]),
        axis=1,
    )
    compact["miss_rate"] = compact.apply(
        lambda r: fmt_mean_std(r["labeled_test_disease_box_miss_rate_mean"], r["labeled_test_disease_box_miss_rate_std"]),
        axis=1,
    )
    lines.append(markdown_table(compact.sort_values("healthy_aware_score_mean", ascending=False), [
        "module",
        "n_seeds",
        "healthy_aware_score",
        "test_mask_mAP50",
        "full_test_mask_mAP50",
        "healthy_FP",
        "miss_rate",
    ]))
    lines.append("")

    lines.append("## Ranking Tables")
    lines.append("")
    lines.append("### Mean Healthy-Aware Score")
    lines.append(markdown_table(perf, RANKING_COLUMNS, max_rows=10))
    lines.append("")
    lines.append("### Stability")
    lines.append(markdown_table(stable, RANKING_COLUMNS, max_rows=10))
    lines.append("")
    lines.append("### Deployment Trade-Off")
    lines.append(markdown_table(deploy, RANKING_COLUMNS, max_rows=10))
    lines.append("")

    lines.append("## Required Conclusions")
    lines.append("")
    lines.append("### Best performance model")
    lines.append("")
    lines.append(choose_if_complete(
        perf,
        expected_seed_count,
        ["healthy_aware_score_mean", "labeled_test_mask_map50_mean", "full_test_mask_map50_mean"],
        [False, False, False],
    ))
    lines.append("")
    lines.append("### Best stable model")
    lines.append("")
    lines.append(choose_if_complete(
        stable,
        expected_seed_count,
        ["stability_std_sum", "healthy_test_healthy_mask_fp_rate_mean", "labeled_test_disease_box_miss_rate_mean"],
        [True, True, True],
    ))
    lines.append("")
    lines.append("### Best deployment candidate")
    lines.append("")
    lines.append(choose_if_complete(
        deploy,
        expected_seed_count,
        ["deployment_rank_sum", "healthy_aware_score_mean"],
        [True, False],
    ))
    lines.append("")

    lines.append("## Best Threshold Per Module")
    lines.append("")
    if best_thresholds.empty:
        lines.append("_No threshold sweep rows found._")
    else:
        lines.append(markdown_table(best_thresholds, [
            "module",
            "display_name",
            "conf",
            "iou",
            "n_seeds",
            "complete_3seed",
            "healthy_aware_score_mean",
            "labeled_test_mask_map50_mean",
            "full_test_mask_map50_mean",
            "healthy_test_healthy_mask_fp_rate_mean",
            "labeled_test_disease_box_miss_rate_mean",
            "labeled_test_mask_count_mae_mean",
        ]))
    lines.append("")

    if not train_runs.empty:
        lines.append("## Training Run Status")
        lines.append("")
        status_cols = ["module", "seed", "status", "output_dir", "best_checkpoint", "error"]
        lines.append(markdown_table(train_runs, status_cols, max_rows=40))
        lines.append("")

    if not errors.empty:
        lines.append("## Missing Checkpoints / Evaluation Errors")
        lines.append("")
        error_cols = ["module", "seed", "conf", "iou", "status", "checkpoint", "error_type", "error"]
        lines.append(markdown_table(errors, error_cols, max_rows=60))
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    seeds = load_seeds(args.seeds)
    modules = selected_modules(config)

    sweep_csv = repo_path(config["evaluation"]["threshold_sweep_csv"])
    summary_csv = repo_path(config["evaluation"]["summary_csv"])
    ranking_md = repo_path(config["evaluation"]["final_ranking_md"])
    train_csv = repo_path(config["training"]["train_results_csv"])
    errors_csv = repo_path(config["evaluation"]["threshold_sweep_errors_csv"])
    best_threshold_csv = (ranking_md.parent / "top4_threshold_best_by_module.csv") if ranking_md else None
    assert sweep_csv is not None and summary_csv is not None and ranking_md is not None and best_threshold_csv is not None

    ensure_empty_csv(sweep_csv, SWEEP_OUTPUT_COLUMNS)
    sweep = dedupe_sweep(read_csv_optional(sweep_csv))
    train_runs = read_csv_optional(train_csv) if train_csv else pd.DataFrame()
    errors = read_csv_optional(errors_csv) if errors_csv else pd.DataFrame()

    default_rows = default_threshold_rows(
        sweep,
        default_conf=float(config["evaluation"]["default_conf"]),
        default_iou=float(config["evaluation"]["default_iou"]),
    )
    summary = summarize_default_rows(default_rows, modules, seeds)
    best_thresholds = threshold_best_by_module(sweep, modules, seeds)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    best_thresholds.to_csv(best_threshold_csv, index=False)
    write_final_ranking(ranking_md, summary, best_thresholds, errors, train_runs, config, seeds)

    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote best thresholds: {best_threshold_csv}")
    print(f"Wrote final ranking: {ranking_md}")
    incomplete = summary.loc[~summary["complete_3seed"], ["module", "missing_seeds"]]
    if not incomplete.empty:
        print("Incomplete modules:")
        print(incomplete.to_string(index=False))


if __name__ == "__main__":
    main()
