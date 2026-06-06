"""Collect and rank ASL custom-loss screening outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.asl_custom_loss_screening.screening_lib import (
    DEFAULT_OUTPUT_DIR,
    collect_screening_results,
    filter_screening_runs,
    list_screening_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ASL custom-loss screening results.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--backend", choices=["torch", "yolo", "all"], default="all")
    parser.add_argument("--model", default=None)
    parser.add_argument("--loss", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = filter_screening_runs(
        list_screening_runs(smoke_test=args.smoke_test),
        backend=args.backend,
        model=args.model,
        loss=args.loss,
        run_id=args.run_id,
        start=args.start,
        limit=args.limit,
    )
    summary, failures, ranked = collect_screening_results(args.output_dir, rows=rows)
    print(json.dumps({
        "planned_count": len(rows),
        "completed_count": len(summary),
        "failed_or_missing_count": len(failures),
        "ranked_count": len(ranked),
        "output_dir": args.output_dir,
    }, indent=2))


if __name__ == "__main__":
    main()
