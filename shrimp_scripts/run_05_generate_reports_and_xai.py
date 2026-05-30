"""Aggregate outputs, generate selected XAI, tables, figures, Excel, report, and zip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shrimp_scripts import config
from shrimp_scripts.progress import log_event
from shrimp_scripts.report import generate_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final paper reports, selected XAI, and zip package.")
    parser.add_argument("--output_dir", default=str(config.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.progress:
        log_event("Starting report/XAI generation script.", output_dir=args.output_dir, extra={"dry_run": args.dry_run})
    payload = generate_reports(args.output_dir, dry_run=args.dry_run, progress_enabled=args.progress)
    if args.progress:
        log_event("Report/XAI generation script completed.", output_dir=args.output_dir, extra=payload)
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
