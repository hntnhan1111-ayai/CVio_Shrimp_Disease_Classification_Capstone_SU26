"""Create a pre-training manifest; this script never trains a model."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvio_shrimp_seg.candidates import candidate_by_id  # noqa: E402
from cvio_shrimp_seg.datasets import dataset_by_id  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an immutable run-plan record.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts/generated/run_manifests",
    )
    args = parser.parse_args()

    dataset = dataset_by_id(args.dataset)
    candidate = candidate_by_id(args.candidate)
    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": dataset.identifier,
        "candidate": candidate.identifier,
        "candidate_port_status": candidate.port_status,
        "seed": args.seed,
        "image_size": args.image_size,
        "threshold": args.threshold,
        "protocol": "strong_augmentation_v1",
        "training_started": False,
        "required_metrics": [
            "full_mask_map50",
            "disease_mask_map50",
            "disease_mask_map50_95",
            "healthy_fp_rate",
            "mask_count_mae",
            "disease_miss_rate",
            "healthy_aware",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / (
        f"{dataset.identifier}__{candidate.identifier}__seed{args.seed}.json"
    )
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
