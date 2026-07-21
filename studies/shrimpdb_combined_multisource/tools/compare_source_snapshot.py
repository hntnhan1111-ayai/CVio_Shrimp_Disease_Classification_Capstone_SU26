#!/usr/bin/env python3
"""Compare the in-tree cvio_asl_ldam implementation to the packaged source snapshot.

Produces:
    artifacts/metadata/source_comparison.json
with SHA-256 hashes, content diffs, missing files, and an aggregate
deterministic tree hash.

Exit codes:
    0 -> documented match (status == "match" or "match_with_explicit_diff")
    1 -> undocumented content mismatch
    2 -> user/argument error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

IGNORE_PATH_PREFIXES = (
    "__pycache__/",
    ".pytest_cache/",
)

ALLOWLISTED_DIFFS = {
    # The repo re-exports additional symbols that the snapshot did not surface
    # via __init__.py. No behavior change; documented in source_comparison.json.
    "data/__init__.py": "repository adds explicit re-exports of audit_dataset, create_split, materialize_split_tree; no behavior change",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(rel.startswith(pref) for pref in IGNORE_PATH_PREFIXES):
            continue
        out[rel] = _sha256_file(path)
    return out


def _tree_hash(file_hashes: dict[str, str]) -> str:
    """Deterministic aggregate hash: sorted by relative path."""
    joined = "\n".join(f"{rel} {h}" for rel, h in sorted(file_hashes.items())).encode("utf-8")
    return _sha256_bytes(joined)


def run(
    repo_root: Path,
    snapshot_root: Path,
    notebook: Path,
    out_path: Path,
) -> int:
    if not snapshot_root.is_dir():
        LOGGER.error("Source snapshot root missing: %s", snapshot_root)
        return 2
    pkg_src = snapshot_root / "src" / "cvio_asl_ldam"
    repo_src = repo_root / "src" / "cvio_asl_ldam"
    if not pkg_src.is_dir():
        LOGGER.error("Snapshot has no src/cvio_asl_ldam: %s", pkg_src)
        return 2
    if not repo_src.is_dir():
        LOGGER.error("Repo has no src/cvio_asl_ldam: %s", repo_src)
        return 2

    pkg_hashes = _list_files(pkg_src)
    repo_hashes = _list_files(repo_src)

    compared = sorted(set(pkg_hashes) | set(repo_hashes))
    missing_in_repo = [r for r in compared if r in pkg_hashes and r not in repo_hashes]
    missing_in_pkg = [r for r in compared if r in repo_hashes and r not in pkg_hashes]
    mismatches = [
        {"path": r, "snapshot_sha256": pkg_hashes.get(r), "repository_sha256": repo_hashes.get(r)}
        for r in compared
        if r in pkg_hashes and r in repo_hashes and pkg_hashes[r] != repo_hashes[r]
    ]

    undocumented = [m for m in mismatches if m["path"] not in ALLOWLISTED_DIFFS]

    notebook_sha = _sha256_file(notebook) if notebook.is_file() else None
    pkg_tree = _tree_hash(pkg_hashes)
    repo_tree = _tree_hash(repo_hashes)

    status = "match"
    explanation_parts: list[str] = []
    if missing_in_repo or missing_in_pkg:
        status = "divergence_in_inventory"
        explanation_parts.append(
            "File inventory differs: snapshot={p}, repo={r}".format(
                p=len(pkg_hashes), r=len(repo_hashes)
            )
        )
    if undocumented:
        status = "content_mismatch"
        explanation_parts.append(f"{len(undocumented)} undocumented content mismatches")
    if mismatches and not undocumented:
        status = "match_with_explicit_diff"
        explanation_parts.append(
            "Allowlisted content differences (no behavior impact): "
            + ", ".join(sorted({m["path"] for m in mismatches}))
        )

    payload = {
        "status": status,
        "notebook_path": str(notebook.relative_to(repo_root)) if notebook.is_file() else None,
        "notebook_sha256": notebook_sha,
        "packaged_source_root": str(snapshot_root),
        "packaged_source_tree_sha256": pkg_tree,
        "repository_source_root": str((repo_root / "src" / "cvio_asl_ldam").relative_to(repo_root)),
        "repository_source_tree_sha256": repo_tree,
        "compared_files": compared,
        "missing_in_repository": missing_in_repo,
        "missing_in_package": missing_in_pkg,
        "content_mismatches": mismatches,
        "allowlisted_diffs": ALLOWLISTED_DIFFS,
        "comparison_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_tool": "tools/compare_source_snapshot.py",
        "explanation": "; ".join(explanation_parts) or "trees match exactly",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    if status == "content_mismatch":
        LOGGER.error("Source comparison failed with undocumented mismatches.")
        for m in undocumented:
            LOGGER.error("  - %s", m["path"])
        return 1
    LOGGER.info("Source comparison status: %s", status)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare in-tree source to packaged snapshot.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--snapshot-root",
        default="../../artifacts/source_snapshot",
        help="Path to packaged source snapshot.",
    )
    parser.add_argument(
        "--notebook",
        default="notebooks/CVio_Kaggle_AdaptiveGPU_ShrimpDB_Combined_ASL_LDAM_SimAM_DCFR_E2E_v7_RUNTIME_FIX.ipynb",
    )
    parser.add_argument(
        "--out",
        default="artifacts/metadata/source_comparison.json",
    )
    args = parser.parse_args()
    try:
        return run(
            repo_root=Path(args.repo_root).resolve(),
            snapshot_root=Path(args.snapshot_root).resolve(),
            notebook=Path(args.notebook).resolve(),
            out_path=Path(args.out).resolve(),
        )
    except Exception as exc:
        LOGGER.error("Source comparison error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

