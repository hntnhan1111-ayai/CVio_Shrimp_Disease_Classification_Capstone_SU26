"""Verify released checkpoint hashes, roles, classes, and Git LFS pointers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "weights" / "manifest.json"
VERIFIED = {
    "VERIFIED_EXT3_ORIGINAL_CE",
    "VERIFIED_EXT3_ORIGINAL_PROPOSED",
}
EXPECTED_CLASSES = ["Healthy", "BG", "WSSV"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def staged_pointer(relative: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if len(data["entries"]) != 6:
        raise AssertionError(
            f"expected 6 manifest entries, found {len(data['entries'])}"
        )
    required = {
        "filename",
        "sha256",
        "dataset_regime",
        "task",
        "classes",
        "seed",
        "image_size",
        "architecture",
        "loss",
        "metrics",
        "test_size",
        "provenance",
        "verification_status",
        "intended_use",
        "limitations",
        "download_location",
    }
    verified_count = 0
    for entry in data["entries"]:
        missing = required.difference(entry)
        if missing:
            raise AssertionError(f"{entry.get('id')}: missing {sorted(missing)}")
        if entry["verification_status"] not in VERIFIED:
            continue
        verified_count += 1
        if entry["dataset_regime"] != "EXT-3-Original":
            raise AssertionError(f"{entry['id']}: verified release has wrong regime")
        if entry["classes"] != EXPECTED_CLASSES:
            raise AssertionError(f"{entry['id']}: wrong class order")
        if entry["test_size"] != 47:
            raise AssertionError(f"{entry['id']}: wrong test size")
        path = ROOT / entry["filename"]
        if not path.is_file():
            raise AssertionError(f"{entry['id']}: missing {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise AssertionError(f"{entry['id']}: {actual} != {entry['sha256']}")
        pointer = staged_pointer(entry["filename"])
        expected_oid = f"oid sha256:{entry['sha256']}"
        if (
            "version https://git-lfs.github.com/spec/v1" not in pointer
            or expected_oid not in pointer
        ):
            raise AssertionError(
                f"{entry['id']}: Git object is not the expected LFS pointer"
            )
        print(f"PASS {entry['filename']} {actual}")
    if verified_count != 2:
        raise AssertionError(f"expected 2 verified releases, found {verified_count}")
    print(f"manifest_entries={len(data['entries'])}")
    print("verified_release_entries=2")
    print("status=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
