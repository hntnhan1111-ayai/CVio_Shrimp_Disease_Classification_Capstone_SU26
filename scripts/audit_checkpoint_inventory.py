"""Inventory local checkpoint binaries, archives, references, Git, and LFS.

The script never imports or deserializes checkpoint content. Archive members are
hashed through streaming readers without extracting them into the repository.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import os
import re
import subprocess
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
EXPORT_SUFFIXES = {".onnx", ".tflite"}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".7z"}
TOKENS = (
    "asl",
    "ldam",
    "simam",
    "dcfr",
    "yolo26m",
    "classification",
    "external3",
    "ext3",
)
KNOWN_HASHES = (
    "4305e49158129c4a6acaa8fdaf7b5982a7f4d93cc233f11d17479e04b0e438c1",
    "055e22edd699bea623a4b1b8c34ad7a395b56658036a4495e5b54159b786c8ea",
    "ce0352be3fc20d2429605072fde4e01acce86216f4bdbaaec92895d1de98fb36",
)
TEXT_SUFFIXES = {
    ".csv",
    ".ipynb",
    ".json",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".playwright-cli",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
REFERENCE_PATTERN = re.compile(
    r"(?i)(?:best\.pt|last\.pt|\S+\.pth\b|\S+\.pt\b|"
    r"(?:^|[^a-z0-9])(?:asl|ldam|simam|dcfr|external3|ext[-_ ]?3)(?:[^a-z0-9]|$)|"
    r"yolo26m|" + "|".join(KNOWN_HASHES) + r")"
)
SENSITIVE_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|"
    r"client[_-]?secret|authorization\s*[:=]|bearer\s+[A-Za-z0-9]|ghp_[A-Za-z0-9])"
)


def sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def iso_mtime(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()


def candidate_reason(name: str) -> str | None:
    normalized = name.replace("\\", "/").lower()
    base = normalized.rsplit("/", 1)[-1]
    suffix = Path(base).suffix.lower()
    reasons: list[str] = []
    if base in {"best.pt", "last.pt"}:
        reasons.append(base)
    if suffix in CHECKPOINT_SUFFIXES:
        reasons.append(f"suffix:{suffix}")
    if suffix in EXPORT_SUFFIXES and any(token in normalized for token in TOKENS):
        reasons.append(f"deployment_export:{suffix}")
    token_hits = [token for token in TOKENS if token in normalized]
    if token_hits and suffix in CHECKPOINT_SUFFIXES | EXPORT_SUFFIXES:
        reasons.append("tokens:" + ",".join(token_hits))
    return ";".join(dict.fromkeys(reasons)) if reasons else None


def archive_kind(path: Path) -> str | None:
    lower = path.name.lower()
    if lower.endswith(".tar.gz"):
        return ".tar.gz"
    suffix = path.suffix.lower()
    return suffix if suffix in ARCHIVE_SUFFIXES else None


def iter_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[str] = set()
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for current, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
            for filename in filenames:
                path = Path(current, filename)
                key = os.path.normcase(os.path.abspath(path))
                if key not in seen:
                    seen.add(key)
                    yield path


def git_objects(repo: Path) -> list[tuple[str, str]]:
    process = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    rows: list[tuple[str, str]] = []
    for line in process.stdout.splitlines():
        oid, separator, path = line.partition(" ")
        if separator and candidate_reason(path):
            rows.append((oid, path))
    return rows


def git_blob_sha256(repo: Path, oid: str) -> tuple[int, str]:
    size = int(
        subprocess.check_output(
            ["git", "cat-file", "-s", oid], cwd=repo, text=True
        ).strip()
    )
    process = subprocess.Popen(
        ["git", "cat-file", "blob", oid], cwd=repo, stdout=subprocess.PIPE
    )
    assert process.stdout is not None
    digest = sha256_stream(process.stdout)
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"git cat-file failed for {oid}: {return_code}")
    return size, digest


def append_reference_hits(
    rows: list[list[object]], source_type: str, source: str, text: str
) -> None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        if REFERENCE_PATTERN.search(line) and not SENSITIVE_PATTERN.search(line):
            rows.append([source_type, source, line_number, line.strip()[:1000]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", action="append", default=[], type=Path)
    parser.add_argument("--archive", action="append", default=[], type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    roots = [path.resolve() for path in args.root]
    explicit_archives = [path.resolve() for path in args.archive if path.is_file()]

    path_rows: list[list[object]] = []
    hash_rows: list[list[object]] = []
    archive_rows: list[list[object]] = []
    reference_rows: list[list[object]] = []
    archives: dict[str, Path] = {
        os.path.normcase(str(path)): path for path in explicit_archives
    }

    for path in iter_files(roots):
        reason = candidate_reason(str(path))
        if reason:
            try:
                stat = path.stat()
                digest = sha256_file(path)
            except (OSError, PermissionError) as exc:
                path_rows.append(
                    ["filesystem_error", str(path), "", "", reason, repr(exc)]
                )
                continue
            path_rows.append(
                [
                    "filesystem",
                    str(path),
                    stat.st_size,
                    iso_mtime(stat.st_mtime),
                    reason,
                    "",
                ]
            )
            hash_rows.append(["filesystem", str(path), stat.st_size, digest])

        if archive_kind(path):
            archives.setdefault(os.path.normcase(str(path.resolve())), path.resolve())

        try:
            stat = path.stat()
            if (
                path.suffix.lower() in TEXT_SUFFIXES
                and stat.st_size <= 10 * 1024 * 1024
            ):
                text = path.read_text(encoding="utf-8", errors="replace")
                append_reference_hits(
                    reference_rows, "filesystem_text", str(path), text
                )
        except (OSError, PermissionError):
            pass

    for archive in sorted(archives.values(), key=lambda value: str(value).lower()):
        kind = archive_kind(archive)
        if kind == ".zip":
            try:
                with zipfile.ZipFile(archive) as handle:
                    for info in handle.infolist():
                        if info.is_dir():
                            continue
                        reason = candidate_reason(info.filename)
                        if reason:
                            with handle.open(info) as stream:
                                digest = sha256_stream(stream)
                            archive_rows.append(
                                [
                                    str(archive),
                                    info.filename,
                                    info.file_size,
                                    digest,
                                    reason,
                                    "ok",
                                ]
                            )
                        suffix = Path(info.filename).suffix.lower()
                        if (
                            suffix in TEXT_SUFFIXES
                            and info.file_size <= 10 * 1024 * 1024
                        ):
                            with handle.open(info) as stream:
                                text = stream.read().decode("utf-8", errors="replace")
                            append_reference_hits(
                                reference_rows,
                                "archive_text",
                                f"{archive}!{info.filename}",
                                text,
                            )
            except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
                archive_rows.append([str(archive), "", "", "", "", f"error:{exc!r}"])
        elif kind in {".tar", ".tgz", ".tar.gz"}:
            try:
                with tarfile.open(archive, "r:*") as handle:
                    for info in handle:
                        if not info.isfile():
                            continue
                        reason = candidate_reason(info.name)
                        suffix = Path(info.name).suffix.lower()
                        if reason or (
                            suffix in TEXT_SUFFIXES and info.size <= 10 * 1024 * 1024
                        ):
                            stream = handle.extractfile(info)
                            if stream is None:
                                continue
                            data = stream.read()
                            if reason:
                                archive_rows.append(
                                    [
                                        str(archive),
                                        info.name,
                                        info.size,
                                        hashlib.sha256(data).hexdigest(),
                                        reason,
                                        "ok",
                                    ]
                                )
                            if (
                                suffix in TEXT_SUFFIXES
                                and info.size <= 10 * 1024 * 1024
                            ):
                                append_reference_hits(
                                    reference_rows,
                                    "archive_text",
                                    f"{archive}!{info.name}",
                                    data.decode("utf-8", errors="replace"),
                                )
            except (OSError, EOFError, tarfile.TarError) as exc:
                archive_rows.append([str(archive), "", "", "", "", f"error:{exc!r}"])
        else:
            archive_rows.append(
                [str(archive), "", "", "", "", "listing_unavailable:no_7z_reader"]
            )

    for oid, path in git_objects(args.repo):
        size, digest = git_blob_sha256(args.repo, oid)
        source = f"git:{oid}:{path}"
        reason = candidate_reason(path) or "git_object"
        path_rows.append(["git_object", source, size, "", reason, ""])
        hash_rows.append(["git_object", source, size, digest])

    lfs_root = args.repo / ".git" / "lfs" / "objects"
    if lfs_root.is_dir():
        for path in iter_files([lfs_root]):
            try:
                stat = path.stat()
                digest = sha256_file(path)
            except (OSError, PermissionError):
                continue
            source = str(path)
            path_rows.append(
                [
                    "lfs_object",
                    source,
                    stat.st_size,
                    iso_mtime(stat.st_mtime),
                    "lfs_object",
                    "",
                ]
            )
            hash_rows.append(["lfs_object", source, stat.st_size, digest])

    tables = {
        "all_checkpoint_paths.tsv": (
            [
                "source_type",
                "path",
                "size_bytes",
                "mtime_utc",
                "candidate_reason",
                "error",
            ],
            path_rows,
        ),
        "all_checkpoint_hashes.tsv": (
            ["source_type", "path", "size_bytes", "sha256"],
            hash_rows,
        ),
        "archive_checkpoint_hits.tsv": (
            ["archive", "member", "size_bytes", "sha256", "candidate_reason", "status"],
            archive_rows,
        ),
        "checkpoint_reference_hits.tsv": (
            ["source_type", "source", "line", "reference"],
            reference_rows,
        ),
    }
    for filename, (header, rows) in tables.items():
        with (args.output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            writer.writerows(
                ["NA" if str(cell) == "" else str(cell).rstrip() for cell in row]
                for row in rows
            )

    print(f"filesystem_and_git_candidates={len(path_rows)}")
    print(f"archive_checkpoint_hits={sum(row[-1] == 'ok' for row in archive_rows)}")
    print(f"archives_inspected={len(archives)}")
    print(f"reference_hits={len(reference_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
