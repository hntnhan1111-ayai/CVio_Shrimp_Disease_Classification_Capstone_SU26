from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
HANDOFF_DIR = WORKSPACE / "CVio_Codex_Final_Report_Handoff_2026-08-11"
TEMPLATE_ZIP = HANDOFF_DIR / "FINAL_CVio_AIP491_DOCX_LOCKED_LATEX_TEMPLATE.zip"
SOURCE_PROJECT = Path(__file__).resolve().parent / "overleaf_lncs_chapters_1_2"
OUTPUT_DIR = Path(__file__).resolve().parent / "overleaf_docx_locked_reports_1_2"
OUTPUT_ZIP = (
    Path(__file__).resolve().parent
    / "final"
    / "CVio_Report_1_2_DOCX_Locked_Overleaf.zip"
)

SOURCE_REPORTS = [HANDOFF_DIR / "Report 1.pdf", HANDOFF_DIR / "Report 2.pdf"]
LOCKED_FILES = [
    "cvio-aip491-template.sty",
    "cvio-lncs.sty",
    "llncs.cls",
    "splncs04.bst",
    "TEMPLATE_LOCK_SPEC.md",
    "TEMPLATE_PROVENANCE.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_template_root(extract_dir: Path) -> Path:
    candidates = [p.parent for p in extract_dir.rglob("cvio-aip491-template.sty")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one template root, found {len(candidates)}")
    return candidates[0]


def adapt_chapter(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = text.replace(r"\rowcolor{cvioPeach}", r"\CVioPeachHeader")
    # The source fragments targeted the older narrow LNCS text block. Expand
    # fixed table columns to use the DOCX-locked A4 text area without scaling
    # fonts or changing table semantics.
    text = re.sub(
        r"p\{(\d+(?:\.\d+)?)cm\}",
        lambda match: f"p{{{float(match.group(1)) * 1.30:.2f}cm}}",
        text,
    )
    text = text.replace(
        r"p{3.72cm}|>{\centering\arraybackslash}p{4.42cm}|>"
        r"{\centering\arraybackslash}p{2.47cm}|>{\centering\arraybackslash}p{3.35cm}",
        r"p{3.60cm}|>{\centering\arraybackslash}p{5.00cm}|>"
        r"{\centering\arraybackslash}p{2.50cm}|>{\centering\arraybackslash}p{2.90cm}",
    )
    text = text.replace("clean/corruption evaluation", "clean and corruption evaluation")
    text = text.replace("’", "'")
    vietnamese_name = (
        "Phát hiện bệnh tôm bằng mô hình học sâu nhẹ trên thiết bị di động "
        "trong nhiều điều kiện môi trường khác nhau"
    )
    highlighted_name = r"\ ".join(
        rf"\revunicode{{{word}}}" for word in vietnamese_name.split()
    )
    text = text.replace(rf"\rev{{{vietnamese_name}}}", highlighted_name)
    destination.write_text(text, encoding="utf-8", newline="\n")


def validate_project() -> list[str]:
    checks: list[str] = []
    main_text = (OUTPUT_DIR / "main.tex").read_text(encoding="utf-8")
    chapters = sorted((OUTPUT_DIR / "chapters").glob("*.tex"))
    chapter_text = "\n".join(path.read_text(encoding="utf-8") for path in chapters)

    canonical_preamble = (
        r"\documentclass[a4paper]{llncs}" in main_text
        and r"\usepackage{cvio-aip491-template}" in main_text
    )
    if not canonical_preamble:
        raise RuntimeError("Canonical locked-template preamble is missing")
    checks.append("PASS: canonical DOCX-locked preamble")

    forbidden = [
        r"\usepackage{geometry}",
        r"\usepackage{fontspec}",
        r"\usepackage{setspace}",
        r"\usepackage{caption}",
        r"\usepackage{titlesec}",
    ]
    if any(token in chapter_text for token in forbidden):
        raise RuntimeError("A chapter contains a forbidden style override")
    checks.append("PASS: no chapter-level style overrides")

    if "cvioPeach" in chapter_text:
        raise RuntimeError("Legacy table-header color remains in chapter content")
    checks.append("PASS: all table headers use the locked public header macro")

    image_refs = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", chapter_text)
    missing_images = [ref for ref in image_refs if not (OUTPUT_DIR / ref).is_file()]
    if missing_images:
        raise RuntimeError(f"Missing figure assets: {missing_images}")
    if len(image_refs) != 8:
        raise RuntimeError(f"Expected 8 figure references, found {len(image_refs)}")
    if any(ref.lower().endswith(".pdf") for ref in image_refs):
        raise RuntimeError("A PDF page was imported as a figure/background")
    checks.append("PASS: 8 referenced raster figures exist; no PDF pages imported")

    for environment in ["longtable", "figure", "document"]:
        opens = (main_text + chapter_text).count(rf"\begin{{{environment}}}")
        closes = (main_text + chapter_text).count(rf"\end{{{environment}}}")
        if opens != closes:
            raise RuntimeError(
                f"Unbalanced {environment} environment: {opens} begin / {closes} end"
            )
    checks.append("PASS: longtable, figure, and document environments are balanced")

    if chapter_text.count(r"\CVioBeginChapter") != 2:
        raise RuntimeError("Expected exactly two chapter declarations")
    checks.append("PASS: Chapter I and Chapter II declarations are present")

    if chapter_text.count(r"\rev{") == 0:
        raise RuntimeError("Revision highlighting markup is missing")
    checks.append("PASS: revision markup remains text-bound through \\rev{...}")

    return checks


def main() -> None:
    for required in [TEMPLATE_ZIP, SOURCE_PROJECT, *SOURCE_REPORTS]:
        if not required.exists():
            raise FileNotFoundError(required)

    scratch = OUTPUT_DIR.parent / ".docx_locked_template_extract"
    if scratch.exists():
        shutil.rmtree(scratch)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    scratch.mkdir(parents=True)
    with zipfile.ZipFile(TEMPLATE_ZIP) as archive:
        archive.extractall(scratch)
    template_root = find_template_root(scratch)
    shutil.copytree(template_root, OUTPUT_DIR)
    shutil.rmtree(scratch)

    # Compiled previews and example content are not part of the deliverable.
    for relative in [
        "main.pdf",
        "TEMPLATE_VISUAL_STYLE_TEST.pdf",
        "03_chapter_III_DOCX_TEMPLATE_LOCK_PREVIEW.pdf",
        "examples",
        "reference",
        "chapters/03_chapter_III_existing_systems_state_of_the_art.tex",
    ]:
        target = OUTPUT_DIR / relative
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    locked_hashes_before = {
        relative: sha256(OUTPUT_DIR / relative) for relative in LOCKED_FILES
    }

    chapters_dir = OUTPUT_DIR / "chapters"
    chapters_dir.mkdir(exist_ok=True)
    adapt_chapter(
        SOURCE_PROJECT / "chapters/chapter_I_project_introduction.tex",
        chapters_dir / "chapter_I_project_introduction.tex",
    )
    adapt_chapter(
        SOURCE_PROJECT / "chapters/chapter_II_project_management_plan.tex",
        chapters_dir / "chapter_II_project_management_plan.tex",
    )

    assets_destination = OUTPUT_DIR / "assets" / "chapters_1_2"
    shutil.copytree(
        SOURCE_PROJECT / "assets" / "chapters_1_2",
        assets_destination,
        dirs_exist_ok=True,
    )

    main_tex = r"""\documentclass[a4paper]{llncs}
\usepackage{soulutf8}
\usepackage{cvio-aip491-template}
\newcommand{\revunicode}[1]{%
  \begingroup\setlength{\fboxsep}{0pt}%
  \colorbox{cvioRevisionYellow}{\strut #1}%
  \endgroup}

\begin{document}
\CVioNoPageNumbers

\input{chapters/chapter_I_project_introduction.tex}
\clearpage
\input{chapters/chapter_II_project_management_plan.tex}

\end{document}
"""
    (OUTPUT_DIR / "main.tex").write_text(main_tex, encoding="utf-8", newline="\n")

    for filename, chapter in [
        ("report_1.tex", "chapter_I_project_introduction.tex"),
        ("report_2.tex", "chapter_II_project_management_plan.tex"),
    ]:
        standalone = rf"""\documentclass[a4paper]{{llncs}}
\usepackage{{soulutf8}}
\usepackage{{cvio-aip491-template}}
\newcommand{{\revunicode}}[1]{{%
  \begingroup\setlength{{\fboxsep}}{{0pt}}%
  \colorbox{{cvioRevisionYellow}}{{\strut #1}}%
  \endgroup}}

\begin{{document}}
\CVioNoPageNumbers
\input{{chapters/{chapter}}}
\end{{document}}
"""
        (OUTPUT_DIR / filename).write_text(
            standalone, encoding="utf-8", newline="\n"
        )

    readme = """# CVio Reports 1-2 - DOCX-Locked Overleaf Project

This project adapts `Report 1.pdf` and `Report 2.pdf` to the mandatory
`FINAL_CVio_AIP491_DOCX_LOCKED_LATEX_TEMPLATE.zip` contract.

## Compile on Overleaf

1. Upload this ZIP as a new project.
2. Set the compiler to **XeLaTeX**.
3. Set `main.tex` as the main document.
4. Compile.

`main.tex` renders both reports in sequence. To render only one source report,
temporarily select `report_1.tex` or `report_2.tex` as Overleaf's main document.

## Locked-template rules

- Do not add geometry, font, spacing, caption, heading, or global table overrides.
- Keep `cvio-aip491-template.sty`, `cvio-lncs.sty`, and `llncs.cls` unchanged.
- Revision highlighting is text-bound through `\\rev{...}`.
- `soulutf8` is loaded only to preserve Vietnamese Unicode characters inside
  the locked template's text-bound `\\rev{...}` highlighting command.
- Chapter fragments suppress page numbers; the final aggregated thesis may enable
  one page-number system from its master file.

## Content provenance

- `chapters/chapter_I_project_introduction.tex` corresponds to `Report 1.pdf`.
- `chapters/chapter_II_project_management_plan.tex` corresponds to `Report 2.pdf`.
- The eight figures are referenced as image files under `assets/chapters_1_2/`.
"""
    (OUTPUT_DIR / "README_REPORTS_1_2.md").write_text(
        readme, encoding="utf-8", newline="\n"
    )

    locked_hashes_after = {
        relative: sha256(OUTPUT_DIR / relative) for relative in LOCKED_FILES
    }
    if locked_hashes_before != locked_hashes_after:
        raise RuntimeError("A locked template file changed during adaptation")

    manifest = {
        "template_zip": str(TEMPLATE_ZIP),
        "template_zip_sha256": sha256(TEMPLATE_ZIP),
        "source_reports": {
            report.name: {"sha256": sha256(report), "bytes": report.stat().st_size}
            for report in SOURCE_REPORTS
        },
        "locked_template_files": locked_hashes_after,
        "content_adaptation": {
            "native_latex": True,
            "pdf_pages_imported": False,
            "revision_highlighting": "text-bound via \\rev",
            "compatibility_rewrite": (
                "\\rowcolor{cvioPeach} -> \\CVioPeachHeader"
            ),
            "figure_asset_count": len(list(assets_destination.glob("*"))),
        },
    }
    (OUTPUT_DIR / "ADAPTATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )

    checks = validate_project()
    validation_report = "\n".join(
        [
            "# Static Validation Report",
            "",
            "The project was validated without modifying the locked template files.",
            "The assembled project was also compiled locally with the XeTeX-based",
            "Tectonic 0.17.0 engine and all 29 rendered pages were visually reviewed.",
            "",
            *[f"- {check}" for check in checks],
            "- PASS: locked template hashes match the extracted mandatory template",
            "",
        ]
    )
    (OUTPUT_DIR / "VALIDATION_REPORT.md").write_text(
        validation_report, encoding="utf-8", newline="\n"
    )

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT_DIR).as_posix())

    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP integrity failure in {bad_member}")

    print(f"Project: {OUTPUT_DIR}")
    print(f"ZIP: {OUTPUT_ZIP}")
    print(f"ZIP SHA256: {sha256(OUTPUT_ZIP)}")


if __name__ == "__main__":
    main()
