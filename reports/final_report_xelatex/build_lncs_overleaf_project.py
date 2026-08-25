from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "final_report_xelatex"
TEMPLATE_ZIP = ROOT / "CVio_Codex_Final_Report_Handoff_2026-08-11" / "CVio_LNCS_Hybrid_Template_Overleaf.zip"
SOURCE_TEX = REPORT / "chapters_1_2_review.tex"
SOURCE_ASSETS = REPORT / "assets" / "chapters_1_2"
PROJECT = REPORT / "overleaf_lncs_chapters_1_2"
OUTPUT_ZIP = REPORT / "final" / "CVio_Chapters_I_II_LNCS_Overleaf.zip"
TEMPLATE_ROOT = "CVio_LNCS_Hybrid_Template/"


MAIN_TEX = r"""\documentclass[a4paper]{llncs}
\usepackage{cvio-lncs}

\begin{document}
\CVioNoPageNumbers

\input{chapters/chapter_I_project_introduction.tex}
\clearpage
\input{chapters/chapter_II_project_management_plan.tex}

\end{document}
"""


PROJECT_README = r"""# CVio Chapters I-II - Overleaf project

This upload package uses the supplied CVio LNCS hybrid template.

## Compile on Overleaf

1. Upload `CVio_Chapters_I_II_LNCS_Overleaf.zip` as a new project.
2. Set the compiler to **XeLaTeX**.
3. Compile `main.tex`.

## Structure

- `main.tex`: project entry point.
- `cvio-lncs.sty`: shared CVio/LNCS formatting rules from the supplied template.
- `llncs.cls`: supplied Springer LNCS class.
- `chapters/`: semantic Chapter I and Chapter II sources.
- `assets/chapters_1_2/`: extracted source figures referenced by relative image paths.

Yellow revision marking uses `\rev{...}` and is tight to the text. No revision row or cell uses a yellow background fill. Peach table headers remain normal table styling rather than revision marking.

The template hides page numbers for standalone chapter review. Remove `\CVioNoPageNumbers` only when these chapters are integrated into a full-report pagination workflow.
"""


def reset_directory(path: Path) -> None:
    resolved = path.resolve()
    if REPORT.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing to reset path outside report workspace: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def extract_template_files() -> None:
    keep = {
        "cvio-lncs.sty",
        "llncs.cls",
        "splncs04.bst",
    }
    with zipfile.ZipFile(TEMPLATE_ZIP) as archive:
        for member in archive.infolist():
            if not member.filename.startswith(TEMPLATE_ROOT):
                continue
            relative = member.filename[len(TEMPLATE_ROOT):]
            if relative not in keep:
                continue
            target = PROJECT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))


def text_bound_row_highlight(match: re.Match[str]) -> str:
    row = match.group(1)
    cells = row.split(" & ")
    converted: list[str] = []
    for cell in cells:
        cell = cell.strip()
        if not cell:
            converted.append(cell)
            continue
        bold = re.fullmatch(r"\\textbf\{(.*)\}", cell)
        if bold:
            converted.append(r"\textbf{\rev{" + bold.group(1) + "}}")
        else:
            converted.append(r"\rev{" + cell + "}")
    return " & ".join(converted) + r" \\ \hline"


def convert_body(body: str) -> str:
    body = body.replace(r"\begin{revisionblock}", r"\rev{")
    body = body.replace(r"\end{revisionblock}", "}")
    body = re.sub(
        r"\\texorpdfstring\{\\colorbox\{RevisionYellow\}\{([^{}]*)\}\}\{[^{}]*\}",
        lambda match: r"\rev{" + match.group(1) + "}",
        body,
    )
    body = re.sub(
        r"\\colorbox\{RevisionYellow\}\{\\parbox\{0\.91\\linewidth\}\{(.*?)\}\}",
        lambda match: r"\rev{" + match.group(1) + "}",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(
        r"\\rowcolor\{RevisionYellow\}(.+?) \\\\ \\hline",
        text_bound_row_highlight,
        body,
    )
    body = body.replace("HeaderPeach", "cvioPeach")
    body = body.replace("RevisionYellow", "cvioRevisionYellow")
    body = body.replace(r"\begin{figure}[H]", r"\begin{figure}[tbp]")
    body = body.replace("AIP491 01 \\_", "AIP491\\_01")
    body = body.replace("học sâu nhẹtrên", "học sâu nhẹ trên")

    body = re.sub(
        r"\\begingroup\n\\footnotesize\n\\setlength\{\\tabcolsep\}\{2pt\}\n\\renewcommand\{\\arraystretch\}\{1\.16\}",
        r"\\begingroup\n\\CVioTableSetup",
        body,
    )
    body = re.sub(
        r"\\begingroup\n\\scriptsize\n\\setlength\{\\tabcolsep\}\{2pt\}\n\\renewcommand\{\\arraystretch\}\{1\.16\}",
        r"\\begingroup\n\\CVioTableSetup\n\\scriptsize",
        body,
    )

    # The LNCS one-column text block is narrower than the former A4 report block.
    # Scale only fixed p-column widths; figure dimensions remain unchanged.
    def scale_column(match: re.Match[str]) -> str:
        width = float(match.group(1)) * 0.68
        return f"p{{{width:.2f}cm}}"

    body = re.sub(r"p\{([0-9]+(?:\.[0-9]+)?)cm\}", scale_column, body)
    return body.strip() + "\n"


def split_chapters(source: str) -> tuple[str, str]:
    document = source.split(r"\begin{document}", 1)[1].rsplit(r"\end{document}", 1)[0]
    first_marker = r"\chapter{Project Introduction}"
    second_marker = r"\chapter{Project Management Plan}"
    if first_marker not in document or second_marker not in document:
        raise RuntimeError("Expected Chapter I and Chapter II markers were not found")
    chapter_one, chapter_two = document.split(second_marker, 1)
    chapter_one = chapter_one.split(first_marker, 1)[1]
    chapter_one = r"\CVioBeginChapter{1}{I. Project Introduction}" + chapter_one
    chapter_two = r"\CVioBeginChapter{2}{II. Project Management Plan}" + chapter_two
    return convert_body(chapter_one), convert_body(chapter_two)


def write_project() -> None:
    reset_directory(PROJECT)
    extract_template_files()
    (PROJECT / "chapters").mkdir(parents=True, exist_ok=True)
    assets_target = PROJECT / "assets" / "chapters_1_2"
    shutil.copytree(SOURCE_ASSETS, assets_target)

    source = SOURCE_TEX.read_text(encoding="utf-8")
    chapter_one, chapter_two = split_chapters(source)
    (PROJECT / "main.tex").write_text(MAIN_TEX, encoding="utf-8", newline="\n")
    (PROJECT / "README.md").write_text(PROJECT_README, encoding="utf-8", newline="\n")
    (PROJECT / "chapters" / "chapter_I_project_introduction.tex").write_text(chapter_one, encoding="utf-8", newline="\n")
    (PROJECT / "chapters" / "chapter_II_project_management_plan.tex").write_text(chapter_two, encoding="utf-8", newline="\n")


def validate_project() -> None:
    required = {
        "main.tex",
        "cvio-lncs.sty",
        "llncs.cls",
        "splncs04.bst",
        "README.md",
        "chapters/chapter_I_project_introduction.tex",
        "chapters/chapter_II_project_management_plan.tex",
    }
    present = {path.relative_to(PROJECT).as_posix() for path in PROJECT.rglob("*") if path.is_file()}
    missing = required - present
    if missing:
        raise RuntimeError(f"Missing required project files: {sorted(missing)}")

    tex_files = [PROJECT / "main.tex", *sorted((PROJECT / "chapters").glob("*.tex"))]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
    forbidden = {
        r"\documentclass[12pt,a4paper]{report}": "old report class",
        r"\rowcolor{RevisionYellow}": "old full-row revision highlighting",
        r"\rowcolor{cvioRevisionYellow}": "full-row revision highlighting",
        r"\cellcolor{cvioRevisionYellow}": "full-cell revision highlighting",
        r"\begin{revisionblock}": "old block revision highlighting",
        r"\colorbox{cvioRevisionYellow}": "old box revision highlighting",
    }
    violations = [description for token, description in forbidden.items() if token in combined]
    if violations:
        raise RuntimeError(f"Forbidden source patterns remain: {violations}")

    if r"\documentclass[a4paper]{llncs}" not in combined:
        raise RuntimeError("main.tex does not use the supplied LNCS class")
    if r"\usepackage{cvio-lncs}" not in combined:
        raise RuntimeError("main.tex does not load the supplied CVio style")
    if combined.count(r"\CVioBeginChapter") != 2:
        raise RuntimeError("Expected exactly two CVio chapter declarations")
    if combined.count(r"\rev{") < 10:
        raise RuntimeError("Revision highlights were not migrated to text-bound \\rev markup")

    image_refs = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", combined)
    if len(image_refs) != 8:
        raise RuntimeError(f"Expected 8 image references, found {len(image_refs)}")
    unresolved = [ref for ref in image_refs if not (PROJECT / ref).is_file()]
    if unresolved:
        raise RuntimeError(f"Unresolved image references: {unresolved}")

    generated_suffixes = {".pdf", ".log", ".aux", ".out", ".xdv", ".synctex.gz"}
    generated = [path for path in PROJECT.rglob("*") if path.is_file() and any(path.name.endswith(suffix) for suffix in generated_suffixes)]
    if generated:
        raise RuntimeError(f"Generated compile artifacts should not be packaged: {generated}")


def create_zip() -> None:
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(PROJECT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(PROJECT).as_posix())
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failure at {bad}")
        names = archive.namelist()
        if "main.tex" not in names or "llncs.cls" not in names:
            raise RuntimeError("ZIP root is not Overleaf-ready")


def main() -> None:
    write_project()
    validate_project()
    create_zip()
    print(f"Project: {PROJECT}")
    print(f"ZIP: {OUTPUT_ZIP}")
    print(f"Files: {sum(1 for path in PROJECT.rglob('*') if path.is_file())}")
    print(f"ZIP size: {OUTPUT_ZIP.stat().st_size}")


if __name__ == "__main__":
    main()
