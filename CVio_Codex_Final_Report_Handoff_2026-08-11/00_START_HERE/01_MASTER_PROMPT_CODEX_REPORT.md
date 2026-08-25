# MASTER TASK FOR CODEX - CVio Final Report

You are the lead technical editor, academic-paper engineer, XeLaTeX maintainer, evidence auditor, and PDF preflight reviewer for the CVio capstone project.

## Mission
Rebuild the integrated Final Report as a clean native XeLaTeX project, compile it locally, validate it aggressively, and push the final source + PDF + validation reports to a NEW GitHub branch.

Read, in this order:
1. repository `AGENTS.md`;
2. `00_START_HERE/README_FIRST.md`;
3. `01_CONTEXT/CVio_Project_Context.md`;
4. `01_CONTEXT/Report7_Repair_Requirements.md`;
5. `01_CONTEXT/Council_Review3_Classification_Summary.md`;
6. current `02_BASELINE/Report7.pdf` and FPT template;
7. evidence tables/archives under `03_EVIDENCE/`.

Use `$cvio-academic-report` for authoring and `$cvio-pdf-preflight` for the mandatory validation loop.

## Phase A - environment and repository
- Verify `codex`, `git`, `gh`, `xelatex`, `latexmk`, `python`, `qpdf`, `pdfinfo`, `pdffonts`, `pdftoppm`.
- If missing, run the supplied PowerShell installer or install equivalent packages.
- Verify `gh auth status`. If not authenticated, stop and ask the user to complete `gh auth login`; never invent credentials.
- Clone the CVio repository if absent.
- Fetch origin.
- Checkout the base branch `paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp` if it exists, otherwise inspect the default branch and report the fallback.
- Create new branch `report/final-xelatex-council-repair-2026-08-11`; if it exists, append `-v2`, `-v3`, etc.
- Never force-push.

## Phase B - asset recovery BEFORE rewriting
- Inspect all pages of `Report7.pdf`.
- Extract the exact official transparent logo from `Template_AIP491_CP_StudentsGuide.docx` at `word/media/image1.png`.
- Build a complete figure inventory from Report7.
- Recover every embedded figure or crop/vector fallback before rebuilding pages.
- Explicitly recover Figure 6.21 and split it into large BG / Healthy / WSSV panels if the original is a tall composite.
- A caption with no visible figure is a hard failure.

## Phase C - native XeLaTeX rebuild
Create a maintainable structure under `reports/final_report_xelatex/`:
- `main.tex`
- `preamble.tex`
- `references.bib`
- `chapters/*.tex`
- `figures/`
- `tables/`
- `scripts/`
- `validation/`
- `final/`

Do NOT assemble the body using `pdfpages` from Report1-Report6. Do NOT reproduce text by PDF x/y coordinates.

### Institutional vs IEEE
This is an FPT thesis, not an IEEE two-column paper. Preserve FPT A4, cover, front matter, one-column thesis structure, chapter numbering, and chapter-based equation numbering. Apply IEEE-quality academic writing, math typesetting, numeric citations, references, figure/table discipline, and source attribution where compatible.

### Typography
- Times New Roman for normal prose, headings, captions, tables, and literal filenames.
- Filenames/patterns such as `Disease-ShrimpID-img-ImageNumber.jpg`, `yolo11n-seg.pt`, `yolo26m_asl_ldam_simam_dcfr_fp32.tflite` stay in the same Times family; do not switch to monospace.
- Use `xurl`/`\path{}` with `\urlstyle{same}` if line breaking is needed.
- Use a Times-compatible math font.
- Embed all fonts.

### Equations
Use semantic AMS-LaTeX (`equation`, `align`, `aligned`, `split`, `multline`, `cases`). Never use `eqnarray` or `$$...$$`. Preserve chapter numbering `(4.1)`, `(4.2)`, etc. Use upright textual math identifiers (`\mathrm`, `\operatorname`, `\DeclareMathOperator`). Ensure Eq. 4.18 label smoothing has one clean `\bar{q}_{ik}` and no overlapping duplicate glyphs.

### Tables
- Center table contents horizontally and vertically by default.
- Keep normal body text legible (roughly >=9pt in dense tables).
- Prefer `tabularx`, `xltabular`, `longtable`, `booktabs`, `array`, `makecell`.
- Do not use `\resizebox` as the default solution.
- Short tables should not orphan one row on the next page.
- Update attention epoch presentation to `Executed / configured epochs`: 60/100, 100/100, 75/100, 200/200, 184/200 as applicable.

### Spacing
- Use consistent readable body line spacing around 1.15-1.20 unless the FPT template clearly dictates another value.
- Prevent giant subsection gaps, ultra-tight paragraphs, widows/orphans, and fragments such as `The system` isolated at a page edge.
- Use `raggedbottom`, sensible heading spacing, and `needspace`; avoid manual coordinate/negative-vspace hacks.

### Required explicit content corrections in this repair baseline
- State SDI-4 classification split as approximately 70/15/15 = 804/172/173, seed 42.
- Remove Conclusion subsection `3.5 Prioritized Future Work` and update the parent heading/TOC as appropriate.
- Normalize extraction-damaged tokens such as `MacroF1`, `EXT3Original`, `predictionlevel`, `173image`, `crossentropy`, etc. to correct academic forms.
- Preserve current accepted scientific claims unless a change is explicitly supported by handoff evidence.

## Phase D - mandatory validation gates
Create and run scripts. Hard-fail on any violation.

### Source gate
- missing inputs/assets;
- duplicate labels;
- unmatched environments/braces;
- raw `$$`;
- `eqnarray`;
- prohibited `pdfpages` body assembly;
- unsafe Unicode/soft hyphens/zero-width characters;
- known malformed merged tokens;
- manual page counter resets after Chapter I.

### Compile gate
Run `latexmk -xelatex -interaction=nonstopmode -file-line-error -halt-on-error main.tex` until references/TOC/LoT/LoF stabilize.

### Log gate
Fail on: Undefined control sequence, Fatal error, Overfull hbox/vbox, Float too large, too many unprocessed floats, undefined citations/references, multiply defined labels/destinations.

### PDF gate
Run qpdf, pdfinfo, pdffonts. Require valid A4 PDF, no encryption, embedded fonts.

### Automated layout gate
Use PyMuPDF to verify:
- cover has no page number;
- Roman front matter;
- one Arabic page number per body page;
- no duplicate footer number and no `9A`;
- no body text in footer exclusion zone;
- no objects outside physical page;
- no caption-only/missing-figure areas;
- no suspicious giant whitespace on text-heavy pages;
- no known malformed tokens in final text extraction.

### Visual gate
Render EVERY page at 180-200 dpi with Poppler/PDFium. Inspect contact sheets plus individual high-risk pages: cover, WBS tables, Chapter IV equations, filename-pattern page, attention tables, Chapter VI 4.2-4.4, Classification Error Analysis, XAI, Figures 6.16-6.21, Threats to Validity, Conclusion/Limitations, References, Appendices.

If any gate fails: fix, rebuild, rerun all gates. Do not ship a known defect.

## Phase E - Git delivery
Only after all gates PASS:
- include native `.tex`, `.bib`, scripts, figures, validation reports, and final PDF;
- create `README_BUILD.md` with exact build and validation commands;
- `git status` and review diff;
- commit with a descriptive message;
- push the NEW branch normally;
- optionally open a PR using `gh pr create` only if authenticated and the user has not prohibited it;
- report branch name, commit SHA, final PDF path, validation status, and any remaining evidence limitations.

Never claim PASS without generated machine-readable validation output and actual full-page render review.
