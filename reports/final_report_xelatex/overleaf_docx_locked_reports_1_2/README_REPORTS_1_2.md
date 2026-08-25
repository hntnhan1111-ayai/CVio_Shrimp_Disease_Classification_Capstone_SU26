# CVio Reports 1-2 - DOCX-Locked Overleaf Project

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
- Revision highlighting is text-bound through `\rev{...}`.
- `soulutf8` is loaded only to preserve Vietnamese Unicode characters inside
  the locked template's text-bound `\rev{...}` highlighting command.
- Chapter fragments suppress page numbers; the final aggregated thesis may enable
  one page-number system from its master file.

## Content provenance

- `chapters/chapter_I_project_introduction.tex` corresponds to `Report 1.pdf`.
- `chapters/chapter_II_project_management_plan.tex` corresponds to `Report 2.pdf`.
- The eight figures are referenced as image files under `assets/chapters_1_2/`.
