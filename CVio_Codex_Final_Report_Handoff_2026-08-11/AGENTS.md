# CVio Final Report - Codex Repository Instructions

## Scope
- Edit ONLY the integrated Final Report and its native XeLaTeX source.
- Do not rewrite Report1-Report6 as separate deliverables unless explicitly requested.
- Treat `Report7.pdf` as the baseline content/figure source and the FPT template as the cover/layout source.

## Non-negotiable workflow
1. Inspect source/evidence before editing.
2. Build semantic XeLaTeX source; never replay extracted PDF glyphs at x/y coordinates.
3. Compile with XeLaTeX/latexmk.
4. Run source lint, log checks, qpdf/pdfinfo/pdffonts, text-token checks, page-number checks, layout checks, and full PDF rendering.
5. Inspect all rendered pages and all high-risk pages individually.
6. If any hard gate fails, fix and rebuild; do not declare completion.
7. Only after PASS: commit source + final PDF + validation reports to a new Git branch and push normally. Never force-push.

## Format
- A4, one-column FPT thesis structure; DO NOT use IEEEtran two-column layout.
- IEEE-style editorial/math/citation discipline where compatible with the FPT thesis template.
- XeLaTeX + Times New Roman for normal text. Filenames remain in the same Times family, not monospace.
- Times-compatible math font (STIX Two Math or XITS Math preferred).
- Exactly one visible page number per numbered page.
- Front matter Roman; body Arabic continuous; no chapter resets; no `9A`.
- Tables: centered cell text by default, readable size, no margin overflow, no unnecessary row orphaning.
- Figures: recover original assets from Report7; caption without a visible figure is a fatal defect.
- Equations: native semantic AMS-LaTeX only. Never raw `$$`, `eqnarray`, PDF-glyph reconstruction, manual equation-number text, or visual spacing hacks.

## Content boundaries
- Never fabricate metrics, citations, dataset counts, hardware results, or Council evidence.
- Do not replace accepted RTX/official results with unrelated reproduction values.
- Preserve negative results and limitations.
- Classification SDI-4 fixed split: 804/172/173 = approximately 70/15/15, seed 42.
- Main official SDI-4 proposed result: Macro-F1 91.01%, Accuracy 91.33%, Kappa 0.881593.
- CE baseline: Macro-F1 89.02%, Accuracy 89.02%.
- EXT-3 proposed Macro-F1 91.29%; CE 72.30%.
- Integrated proposed Macro-F1 82.18%; CE 86.51%. Do not hide this negative result.
- Engineering preference for final lightweight classification candidate: approximately <=15M parameters. Historical broader screening ceiling may be described separately if required by retained report context.

## Git delivery
- Base repo: https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26.git
- Prefer base branch: `paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp` when available.
- New branch: `report/final-xelatex-council-repair-2026-08-11` (or a collision-safe suffix).
- Put report source under `reports/final_report_xelatex/`.
- Put final PDF under `reports/final_report_xelatex/final/`.
- Put validation evidence under `reports/final_report_xelatex/validation/`.
- Commit only after validation PASS. Push branch with `git push -u origin <branch>`.
