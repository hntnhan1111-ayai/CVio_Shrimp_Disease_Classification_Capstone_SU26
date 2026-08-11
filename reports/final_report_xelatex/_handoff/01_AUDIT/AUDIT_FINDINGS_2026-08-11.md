# Local audit findings — authoritative for setup

Audit input: `_CVio_Report_Audit_20260811_215100.zip`.

## Confirmed working
- Git 2.53.0.
- Codex CLI 0.147.0.
- GitHub CLI 2.70.0 executable found.
- MiKTeX XeLaTeX 25.12 / XeTeX 4.16.
- qpdf 12.3.2.
- Times New Roman regular/bold/italic/bold-italic files exist in `C:\Windows\Fonts`.
- Required LaTeX packages checked by `kpsewhich` are present.
- XeLaTeX smoke test generated a PDF and the basic hard-warning scan passed.
- qpdf syntax check of the current 155-page PDF passed.

## Missing or not yet confirmed
- Perl was missing.
- Ghostscript was missing.
- `latexmk` did not have a usable version result; MiKTeX latexmk previously required Perl.
- Poppler commands `pdfinfo`, `pdffonts`, and `pdftoppm` did not produce usable version results.
- GitHub CLI was NOT authenticated. Human must run `gh auth login`.
- Python 3.11 was found, but the package probe inside the audit script had a PowerShell quoting bug, so package availability must be rechecked using the corrected strict audit script.

## Critical repository state
Existing checkout:
`D:\CVio_RECSRA_Shrimp_Disease_Detection`

Remote:
`https://github.com/hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26.git`

The audited checkout had an unrelated modified file:
`docs/GIT_RELEASE_RECORD.md`

Therefore the report agent MUST use a new Git worktree and MUST NOT clean/reset the existing checkout.

## Critical current-PDF observation
The current PDF is structurally readable by qpdf, but structural validity is not equivalent to visual/typographic correctness.

Automated audit still found:
- page 102: trailing `The system`;
- pages 108–109: `173image`;
- page 109: `singledisease`.

User screenshots additionally show merged words, broken references, giant subsection gaps, compressed paragraph leading, table header overlap, and caption/float placement problems.

Conclusion: qpdf-only validation is insufficient. Strict semantic-text + render/layout validation is mandatory.
