# CVio Codex Final Report Handoff

Purpose: hand this folder/ZIP to a local Codex agent so it can rebuild the CVio Final Report as a native XeLaTeX project, compile it locally, run mandatory PDF validation gates, and push the final PDF + XeLaTeX source to a new GitHub branch.

## Recommended workflow

1. Extract this ZIP to a stable path such as `D:\CVio\CVio_Codex_Final_Report_Handoff_2026-08-11`.
2. Open PowerShell.
3. Run `05_SCRIPTS\00_INSTALL_CODEX_AND_LATEX.ps1`.
4. Restart PowerShell/VS Code/Codex after installs.
5. Run `05_SCRIPTS\01_INSTALL_CVIO_SKILLS.ps1`.
6. Verify authentication:
   - `codex` and sign in with ChatGPT if prompted.
   - `gh auth status`; if needed run `gh auth login`.
7. Run `05_SCRIPTS\02_PREP_REPO_BRANCH.ps1` or let Codex perform the equivalent Git steps.
8. Copy this bundle's `AGENTS.md` to the cloned repository root.
9. Start Codex from the repository root and paste `01_MASTER_PROMPT_CODEX_REPORT.md`.
10. Explicitly invoke `$cvio-academic-report` and `$cvio-pdf-preflight` if Codex does not activate them automatically.
11. Codex must not push until every validation gate passes.

## Key principle

The deliverable is not merely a PDF that compiles. It is a native XeLaTeX report whose text, equations, tables, figures, citations, pagination, typography, and rendered PDF all pass structural and visual QA.

## Baseline precedence

1. User-approved facts/evidence in this handoff.
2. Council Review 3 requirements.
3. Current `Report7.pdf` for existing report content and figure recovery.
4. FPT template DOCX for institutional cover/logo/layout.
5. Older standalone reports only as historical context, never as the editing target.
