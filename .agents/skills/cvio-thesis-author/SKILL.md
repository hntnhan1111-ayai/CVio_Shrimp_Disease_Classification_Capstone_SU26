---
name: cvio-thesis-author
description: Author and repair the CVio FPT Final Report in modular native XeLaTeX using the locked report spec, IEEE-quality academic typography, and section-by-section checkpoints.
---

# CVio thesis author

1. Read the repository `AGENTS.md` and report-local override.
2. Read REPORT_SPEC, IEEE_ACADEMIC_STYLE, and PROJECT_CONTEXT_LOCK.
3. Never write the entire thesis in one monolithic pass.
4. Work on one section at a time.
5. Use the shared preamble/macros for consistent typography.
6. Preserve accepted evidence; never infer missing metrics.
7. Compile a section preview, then an integrated chapter preview.
8. Hand off to `$cvio-evidence-guard` and `$cvio-pdf-preflight`.
9. Do not advance while validation fails.
10. Update REPORT_STATE and acceptance record after each pass.
