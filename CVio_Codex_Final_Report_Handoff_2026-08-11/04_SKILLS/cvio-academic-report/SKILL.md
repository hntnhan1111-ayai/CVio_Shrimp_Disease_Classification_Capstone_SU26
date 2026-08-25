---
name: cvio-academic-report
description: Rebuild, edit, or finalize the CVio FPT capstone Final Report as a native XeLaTeX thesis using retained evidence, IEEE-quality academic/math/citation practices, original figures, and strict no-fabrication boundaries. Use for CVio report writing, LaTeX fixes, tables, equations, figures, citations, conclusions, and evidence integration. Do not use to rewrite standalone Report1-Report6 unless explicitly asked.
---

# CVio Academic Report

1. Read repository `AGENTS.md` and the CVio handoff context before editing.
2. Treat FPT institutional layout as authoritative; IEEE is editorial/math/citation guidance only, not a two-column page class.
3. Preserve scientific meaning and accepted metrics. Never invent missing experiments, citations, hardware results, or causal explanations.
4. Use native semantic XeLaTeX. Do not replay PDF glyph coordinates and do not assemble already-paginated body PDFs.
5. Recover original figures from Report7 before authoring replacement pages. Caption without visible image = failure.
6. Equations use AMS-LaTeX and automatic chapter numbering; textual operators/subscripts upright; no `eqnarray`, raw `$$`, or manual number tags.
7. Tables use readable Times text, centered cells by default, wrapping rather than tiny font, and correct bolding only for genuinely best values under the stated metric direction.
8. Preserve negative findings and limitations. Use calibrated academic wording.
9. Run `$cvio-pdf-preflight` before calling the report complete.
10. Only validated outputs may be committed/pushed.
