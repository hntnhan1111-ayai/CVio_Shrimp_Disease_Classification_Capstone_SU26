---
name: cvio-pdf-preflight
description: Validate the CVio XeLaTeX Final Report PDF before delivery or Git push. Use for compile logs, qpdf/pdfinfo/pdffonts checks, page-number validation, text-token corruption checks, figure-presence checks, overflow/footer checks, full-page rendering, and visual regression. Any hard failure requires fixing and rerunning the gates.
---

# CVio PDF Preflight

Mandatory loop:
1. run source checks;
2. compile with `latexmk -xelatex` until references stabilize;
3. fail on serious LaTeX/log errors and all overfull hbox/vbox warnings;
4. run qpdf/pdfinfo/pdffonts;
5. run PyMuPDF validation scripts;
6. render every page at 180-200 dpi;
7. inspect contact sheets and high-risk pages individually;
8. fix all defects and repeat from step 1.

Hard failures include: duplicate page numbers, `9A`, content below footer, missing/blank figures, caption-only figure page, table overflow, unreadable table text, font non-embedding, broken glyphs, overlapping equation symbols, malformed merged words, unresolved citations/references, or output not A4.

Do not accept “compiled successfully” as sufficient evidence of correctness.
