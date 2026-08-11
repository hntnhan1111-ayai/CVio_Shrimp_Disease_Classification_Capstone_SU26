---
name: cvio-pdf-preflight
description: Build, lint, render, and strictly validate CVio XeLaTeX report stages; refuse checkpoint/final delivery until all machine and visual gates pass.
---

# PDF preflight

Use the report-local scripts and VALIDATION_SPEC.

For each stage:
1. run source lint;
2. compile from a clean build directory;
3. scan the LaTeX log;
4. run qpdf;
5. run pdffonts and pdfinfo;
6. run the Python PDF validator;
7. render every page for the current stage;
8. inspect high-risk pages;
9. return exact failures with file/page references.

No PASS if any hard validator fails, a required figure/table is missing, malformed text remains, or a required visual review was not actually performed.
