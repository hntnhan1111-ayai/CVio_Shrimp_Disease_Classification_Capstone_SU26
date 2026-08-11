# Final-report local override

The files in this directory are publication artifacts.

Before editing any `.tex`:
1. read `REPORT_STATE.md`;
2. read the applicable section acceptance record;
3. read REPORT_SPEC and VALIDATION_SPEC from `_handoff`;
4. inspect the corresponding baseline PDF pages.

Rules:
- semantic XeLaTeX only;
- no PDF-coordinate text reconstruction;
- no `pdfpages` body assembly;
- no `eqnarray`, raw `$$`, manual equation numbering;
- no monospace for filenames;
- no tiny tables or resize-to-fit as a default;
- recover figures before captions;
- no giant spacing or compressed line leading;
- run validators after every section integration;
- commit/push only after PASS.
