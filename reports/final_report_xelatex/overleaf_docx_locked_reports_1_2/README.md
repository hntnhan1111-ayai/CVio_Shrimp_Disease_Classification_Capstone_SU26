# CVio AIP491 DOCX-Locked LaTeX Template

This package updates the previously working Chapter III LNCS hybrid package so the visible report layout follows the official `Template_AIP491_CP_StudentsGuide.docx`.

## Overleaf

1. Upload the entire folder/ZIP.
2. Set compiler to **XeLaTeX**.
3. Set `main.tex` as the main document.
4. Compile.

## Canonical preamble

```tex
\documentclass[a4paper]{llncs}
\usepackage{cvio-aip491-template}
```

Do not add `geometry`, `fontspec`, `setspace`, `caption`, `titlesec`, or global font-size overrides in chapter files. The canonical package already owns those settings.

## Files

- `cvio-aip491-template.sty` - canonical public package name.
- `cvio-lncs.sty` - implementation layer, kept for compatibility with earlier CVio packages.
- `llncs.cls` - retained base class from the working package.
- `chapters/03_...tex` - Chapter III content example, not a screenshot/PDF import.
- `assets/fpt_education.png` - image extracted from the official DOCX for future cover-page work.
- `reference/` - rendered DOCX reference pages used for visual QA.
- `TEMPLATE_LOCK_SPEC.md` - authoritative style contract.

## Content safety

Chapter III scientific text is not rewritten by this template update. Only typesetting/layout rules are changed.
