# CVio AIP491 LaTeX Template Lock

This package is the canonical future-report layout for CVio.

## Source of truth

Primary visual source: `Template_AIP491_CP_StudentsGuide.docx`.
The existing `CVio_Chapter_III_LNCS_Hybrid_Overleaf_FIXED` package is used as the working LaTeX base and content example.

## DOCX-derived values (locked)

- Page: A4 (210 x 297 mm).
- Margins: 1 inch top, bottom, left, right.
- Chapter title / Word Heading 1:
  - 16 pt
  - bold
  - left aligned
  - RGB/HTML `C00000`
- Section / Word Heading 2:
  - 13 pt
  - bold
  - black
  - left aligned
  - format `1. Heading`
- Subsection / Word Heading 3:
  - 12 pt
  - bold
  - black
  - left aligned
  - format `1.1 Heading`
- Normal/body source size: 11 pt.
- Normal paragraph source spacing: approximately 1.08 line spacing and 8 pt after paragraph.
- Body paragraph first-line indent: none in the Word template.
- Table header fill: `FFE8E1`.
- Table grid: 0.5 pt black rules.
- Table visible reference text: 11 pt, vertically centered.
- Table header text: bold, centered.
- Instructional template text: blue `0000FF`, italic.
- Revision marking for CVio report changes: yellow background `FFF200`.

## Font portability

The DOCX uses Microsoft theme fonts. Font binaries are **not bundled**.
The Overleaf/XeLaTeX package uses TeX-Live redistributable metric/visual substitutes:

- body and tables: `Carlito` (Calibri-compatible)
- headings: `Noto Serif` to match the supplied DOCX reference rendering

Fallbacks are included if those fonts are unavailable.

## Caption rule

The supplied DOCX contains no scientific table-caption or figure-caption example. Therefore no exact caption style can be truthfully extracted from the DOCX.
The project-approved CVio convention is locked instead:

- entire caption bold
- centered
- 9 pt
- table caption above table
- figure caption below figure
- table label style: `Table 3.1: Caption`
- figure label style: `Fig. 3.1. Caption`

This is an explicit CVio project convention, not a claim about the Word template.

## Mandatory future-report rules

1. Compile with XeLaTeX.
2. Use `\documentclass[a4paper]{llncs}` plus `\usepackage{cvio-aip491-template}`.
3. Do not redefine geometry, fonts, heading sizes, caption sizes, table header color, paragraph spacing, or table rule width inside chapter files.
4. New/revised prose uses yellow background through `\rev{...}` when a revision copy is required.
5. Do not import old PDF pages as backgrounds. Text and tables must remain native LaTeX; figures may use legitimate image assets.
6. Chapter fragments should call `\CVioNoPageNumbers`; a master report is the only document allowed to add the single final page number.
