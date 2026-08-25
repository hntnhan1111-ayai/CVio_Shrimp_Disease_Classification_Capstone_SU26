# CVio Chapters I-II - Overleaf project

This upload package uses the supplied CVio LNCS hybrid template.

## Compile on Overleaf

1. Upload `CVio_Chapters_I_II_LNCS_Overleaf.zip` as a new project.
2. Set the compiler to **XeLaTeX**.
3. Compile `main.tex`.

## Structure

- `main.tex`: project entry point.
- `cvio-lncs.sty`: shared CVio/LNCS formatting rules from the supplied template.
- `llncs.cls`: supplied Springer LNCS class.
- `chapters/`: semantic Chapter I and Chapter II sources.
- `assets/chapters_1_2/`: extracted source figures referenced by relative image paths.

Yellow revision marking uses `\rev{...}` and is tight to the text. No revision row or cell uses a yellow background fill. Peach table headers remain normal table styling rather than revision marking.

The template hides page numbers for standalone chapter review. Remove `\CVioNoPageNumbers` only when these chapters are integrated into a full-report pagination workflow.
