# REPORT SPEC — FPT thesis + IEEE-quality academic typesetting

## Layout authority
Institutional authority: FPT thesis/template.
Academic/typesetting guidance: IEEE where compatible.

DO NOT convert the thesis to IEEEtran or two-column layout.

## Page and font
- A4, one column.
- XeLaTeX.
- Main text: Times New Roman.
- Main body target: 12 pt unless template evidence requires a local exception.
- Body line spacing: target ~1.18, acceptable calibration 1.15–1.20.
- Use `\raggedbottom`.
- Use semantic heading spacing, not manual vertical-placement hacks.
- Normal report text is black during this formatting-only baseline.

Suggested core:
```latex
\usepackage{fontspec}
\setmainfont{Times New Roman}
\usepackage{unicode-math}
\setmathfont{STIX Two Math}
\usepackage{setspace}
\setstretch{1.18}
\raggedbottom
```

## Filenames and literal patterns
Literal filenames are identifiers, not code.
They must remain in the Times family.

Examples:
- `Disease-ShrimpID-img-ImageNumber.jpg`
- `yolo11n-seg.pt`
- `yolo26m_asl_ldam_simam_dcfr_fp32.tflite`

Use:
```latex
\usepackage{xurl}
\urlstyle{same}
```
and `\path{...}` where safe line breaking is needed.
Do not use `\texttt` or `\verb` for those identifiers.

## Equations
- Semantic AMS-LaTeX only.
- Thesis chapter-based numbering is allowed/required, e.g. `(4.1)`.
- Use `equation`, `align`, `aligned`, `split`, `multline`, `cases`.
- No `eqnarray`.
- No raw `$$`.
- No hand-painted glyphs or absolute x/y positioning.
- No manual equation-number text.
- Variables italic.
- Vectors bold.
- Functions/operators and textual acronyms upright.
- Use `\max`, `\min`, `\log`, `\operatorname`, `\DeclareMathOperator`.
- Use `\mathrm{}` / `\text{}` for textual subscripts.
- Use `\times` for dimensions and `\odot` for element-wise product when semantically correct.
- Equation punctuation follows sentence grammar.
- Eq. 4.18 must contain one clean target symbol, with no duplicate/overlapping glyph.

## Tables
Default table-cell content:
- horizontally centered;
- vertically centered;
- readable font;
- no margin overflow.

Define reusable centered `m{}` column types rather than formatting every cell ad hoc.

Prefer `booktabs`, `array`, `tabularx`, and `longtable` only when truly multi-page.
Avoid tiny/super-small text, default `\resizebox` shrinking, manual continuation prose, and one-row spill when a short table can fit as one unit.

Attention epoch display must be executed/configured:
- 60/100
- 100/100
- 75/100
- 200/200
- 184/200

## Figures
- Recover the original asset before rebuilding its caption.
- Caption without visible graphics = hard failure.
- Do not recreate photographic/research panels from memory.
- Keep figures within margins.
- Do not push a caption to a nearly empty page while the graphic disappears.
- Use `\FloatBarrier` strategically at section boundaries, not after every paragraph.
- Figure 6.21(a), 6.21(b), 6.21(c) must all be visibly present.

## Paragraphs and headings
- Do not reproduce OCR/PDF extraction damage.
- No giant whitespace between subsections.
- No compressed body leading.
- A heading should normally retain several lines of following body text.
- Use `needspace`.
- Avoid large negative `\vspace`.
- Prevent fragments such as `The system` at a page bottom.

## Page numbers
- cover: no visible page number;
- front matter: Roman numerals;
- main body: Arabic, continuous;
- exactly one visible page number;
- no `9A`;
- no duplicate local/global footer numbers.

## Classification split
State clearly:
`70/15/15 = 804/172/173`, seed 42.

## Conclusion repair baseline
Remove `3.5 Prioritized Future Work` from the formatting-repair baseline unless the user explicitly restores it later.
