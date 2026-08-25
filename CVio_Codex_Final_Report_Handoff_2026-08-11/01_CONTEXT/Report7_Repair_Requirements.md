# CVio — FINAL REPORT REPAIR PROMPT v2
## Rebuild `Report7.pdf` cleanly with XeLaTeX, recover all original figures/assets, repair equations/tables/spacing, and preserve the FPT thesis format

## 0. ROLE AND GOAL

Act as a **senior thesis production engineer, XeLaTeX editor, PDF preflight specialist, and technical-typesetting reviewer**.

Your job is to create a **new, clean Final Report PDF only** from the supplied files.

### Authoritative inputs

1. `Report7.pdf`
   - This is the **authoritative current Final Report content source**.
   - Use it for:
     - report text;
     - chapter/section order;
     - table values;
     - equation semantics;
     - figure captions;
     - original embedded figures;
     - original terminology and scientific claims.
   - Do **not** edit Report1–Report6.

2. `Template_AIP491_CP_StudentsGuide.docx`
   - This is the **authoritative FPT institutional template source** for the cover and official logo.
   - The DOCX contains the original FPT Education/FPT University logo in:
     - `word/media/image1.png`
     - verified dimensions: **329 × 128 px**
     - transparent PNG/RGBA.
   - Extract and use this exact asset. Do not recreate the logo.

3. User-supplied screenshots
   - Treat them as **defect examples**, not as content sources.
   - They show what must NOT appear in the repaired report.

### Output goal

Produce a new Final Report that:

- preserves the accepted scientific content of `Report7.pdf`;
- fixes all known formatting/assembly defects;
- restores every figure that disappeared in the previous repair attempt;
- uses true semantic LaTeX for equations rather than PDF-glyph reconstruction;
- uses Times New Roman for normal text, including literal filenames/file patterns;
- has clean table alignment and page-safe layout;
- has consistent paragraph and subsection spacing;
- compiles successfully with XeLaTeX;
- passes automated and full-page visual QA.

This repair pass is **not yet the Council revision-blue pass**.

**Do not apply dark-blue revision coloring in this version. All normal content stays black.**

---

# 1. CRITICAL LESSON FROM THE FAILED REPAIR

The previous format-repair attempt introduced new defects because it reconstructed PDF content too mechanically.

Examples of unacceptable defects from that attempt:

- official FPT logo became corrupted with a black rectangle/background;
- some figures disappeared while their captions remained;
- Figure 6.21 became caption-only / blank-page content;
- formulas were reconstructed from glyph positions and produced overlapping symbols;
- `\bar{q}_{ik}` around Eq. (4.18) became visually corrupted;
- filenames and filename patterns changed to an inappropriate code/monospace font;
- normal paragraphs lost spaces or hyphens:
  - `predictionlevel`;
  - `173image`;
  - `primarytest`;
  - `MacroF1`;
  - `SDI4`;
  - `EXT3Original`;
  - `crossentropy`;
  - `trainingonly`;
  - `testpartition`;
  - `modelfamily`;
  - `classspecific`;
  - `additionalsource`;
  - `mixedsource`;
  - `outofsource`;
  - `finalmodel`;
  - `featurerecalibration`;
  - `runtorun`;
  - `errortransition`;
  - `unseenspecimen`;
  - `shrimpbody`;
  - `methoddependent`;
  - `WSSVrecall`;
  - `singledisease`;
  - `datasetspecific`;
  - broken fragments such as `decom posed`, `con figuration`, `re peated`;
- paragraphs became too tightly line-spaced;
- subsection headings had extremely large blank gaps between them;
- short sentence fragments were stranded at page bottoms;
- some short tables were split so that one final row appeared alone on the next page;
- table text was inconsistently aligned.

### Absolute prohibition

**Do NOT rebuild the report by replaying PDF text spans at their original x/y coordinates.**

**Do NOT use PDF text extraction as a final typesetting representation.**

Use PDF extraction only to recover semantic text/assets. Reconstruct the document as a real semantic LaTeX report:

- paragraphs as paragraphs;
- equations as equations;
- tables as tables;
- figures as figures;
- headings as headings.

---

# 2. FORMAT PRECEDENCE

This is an **FPT University capstone thesis**, not an IEEE conference paper.

Therefore:

- preserve A4 one-column thesis layout;
- preserve the FPT cover style;
- preserve front matter;
- preserve chapter hierarchy I–IX;
- preserve thesis-style chapter-based equation numbers such as `(4.1)`, `(4.2)`, etc.;
- do **not** convert to IEEEtran;
- do **not** convert equations to global `(1), (2), ...` numbering.

Use IEEE guidance only for:

- mathematical typography;
- equation construction;
- technical writing conventions;
- figure/table professionalism;
- citation/reference style where compatible.

FPT institutional structure takes precedence over IEEE article page layout.

---

# 3. OFFICIAL LOGO — MUST BE FIXED FIRST

## 3.1 Use the official DOCX asset

Extract:

```text
Template_AIP491_CP_StudentsGuide.docx
└── word/media/image1.png
```

The verified asset is a transparent **329 × 128 px PNG**.

Example extraction:

```python
from zipfile import ZipFile
from pathlib import Path

with ZipFile("Template_AIP491_CP_StudentsGuide.docx") as z:
    data = z.read("word/media/image1.png")

Path("figures/fpt_official_logo.png").write_bytes(data)
```

## 3.2 Preserve transparency correctly

The final logo must have:

- transparent/white background;
- no black rectangle;
- no black alpha matte;
- no JPEG conversion;
- no screenshot recreation;
- no recoloring.

XeLaTeX should normally include the PNG directly:

```latex
\includegraphics[
  width=1.54in,
  keepaspectratio
]{figures/fpt_official_logo.png}
```

The official template displays the logo at approximately **1.54 in × 0.60 in**. Use that as the starting reference and visually match the template page.

If any rendering library mishandles alpha, flatten the PNG on **pure white**, never black.

## 3.3 Cover validation

The final cover must visually match the official template structure:

- FPT logo upper left;
- `MINISTRY OF EDUCATION AND TRAINING` upper right;
- `FPT UNIVERSITY`;
- `Capstone Project Document`;
- project title;
- CVio group block;
- three current members;
- supervisor;
- project code;
- Can Tho date.

Render page 1 separately and compare it visually with `Template_AIP491_CP_StudentsGuide.docx`.

The build must fail if the logo area contains a dark/black rectangular background.

---

# 4. TYPOGRAPHY — TIMES NEW ROMAN MUST BE THE MAIN TEXT FONT

## 4.1 Main font

Use XeLaTeX + `fontspec`.

Hard requirement:

```latex
\usepackage{fontspec}
\setmainfont{Times New Roman}
```

Before building:

```bash
fc-match "Times New Roman"
```

If Times New Roman is unavailable, **do not silently replace it**. Report the missing font before final delivery unless the user explicitly approves a fallback.

Normal prose, headings, table text, captions, literal filenames, and filename patterns must use the same Times New Roman family unless a mathematically specialized font is genuinely required.

## 4.2 Math font

Use a Times-compatible mathematical font, preferably:

- STIX Two Math; or
- XITS Math.

Example:

```latex
\usepackage{unicode-math}
\setmathfont{STIX Two Math}
```

Do not fall back to obvious Computer Modern/Latin Modern math if a Times-compatible math font is available.

## 4.3 No accidental code font for filenames

The following are **literal identifiers**, not code blocks:

```text
Disease-ShrimpID-img-ImageNumber.jpg
yolo26m_asl_ldam_simam_dcfr_fp32.tflite
yolo11n-seg.pt
```

They must use the **same Times New Roman family** as surrounding prose.

Do NOT use:

```latex
\texttt{...}
\verb|...|
```

for these identifiers.

Preferred strategy:

```latex
\usepackage{xurl}
\urlstyle{same}
```

then:

```latex
\path{Disease-ShrimpID-img-ImageNumber.jpg}
```

and:

```latex
\path{yolo26m_asl_ldam_simam_dcfr_fp32.tflite}
```

This preserves literal punctuation/underscores, allows safe line breaking, and keeps the current text font.

Check the rendered PDF to confirm these strings are not monospaced.

---

# 5. BODY TEXT NORMALIZATION — REPAIR PDF-EXTRACTION DAMAGE

The semantic wording of Report7 must be preserved, but typography artifacts caused by PDF extraction/reconstruction must be repaired.

## 5.1 Known tokens that must never appear in the final PDF

Fail the build if extracted final PDF text contains any of these malformed tokens unless they are intentionally quoted:

```text
predictionlevel
173image
primarytest
MacroF1
SDI4
EXT3Original
crossentropy
trainingonly
testpartition
modelfamily
classspecific
additionalsource
mixedsource
outofsource
finalmodel
featurerecalibration
runtorun
errortransition
unseenspecimen
shrimpbody
methoddependent
WSSVrecall
singledisease
datasetspecific
```

Also detect broken mid-word spaces such as:

```text
decom posed
con figuration
re peated
anal ysis
source specific
outof source
```

where they are clearly PDF-layout artifacts.

## 5.2 Required normalized forms

Use consistent forms such as:

```text
prediction-level
173-image
primary-test
Macro-F1
SDI-4
EXT-3-Original
cross-entropy
training-only
test-partition
model-family
class-specific
additional-source
mixed-source
out-of-source
final-model
feature-recalibration
run-to-run
error-transition
unseen-specimen
shrimp-body
method-dependent
WSSV recall
single-disease
dataset-specific
```

Preserve the project label:

```text
WSSV_BG
```

exactly where that class name is intended.

Do not turn this cleanup into substantive rewriting.

---

# 6. PARAGRAPH AND HEADING SPACING

The repaired version must not reproduce the extremely tight text or giant inter-subsection gaps visible in the screenshots.

## 6.1 Body line spacing

Use a consistent readable body spacing.

Target:

- Times New Roman 12 pt for main body unless the institutional source indicates otherwise;
- line spacing approximately **1.15–1.20**, calibrated visually against the official template and clean pages of the original Report7;
- never use compressed single-spacing that causes lines to touch visually.

Example:

```latex
\usepackage{setspace}
\setstretch{1.18}
```

The exact value may be tuned slightly after rendering, but must remain visually consistent throughout.

## 6.2 Paragraph spacing

Use semantic paragraphs, not forced coordinate placement.

Example:

```latex
\setlength{\parindent}{0pt}   % if matching the existing report style
\setlength{\parskip}{0.35em}
```

or preserve the original first-line-indent policy if confirmed from the report/template.

Do not insert repeated manual:

```latex
\vspace{...}
\\[...]
\hspace{...}
```

to simulate the original PDF.

## 6.3 Heading spacing

Use `titlesec` or equivalent to define one consistent policy.

Subsections must have:

- a moderate space before the heading;
- a small space after the heading;
- no giant empty blocks.

Use:

```latex
\raggedbottom
```

to avoid vertical glue being stretched across sparse pages.

Use `needspace` to prevent a heading from being stranded:

```latex
\usepackage{needspace}
```

Require at least 3–4 body lines after a subsection heading when possible.

## 6.4 Prevent sentence fragments at page edges

Known bad example:

```text
The system
[page break]
supports preliminary screening ...
```

Do not allow this.

For specific short trailing fragments, use clean local pagination decisions:

- move the complete sentence to the next page;
- use `\Needspace`;
- use a small `\enlargethispage` only when visually safe;
- do not use large arbitrary negative spacing.

Enable strong widow/orphan protection:

```latex
\widowpenalty=10000
\clubpenalty=10000
\displaywidowpenalty=10000
```

---

# 7. HIGH-RISK PARAGRAPHS THAT MUST BE RE-TYPESET MANUALLY

Do not trust mechanical PDF text extraction for these locations.

Review them against the original `Report7.pdf` and rebuild them as normal semantic prose:

### Chapter VI

- `4.2 Classwise Performance`
- `4.3 Confusion-Matrix Analysis`
- `4.4 Model Complexity and Efficiency`
- `6. Classification Error Analysis`
- paragraph beginning:
  - `The matrix belongs only to SDI-4 + EXT-3-Original ...`
- `9. Explainability Analysis`
- paragraph beginning:
  - `Agreement among Grad-CAM++, HiResCAM, and EigenCAM ...`
- paragraph beginning:
  - `The leakage analysis is one of the strongest validity findings ...`
- `16. Threats to Validity and Limitations`

### Chapter VII

- `1.2 Classification Findings`
- `3. Limitations and Future Work`
- `3.1 Classification Limitations`
- `3.2 Segmentation Limitations`
- `3.3 Robustness and Explainability Limitations`
- `3.4 Deployment Limitations`

These paragraphs must have:

- normal word spacing;
- correct hyphenated technical terms;
- consistent line spacing;
- no manual glyph placement;
- no unexplained vertical gaps.

---

# 8. CLASSIFICATION SPLIT — ADD 70/15/15 EXPLICITLY

The primary SDI-4 classification split is:

```text
Train      804
Validation 172
Test       173
Total      1,149
```

This is the fixed seed-42 **approximately 70/15/15 train/validation/test split**.

Add this information clearly to the Classification Methodology.

Recommended sentence:

```text
The fixed seed-42 SDI-4 classification manifest follows an approximately 70/15/15 train/validation/test split, corresponding to 804 training images, 172 validation images, and 173 test images from 1,149 total images.
```

Also update the `Partitioning` row of Table 4.1 if appropriate:

```text
Fixed seed-42 70/15/15 train/validation/test manifest (804/172/173).
```

Do not change the established split membership.

Table 6.1 must remain consistent with:

```text
SDI-4:                    804 / 172 / 173 / 1,149
EXT-3-Original:           221 / 47 / 47 / 315
SDI-4 + EXT-3-Original: 1,025 / 219 / 220 / 1,464
```

---

# 9. EQUATION TYPESETTING — REBUILD ALL MATH SEMANTICALLY

This is one of the highest-priority fixes.

## 9.1 Never reconstruct math from PDF glyph coordinates

The failed repaired PDF produced overlapping symbols because equation glyphs were imported/recreated individually.

Prohibited:

- raw Unicode math copied from PDF extraction;
- positioning individual glyphs;
- rendering equation screenshots;
- reconstructing `q̄`, subscripts, superscripts, operators, etc. as independent PDF text spans.

Every equation must be written as native LaTeX source.

## 9.2 Use AMS-LaTeX

Use:

```latex
\usepackage{amsmath,amssymb,mathtools,bm}
```

Preferred environments:

- `equation`;
- `aligned`;
- `align`;
- `split`;
- `multline`;
- `cases`.

Never use:

```latex
eqnarray
$$ ... $$
```

## 9.3 Preserve thesis chapter-based numbering

The thesis uses:

```text
(4.1), (4.2), ..., (4.x)
```

Keep this convention.

Configure automatic chapter-based numbering rather than manually typing numbers:

```latex
\numberwithin{equation}{chapter}
```

or an equivalent custom counter that reproduces `(4.1)` correctly with the existing thesis chapter scheme.

Use semantic labels:

```latex
\label{eq:simam-mean}
```

and references:

```latex
\eqref{eq:simam-mean}
```

Never hard-code equation numbers in prose.

## 9.4 Mathematical typography rules

- scalar variables: italic;
- vectors/tensors: use the established manuscript convention consistently;
- textual identifiers in math: upright;
- operators/functions: upright;
- use `\times`, not the letter `x`, for dimensions;
- use `\odot` for element-wise multiplication;
- use `\max`, `\exp`, `\log`, etc.;
- use `\operatorname{...}` or `\DeclareMathOperator` for named operators;
- use `\mathrm{SimAM}`, `\mathrm{texture}`, `\mathrm{inv}` for textual subscripts;
- do not treat `mean`, `PWConv`, `DWConv`, `GAP`, or `Conv` as products of italic variables.

Recommended definitions:

```latex
\DeclareMathOperator{\meanop}{mean}
\DeclareMathOperator{\PWConv}{PWConv}
\DeclareMathOperator{\DWConv}{DWConv}
\DeclareMathOperator{\GAP}{GAP}
\DeclareMathOperator{\Conv}{Conv}
```

## 9.5 Required cleanup for Eqs. (4.1)–(4.7)

Use semantic formatting equivalent to:

```latex
For each channel, SimAM first computed the spatial mean
\begin{equation}
    \mu = \meanop_{H,W}(F),
    \label{eq:simam-mean}
\end{equation}
and the squared deviation
\begin{equation}
    D = (F-\mu)^2.
    \label{eq:simam-deviation}
\end{equation}

The spatial variance estimate was
\begin{equation}
    V =
    \frac{\sum_{H,W} D}
         {\max\!\left(1,HW-1\right)}.
    \label{eq:simam-variance}
\end{equation}

Using $\lambda = 10^{-4}$, the inverse-energy response
and attention map were defined as
\begin{equation}
\begin{aligned}
    E_{\mathrm{inv}}
        &= \frac{D}{4(V+\lambda)} + \frac{1}{2}, \\
    A_{\mathrm{SimAM}}
        &= \sigma\!\left(E_{\mathrm{inv}}\right).
\end{aligned}
\label{eq:simam-attention}
\end{equation}

This yielded
\begin{equation}
    F_{\mathrm{SimAM}}
        = F \odot A_{\mathrm{SimAM}}.
    \label{eq:simam-output}
\end{equation}

The texture branch used a channel-wise depthwise convolution
followed by pointwise projection:
\begin{equation}
    A_{\mathrm{texture}}
      =
      \sigma\!\left(
        \PWConv_{1\times1}
        \!\left(
          \DWConv_{3\times3}(F)
        \right)
      \right).
    \label{eq:texture-attention}
\end{equation}

\begin{equation}
    F_{\mathrm{texture}}
      =
      F \odot A_{\mathrm{texture}}.
    \label{eq:texture-output}
\end{equation}
```

Preserve mathematical meaning exactly.

## 9.6 Required cleanup around Eq. (4.18)

The current repaired screenshot shows overlapping/duplicated glyphs around the smoothed target.

It must render cleanly as semantic LaTeX, for example:

```latex
\begin{equation}
    \bar{q}_{ik}
      =
      (1-\varepsilon)q_{ik}
      +
      \frac{\varepsilon}{K},
    \qquad
    \varepsilon = 0.1.
    \label{eq:label-smoothing}
\end{equation}
```

Then:

```latex
\begin{equation}
    L_i
      =
      -\sum_k
      \bar{q}_{ik}
      w_{ik}
      \log p_{ik}.
    \label{eq:sample-loss}
\end{equation}

\begin{equation}
    L_{\mathrm{ASL\text{-}LDAM}}
      =
      \frac{1}{B}
      \sum_{i=1}^{B} L_i.
    \label{eq:batch-loss}
\end{equation}
```

The final PDF must contain **one and only one** visible copy of each symbol.

## 9.7 Filename-pattern / specimen-key area

The prose string:

```text
Disease-ShrimpID-img-ImageNumber.jpg
```

must be Times New Roman, not monospace.

The specimen-key equation should be semantic, e.g.:

```latex
\begin{equation}
    G_i
      =
      \mathrm{Disease}_i
      \mathbin{::}
      \mathrm{ShrimpID}_i.
    \label{eq:specimen-key}
\end{equation}
```

Then:

```latex
\begin{equation}
    G_i = G_j
    \Rightarrow
    S_i = S_j.
    \label{eq:grouped-partition}
\end{equation}
```

## 9.8 Review ALL other equations

Do not stop at (4.1)–(4.22).

Review every displayed equation in:

- Chapter IV;
- evaluation metric definitions;
- Fourier formulas;
- Appendix G.

Specific common fixes:

- `IDFT` must be an upright operator;
- textual subscripts upright;
- tilde/hat/bar attached to the intended symbol;
- no equation wider than `\textwidth`;
- no equation number collision;
- correct punctuation;
- no oversized delimiter abuse;
- no manual spacing hacks used to force fit.

---

# 10. TABLES — CENTER CONTENT AND PREVENT BAD SPLITS

## 10.1 Default alignment

Review **every table**.

Default table-cell policy:

- horizontal: centered;
- vertical: centered;
- header: centered + bold;
- wrapped text: centered inside the cell;
- numeric columns: centered or decimal-aligned where it clearly improves readability.

Define centered column types:

```latex
\usepackage{array,tabularx,longtable,xltabular,booktabs,makecell}

\newcolumntype{Y}{>{\centering\arraybackslash}X}
\newcolumntype{C}[1]{>{\centering\arraybackslash}m{#1}}

\renewcommand{\arraystretch}{1.15}
```

Use `m{}` rather than `p{}` where vertical centering is required.

## 10.2 Preserve table style

Preserve the report's professional style:

- Times New Roman;
- peach header background consistent with the original Report7;
- black borders where the existing report uses them;
- captions above tables;
- no text outside cells;
- no tiny unreadable table font;
- no table beyond page margins.

## 10.3 Short tables must remain together

Do not split a short table merely because it begins near the bottom of a page.

Specific examples:

### Table 2.1 — Team structure and roles

It contains the supervisor plus the three current team members.

Do not leave `Nguyen Van Phong` as a single continuation row on the next page.

If insufficient room remains:

- move the whole table to the next page;
- or adjust surrounding spacing slightly;
- do not split a four-row table.

### Table 4.7 and Table 6.11

These attention protocol tables contain only five configurations.

Prefer keeping each complete table together on one page.

If width is the issue:

- wrap cells;
- use landscape only if genuinely required;
- reduce redundant header wording;
- do not break one or two rows onto a second page unnecessarily.

## 10.4 Long tables

Use `longtable`/`xltabular` only for genuinely long tables such as WBS.

Repeat the header automatically on continuation pages.

Do not repeat captions incorrectly.

Do not orphan a single table row on a new page when avoidable.

---

# 11. ATTENTION CONFIGURATION EPOCH ORDER — REVERSE PRESENTATION CLEANLY

The current report uses the column concept:

```text
Configured / executed epochs
```

with values such as:

```text
100 / 60
```

The user wants the more intuitive visual order:

```text
Executed / configured epochs
```

Therefore change the header and values **together**.

Apply consistently to both the methodology and results protocol tables, including Table 4.7 and Table 6.11.

Required display:

| Configuration | Executed / configured epochs |
|---|---:|
| YOLO11n-seg reference | **60 / 100, early stop** |
| CA-to-SimAM | **100 / 100** |
| LKA-to-SimAM | **75 / 100, early stop** |
| DPCA | **200 / 200** |
| CoTEGate + BoundaryLite | **184 / 200, early stop** |

Do not simply reverse the numbers while keeping the old header.

Preserve the existing initialization information:

- `yolo11n-seg.pt`;
- `594/594 tensors`;
- `579/579 tensors`;
- framework versions.

All table cells should follow the centered alignment policy.

---

# 12. FIGURES — RECOVER ORIGINAL ASSETS BEFORE REBUILDING

This is a blocking requirement.

## 12.1 Build a complete original figure inventory

From `Report7.pdf`, enumerate every figure in the List of Figures.

The final source must have a real graphical object for every retained figure.

For each figure:

- identify the original page(s);
- extract the embedded raster/vector asset if available;
- store it under `figures/`;
- record source page/xref or crop region in a manifest.

If the figure is composed of PDF vector drawing commands and cannot be extracted as one image:

- render/crop the original figure region at at least **300 dpi** as a fallback;
- do not rasterize the entire page including caption;
- keep caption text native in LaTeX.

## 12.2 Fatal condition

A figure caption with no visible figure above/near it is a **build failure**.

The final validation must detect:

- blank figure area;
- caption-only page;
- missing `\includegraphics`;
- broken image path;
- 0-byte extracted asset;
- completely white/transparent image.

---

# 13. FIGURE 6.21 — MUST BE RESTORED AND MADE LARGER

The original `Report7.pdf` contains the retained instance-segmentation panel as an embedded JPEG.

Verified original asset:

```text
920 × 1500 px
```

It contains three vertical logical sections:

1. BG:
   - BG labels;
   - BG predictions.

2. Healthy:
   - Healthy labels;
   - Healthy predictions.

3. WSSV:
   - WSSV labels;
   - WSSV predictions.

The original PDF reused/clipped the same tall image across multiple pages.

The failed repaired PDF lost the graphical content and left captions/blank space.

### New required layout

Extract the original 920 × 1500 asset from `Report7.pdf`.

Then split/crop it into **three logical full-width panels**:

```text
Figure 6.21(a) — BG
Figure 6.21(b) — Healthy
Figure 6.21(c) — WSSV
```

Do not stretch non-uniformly.

Display each panel large enough that:

- the shrimp images are clearly visible;
- ground-truth/prediction overlays can be inspected;
- filenames/labels are materially more readable than in the original tiny rendering.

Preferred width:

```latex
width=0.97\textwidth
```

or the largest safe width allowed by the page.

It is acceptable for the three panels to occupy separate consecutive pages.

### Required subcaptions

Use the existing wording, cleanly typeset:

```text
Figure 6.21(a). BG ground-truth labels and corresponding BG predictions.

Figure 6.21(b). Healthy empty-label negatives and corresponding disease-mask predictions.

Figure 6.21(c). WSSV ground-truth labels and corresponding WSSV predictions.
```

Then preserve the overall explanation:

```text
Figure 6.21. Retained instance-segmentation ground-truth and prediction panels for BG, Healthy, and WSSV evaluation images. BG and WSSV rows compare project-authored disease-region ground-truth masks with predicted masks. Healthy rows are negative controls with no positive disease-mask target; every predicted BG or WSSV region in those rows is therefore a false positive.
```

Implementation may use:

- `subcaption`;
- `ContinuedFloat`;
- or another robust multi-page figure strategy.

But there must be:

- no blank figure page;
- no caption detached from image;
- no repeated giant image with negative y-coordinate clipping;
- no figure wider than margins.

---

# 14. OTHER FIGURE QA

Inspect all figures, especially:

- Figure 6.11 Grad-CAM++ / HiResCAM / EigenCAM;
- Figures 6.16–6.20 segmentation corruption examples;
- Figure 6.21;
- mobile screenshots in Chapter V;
- architecture diagrams in Chapter IV;
- Appendix K figures.

For every figure:

- preserve aspect ratio;
- keep image and caption semantically paired;
- do not place a caption at the bottom of an otherwise blank page;
- do not shrink the image until labels are unreadable;
- use vector original when possible;
- use 300 dpi or better for raster figures.

---

# 15. SPECIFIC CHAPTER VI SPACING REPAIRS

## 15.1 Sections 4.2–4.4

The repaired screenshot showed:

- `4.2 Classwise Performance`;
- a paragraph;
- a very large blank region;
- `4.3 Confusion-Matrix Analysis`;
- another large blank region;
- `4.4 Model Complexity and Efficiency`.

Rebuild these as normal sequential subsections.

There should be no arbitrary blank area between subsection content.

If a page has remaining space, normal text should flow into it unless a real figure/table float occupies that area.

## 15.2 Classification Error Analysis

Repair line/word spacing in the paragraph beginning:

```text
Three evidence constraints shape the error analysis...
```

Ensure terms are typeset correctly:

```text
background-removal
WSSV recall
WSSV_BG
single-disease
dataset-specific
173-image
```

## 15.3 Confusion matrix / XAI transition

Repair the text around:

```text
The matrix belongs only to SDI-4 + EXT-3-Original ...
```

and:

```text
9. Explainability Analysis
```

Use correct forms:

```text
mixed-source
Grad-CAM++
gradient-weighted
activation-gradient
class-independent
cross-method
```

## 15.4 XAI / leakage discussion

The paragraphs beginning:

```text
Agreement among Grad-CAM++, HiResCAM, and EigenCAM ...
```

and:

```text
The leakage analysis is one of the strongest validity findings ...
```

must have normal line spacing and paragraph separation.

Required forms include:

```text
shrimp-body
method-dependent
unseen-specimen
mAP
```

Do not compress these paragraphs into near-touching lines.

---

# 16. THREATS TO VALIDITY — PREVENT BAD PAGE BREAK

Review:

```text
16. Threats to Validity and Limitations
```

The previous output stranded:

```text
The system
```

at the bottom of one page, with:

```text
supports preliminary screening ...
```

on the next.

The final sentence:

```text
The system supports preliminary screening and is not a clinical or veterinary diagnostic method.
```

must remain visually coherent.

Also repair terms such as:

```text
run-to-run
test-partition
Macro-F1
cross-dataset
single-label
inter-annotator
one-factor
```

where required by the original wording.

---

# 17. CONCLUSION — REPAIR TEXT FLOW

## 17.1 `1.2 Classification Findings`

Re-typeset from the original semantic text, not from mechanically extracted line fragments.

Required technical forms include:

```text
YOLO26m-cls
cross-entropy
training-only
ASL-LDAM
SimAM-DCFR
fixed seed-42
Macro-F1
prediction-level
173-image
class-specific
additional-source
EXT-3-Original
SDI-4 + EXT-3-Original
source composition
```

Do not allow merged tokens such as:

```text
YOLO26mcls
crossentropy
trainingonly
SimAMDCFR
MacroF1
predictionlevel
173image
classspecific
additionalsource
EXT3Original
```

## 17.2 `3.1 Classification Limitations`

Repair all extraction artifacts.

Required forms include:

```text
repeated runs
test-partition
model-family
selection-independent
error-transition analysis
source-specific
out-of-source
final-model
feature-recalibration
```

Use normal paragraph spacing.

---

# 18. REMOVE `3.5 Prioritized Future Work`

Delete the entire subsection:

```text
3.5 Prioritized Future Work
```

including:

- the introductory sentence;
- all 13 numbered future-work items;
- the final paragraph beginning:
  - `The project establishes a traceable research and engineering foundation ...`

Do not leave an empty heading.

Because this removes the dedicated future-work subsection, rename the parent Conclusion section from:

```text
3. Limitations and Future Work
```

to:

```text
3. Limitations
```

unless another retained subsection still contains a dedicated future-work block that justifies the old title.

Update:

- Table of Contents;
- internal references;
- bookmarks;
- heading numbering.

Do not remove the substantive limitations in Sections 3.1–3.4.

---

# 19. PREVIOUSLY KNOWN REPORT7 DEFECTS — STILL MUST BE FIXED

In addition to the new feedback above, preserve all previously identified repair requirements.

## 19.1 Page numbering

Exactly one visible page number per numbered page.

- cover: none;
- front matter: Roman;
- main body: Arabic starting at Chapter I;
- no restarts between chapters;
- no duplicated local/global page numbers;
- no `9A`.

Never import Report1–Report6 as already paginated PDF pages.

## 19.2 Objective #4

Reconstruct Objective #4 as one continuous numbered objective.

Do not leave its continuation near/below the footer.

## 19.3 Table 1.4

Rebuild Table 1.4 natively.

The `External evaluation` row must remain inside the table.

No cell text may appear below the footer.

## 19.4 Chapter II opening

The current source begins Chapter II with a truncated phrase equivalent to:

```text
closed as an AI research project.
```

Recover the intended complete paragraph from the surrounding original context.

Do not leave a sentence beginning in the middle.

## 19.5 Table 2.5 WBS

Restore displaced WBS labels, including:

- 2.8;
- 2.9;
- 4.13;
- 4.14.

Every row must contain:

```text
WBS ID | WBS Item | Complexity | Estimated Effort
```

inside the same row.

Remove duplicated total-row wording.

## 19.6 Methodology abnormal page

Do not preserve the old `9A` inserted page.

Integrate the segmentation dataset summary normally into the Methodology hierarchy.

## 19.7 Table 4.2

Ensure the LDAM constants render completely and cleanly:

```text
m_max = 0.5
scale s = 30
```

No truncated `scale` value.

---

# 20. TABLE/FLOAT/PAGE LAYOUT RULES

Prohibited:

- absolute x/y placement for ordinary body text;
- `\resizebox{\textwidth}{!}{...}` as the default table solution;
- repeated `[H]` for every float;
- giant negative `\vspace`;
- manual page-number overlays;
- screenshots of normal tables;
- table content below footer;
- captions without figures;
- figures without captions.

Preferred:

- `tabularx`;
- `xltabular`;
- `longtable`;
- `booktabs` where appropriate;
- `placeins`;
- `needspace`;
- `pdflscape` only for genuinely wide tables;
- natural LaTeX float placement.

Use:

```latex
\usepackage{placeins}
```

and place `\FloatBarrier` only at logical section boundaries.

---

# 21. SOURCE PROJECT STRUCTURE

Use a maintainable XeLaTeX project:

```text
CVio_Final_Report_Repaired/
├── main.tex
├── preamble.tex
├── references.bib
├── chapters/
│   ├── 01_introduction.tex
│   ├── 02_management.tex
│   ├── 03_related_work.tex
│   ├── 04_methodology.tex
│   ├── 05_system_design.tex
│   ├── 06_results_discussion.tex
│   ├── 07_conclusion.tex
│   ├── 08_references.tex
│   └── 09_appendices.tex
├── figures/
│   ├── fpt_official_logo.png
│   ├── ...
│   ├── fig_6_21_bg.*
│   ├── fig_6_21_healthy.*
│   └── fig_6_21_wssv.*
├── tables/
├── scripts/
│   ├── extract_original_assets.py
│   ├── validate_source.py
│   ├── validate_pdf.py
│   ├── validate_text_tokens.py
│   └── build_and_validate.sh
├── validation/
└── final/
```

Use relative paths only.

---

# 22. PRE-COMPILE VALIDATION GATE

Before compilation, validate:

1. every `\input` exists;
2. every figure path exists;
3. every figure asset is non-zero size;
4. every referenced label exists;
5. no duplicate labels;
6. no unmatched braces;
7. no unmatched environments;
8. no raw `$$`;
9. no `eqnarray`;
10. no prohibited `\includepdf` assembly of Report1–Report6;
11. no manual page counter reset after Chapter I;
12. no U+00AD soft hyphen;
13. no zero-width characters;
14. no Unicode replacement characters;
15. no suspicious malformed merged terms listed earlier;
16. no `\texttt`/`\verb` use for the required filename/file-pattern literals.

Run:

```bash
chktex -q main.tex
```

Use ChkTeX as a linter, not the sole gate.

---

# 23. XELATEX BUILD GATE

Compile with:

```bash
latexmk -xelatex \
  -interaction=nonstopmode \
  -file-line-error \
  -halt-on-error \
  main.tex
```

Compilation must return code `0`.

Multiple passes must complete until:

- TOC stable;
- LoT stable;
- LoF stable;
- references stable;
- equation/figure/table references stable.

---

# 24. LOG GATE

Fail if the final LaTeX log contains:

```text
Undefined control sequence
Emergency stop
Fatal error
Overfull \hbox
Overfull \vbox
Float too large
Too many unprocessed floats
Citation ... undefined
Reference ... undefined
There were undefined references
multiply defined
destination with the same identifier
```

Do not deliver with unresolved serious warnings.

---

# 25. PDF PREFLIGHT GATE

Run:

```bash
qpdf --check final_report.pdf
pdfinfo final_report.pdf
pdffonts final_report.pdf
```

Require:

- valid PDF;
- A4 on every page;
- not encrypted;
- all fonts embedded;
- Times New Roman present for normal prose;
- no unexpected page-size changes.

---

# 26. AUTOMATED PDF LAYOUT GATES

Use PyMuPDF or equivalent.

## 26.1 Footer/page number

Check:

- exactly one page number in the footer;
- no duplicate numeral;
- no `9A`;
- no body text in footer zone.

## 26.2 Suspicious vertical gaps

On text-heavy pages, flag unexplained vertical gaps larger than roughly **4 normal text baselines** between consecutive semantic text blocks unless:

- a real figure/table occupies the space;
- a deliberate section break occurs.

This specifically catches the bad 4.2/4.3/4.4 layout.

## 26.3 Tight line spacing

Flag paragraph line spacings that are materially smaller than the global body baseline.

The final body should not resemble compressed OCR text.

## 26.4 Missing figure detection

For every expected figure:

- verify a source asset exists;
- verify rendered page contains non-white graphical content near the figure caption.

Specifically fail if:

```text
Figure 6.21(a)
Figure 6.21(b)
Figure 6.21(c)
```

captions exist without visible corresponding panels.

## 26.5 Page-edge sentence fragment detection

Flag extremely short isolated text lines near the bottom/top page boundary.

Manually inspect all flagged lines.

---

# 27. FULL VISUAL QA — MANDATORY

Render **every final PDF page** at 180–200 dpi.

Example:

```bash
pdftoppm -png -r 180 final_report.pdf validation/render/page
```

Create contact sheets, but also inspect high-risk pages individually.

Mandatory individual visual inspection:

1. cover/logo;
2. Table 2.1;
3. Table 2.5;
4. all Chapter IV equation pages;
5. filename-pattern/specimen-key page;
6. Table 4.7;
7. Chapter VI Sections 4.2–4.4;
8. Classification Error Analysis;
9. Figure 6.11;
10. Table 6.11;
11. Figures 6.16–6.20;
12. Figure 6.21(a);
13. Figure 6.21(b);
14. Figure 6.21(c);
15. Threats to Validity;
16. Conclusion 1.2 Classification Findings;
17. Conclusion 3.1–3.4 Limitations;
18. References;
19. Appendices.

### Fatal visual defects

Do not deliver if any page contains:

- corrupted FPT logo;
- black logo background;
- missing figure;
- blank figure slot;
- caption-only figure page;
- clipped image;
- overlapping equation glyphs;
- duplicate equation glyphs;
- equation number collision;
- monospace filename when main Times font is required;
- table text outside cell;
- table split unnecessarily;
- table cells not following required centered policy;
- text below footer;
- duplicate page numbers;
- large unexplained vertical gap;
- extremely compressed line spacing;
- broken merged words;
- isolated 1–3 word fragment at page edge;
- missing glyph/black square.

---

# 28. CONTENT BOUNDARY FOR THIS REPAIR

This pass may make only these explicit content-level changes requested by the user:

1. add the **70/15/15** classification split wording with exact counts;
2. change the segmentation epoch presentation from:
   - configured / executed
   to:
   - executed / configured;
3. remove:
   - `3.5 Prioritized Future Work`;
4. rename the parent conclusion section to `3. Limitations` if appropriate after deleting 3.5;
5. reconstruct text that is visibly truncated, displaced, merged, or corrupted by document assembly.

Do not add new Council experimental evidence in this pass.

Do not add new YOLO26m-vs-YOLO26x diagnostic results unless the user supplies and explicitly asks for them in the same task.

Do not replace accepted main metrics with unrelated reproduction values.

---

# 29. FINAL OUTPUTS

Deliver:

```text
CVio_Final_Report_Repaired_YYYY-MM-DD.pdf
```

and:

```text
CVio_Final_Report_Repaired_XeLaTeX_Source_Validation_YYYY-MM-DD.zip
```

ZIP must include:

```text
source/
figures/
tables/
scripts/
validation/
README_BUILD.md
```

Validation outputs must include at least:

```text
compile_log.txt
pdffonts.txt
pdfinfo.txt
qpdf_check.txt
footer_validation.json
layout_validation.json
text_token_validation.json
figure_inventory.csv
table_inventory.csv
equation_inventory.csv
visual_review_checklist.md
```

---

# 30. FINAL ACCEPTANCE CHECKLIST

Do not announce success until every item is true.

## Cover

- [ ] Exact official logo extracted from `Template_AIP491_CP_StudentsGuide.docx`.
- [ ] Logo has no black background.
- [ ] Logo aspect ratio preserved.
- [ ] Cover visually matches the official FPT template.

## Typography

- [ ] Main prose uses Times New Roman.
- [ ] Filenames/patterns use Times New Roman rather than monospace.
- [ ] Math uses a Times-compatible math font.
- [ ] All fonts embedded.

## Text

- [ ] No known merged malformed tokens.
- [ ] No broken mid-word spaces from PDF extraction.
- [ ] Paragraph line spacing is consistent.
- [ ] No giant subsection gaps.
- [ ] No short sentence fragment stranded at page edge.

## Classification split

- [ ] 70/15/15 explicitly stated.
- [ ] 804/172/173 exact SDI-4 counts preserved.

## Equations

- [ ] All equations are native semantic LaTeX.
- [ ] No raw PDF-glyph reconstruction.
- [ ] Eq. 4.18 is clean and non-overlapping.
- [ ] Textual operators/subscripts are upright.
- [ ] Chapter-based numbering preserved.
- [ ] No overfull equation.

## Tables

- [ ] Table text centered horizontally/vertically by default.
- [ ] Table 2.1 kept together if feasible.
- [ ] Table 4.7 kept together if feasible.
- [ ] Table 6.11 kept together if feasible.
- [ ] Epoch column changed to executed/configured.
- [ ] Epoch values displayed 60/100, 75/100, 184/200 where applicable.
- [ ] Table 1.4 repaired.
- [ ] WBS Table 2.5 repaired.
- [ ] No table exceeds margins.

## Figures

- [ ] Original figure inventory extracted from Report7.
- [ ] Every caption has a real visible figure.
- [ ] Figure 6.21 original asset recovered.
- [ ] Figure 6.21 split into large BG/Healthy/WSSV panels.
- [ ] Figure 6.21(a), (b), (c) visibly present.
- [ ] No blank/caption-only pages.
- [ ] All figures readable and aspect-ratio safe.

## Conclusion

- [ ] 1.2 Classification Findings spacing/text repaired.
- [ ] 3.1–3.4 Limitations spacing/text repaired.
- [ ] 3.5 Prioritized Future Work removed.
- [ ] TOC updated accordingly.

## Global PDF

- [ ] XeLaTeX/latexmk exit code 0.
- [ ] No serious log warnings.
- [ ] No `Overfull \hbox`/`\vbox`.
- [ ] No undefined references/citations.
- [ ] A4 throughout.
- [ ] Exactly one page number.
- [ ] No `9A`.
- [ ] No content in footer exclusion zone.
- [ ] Every page rendered.
- [ ] Every high-risk page inspected individually.
- [ ] Final PDF visually clean.

---

# 31. FIRST ACTION SEQUENCE

Perform the task in this order:

1. Inspect `Template_AIP491_CP_StudentsGuide.docx`.
2. Extract the official transparent FPT logo.
3. Inspect all 155 pages of original `Report7.pdf`.
4. Extract a complete figure inventory and original figure assets before rewriting any page.
5. Explicitly recover the original Figure 6.21 920×1500 asset.
6. Build semantic chapter text rather than PDF-coordinate text.
7. Rebuild tables.
8. Re-type all equations as semantic LaTeX.
9. Apply paragraph/heading spacing globally.
10. Apply requested content adjustments:
    - 70/15/15;
    - executed/configured epoch order;
    - remove 3.5.
11. Compile a draft.
12. Run source/log/PDF automated gates.
13. Render every page.
14. Inspect every high-risk page.
15. Fix all defects.
16. Recompile and rerun every gate.
17. Deliver PDF + source/validation ZIP only when all gates pass.

The objective is **not** merely to make a PDF that compiles.

The objective is to create a clean FPT Final Report that:
- preserves the original thesis content and figures;
- renders publication-quality mathematics;
- uses consistent Times New Roman typography;
- contains no missing figures;
- contains no broken tables;
- contains no spacing corruption;
- and can safely serve as the baseline for later Council revisions.
