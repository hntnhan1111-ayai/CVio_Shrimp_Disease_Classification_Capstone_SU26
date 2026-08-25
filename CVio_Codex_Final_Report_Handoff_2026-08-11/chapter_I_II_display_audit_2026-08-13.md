# Chapter I-II Display and Consistency Audit

**Audited files**

- `CVio_Final_Report_Split_2026-08-13/CVio_Final_Report_Split_2026-08-13/01_chapter_I_project_introduction.pdf`
- `CVio_Final_Report_Split_2026-08-13/CVio_Final_Report_Split_2026-08-13/02_chapter_II_project_management_plan.pdf`
- Corresponding boundaries in `CVio_Final_Report_2026-08-13_ver2.pdf`

**Scope:** presentation, typography, pagination, section flow, tables, figures, highlights, and internal formatting statements. No content rewriting or LaTeX generation was performed.

Page references use the physical page inside each split PDF, followed by the printed footer where useful.

## Confirmed Implementation Decisions

These decisions were confirmed by the user after reviewing the numbered findings below. They override any conflicting recommendation in the original audit. The working deliverable is the **revised report for the second review, due 15 August 2026**, rather than the later clean archival version.

| Audit item | Confirmed decision | Implementation interpretation |
|---|---|---|
| A1 | Retain | Yellow highlighting is mandatory in the 15 August revised report because it identifies edited material for the second review. Do not remove it in this version. A later clean final copy may remove review markup after acceptance. |
| A2 | Accept, but defer final numbering | Chapter-level page numbering does not need to be finalized in the split files. The aggregated final report will regenerate one continuous page sequence. Avoid hard-coding page numbers in the LaTeX chapter source. |
| A3 | Fix | Reconstruct the complete Section 2.3 sequence and restore Section 2.3.2 before LaTeX generation. |
| A4 | Fix | Restore orphaned and displaced paragraphs to their correct sections. No page may start with an unexplained fragment. |
| A5 | Fix with a firm standard | Use **Times New Roman only**. Text at the same hierarchy level must use the same size, weight, color rule, and spacing. |
| A6 | Retain review marking | Treat revision-color/highlight language as relevant to the marked second-review copy. Do not use it as a permanent scientific quality criterion in the later clean archival version. |
| B1 | Governed by A1 and A5 | Preserve required revision marking, but normalize both chapter titles to Times New Roman with the same chapter-title size and style. |
| B2 | Fix | Normalize the underlying text styles consistently. Yellow highlighting remains the revision indicator for the second-review copy. Do not rely on accidental body-color differences to communicate revisions. |
| B3 | Fix | Use one Times New Roman style per heading level throughout Chapter II. |
| B4 | Fix | Use one Times New Roman caption style and size for all table, figure, and continuation captions. |
| B5 | Fix | Replace all accidental Tinos/Times-Roman span mixtures with the approved Times New Roman body/table styles. |
| C1 | Fix | Add `(continued)` captions and repeated headers to every multi-page table. |
| C2 | Fix | Rebalance Tables 2.1 and 2.3 so they do not begin when only one partial block fits. |
| C3 | No change requested | Keep the complete 90-item WBS in Chapter II. Do not move it to an appendix. It must still use correct multi-page continuation formatting. |
| C4 | Fix | Standardize table typography, borders, padding, alignment, and header styling. |
| D1 | Fix by renumbering | Renumber Figures 2.5 and 2.6 according to first appearance; do not move them solely to preserve the old numbers. Update references and the List of Figures. |
| D2 | No change requested | Keep Figure 2.4 in its current main-chapter role and scale unless a later rendered defect makes it unusable. |
| D3 | No change requested | Keep Figure 2.5 at its current density and portrait placement. |
| D4 | Fix | Remove the unexplained horizontal rule on Chapter II p. 8. |
| E1 | Fix through normal repagination | Rebalance Chapter I p. 9 during LaTeX composition; do not reduce the global font size to fill the page. |
| E2 | Fix through normal repagination | Reassess the final Chapter II page after structural repairs while preserving all content. |
| E3 | Fix | Keep figures, captions, introductions, and continuation context together where possible. |
| F1-F6 | Fix | Apply all listed terminology, symbol, nonbreaking-token, and hyphenation consistency updates. |

### Governing Style Rules for the LaTeX Phase

1. The second-review PDF retains yellow highlights around revised content.
2. Times New Roman is the only report typeface, including headings, body text, captions, tables, and footers.
3. Elements at the same hierarchy level use exactly the same font size and style.
4. Page numbering is generated only after chapters are aggregated; split-chapter numbering is not authoritative.
5. The complete WBS, Gantt chart, and current process figures remain in Chapter II.
6. Structural reading-order defects must be corrected before visual reproduction.
7. Revision markup is temporary review metadata and must not silently alter the scientific hierarchy or base typography.

## A. Submission-Blocking Defects

### A1. Revision highlighting is still visible

**Location:** Chapter I pp. 1-9 and Chapter II pp. 7, 9, 12, 14-18, 20-21, and 24-28.

Large portions of revised text remain highlighted in bright yellow, including ordinary prose, table cells, headings, captions, and conclusion text. The highlighting dominates the visual hierarchy and makes the document look like a review draft rather than a final report.

**Suggested update:** remove all text highlighting in the final version. If revision traceability is required, preserve a separate marked-up response copy and keep the submitted thesis clean.

### A2. Chapter page numbering overlaps

**Location:** Chapter I ends on printed p. 9; Chapter II begins on printed p. 9. In the full PDF these are consecutive physical pages 19 and 20.

**Suggested update:** use one continuous Arabic sequence after the front matter. Chapter II should begin on p. 10 if Chapter I ends on p. 9. Update the TOC, List of Tables, List of Figures, and all cross-references after repagination.

### A3. Chapter II contains a broken section sequence

**Location:** Chapter II pp. 18-22, printed pp. 26-30.

The expected top-level quality-management heading is missing. The visible sequence jumps from `2.2.4` to Table 2.8 and then to `2.3.1`. Section `2.3.2` is not displayed. Its apparent content is split: the page-20 paragraph ends after “may support,” page 21 begins with “protocol is controlled,” and further segmentation/mobile-quality paragraphs appear after the opening sentence of `2.4 Budget and Funding`.

**Suggested update:** reconstruct the intended order before typesetting:

1. `2.3 Quality Management`
2. Table 2.8
3. Figure 2.6 and its explanation, if retained here
4. `2.3.1 Dataset and Annotation Quality Assurance`
5. `2.3.2 Model and Experiment Quality Assurance`
6. `2.3.3 Reproducibility and Artifact Quality`
7. `2.3.4 Mobile and Document Quality`
8. `2.4 Budget and Funding`

Do not copy the current PDF reading order into LaTeX.

### A4. Orphaned and displaced text remains in Chapter II

**Locations:**

- Chapter II p. 21: `2.4 Budget and Funding` starts, but its opening sentence is interrupted by segmentation and mobile-quality paragraphs from the missing Section 2.3.2.
- Chapter II p. 23 starts with the fragment “budget decision. The effect on protocol, schedule, and thesis interpretation must be assessed through change management.”

**Suggested update:** restore these paragraphs to their source sections and require every page to begin with either a complete continuation sentence or a heading, never an unexplained fragment.

### A5. The document contradicts itself about the required font

**Location:** Chapter II p. 21 states that tables use “consistent Times New Roman typography.” Table 2.11 on p. 26 states that final auditing uses “Tinos typography.” The extracted PDF shows Tinos as the dominant font.

**Suggested update:** select one report-wide typeface based on the university template. If Tinos is retained, replace “Times New Roman” with “Tinos” in Section 2.3.4. If Times New Roman is mandatory, the whole report must be restyled rather than merely changing that sentence.

### A6. “Revised navy text” is incorrectly treated as a final-quality criterion

**Location:** Chapter II p. 26, Table 2.11, row `Final quality audit complete`.

Revision color is an editing mechanism, not a final-document quality criterion.

**Suggested update:** replace “revised navy text” with “consistent final body-text color” or remove it entirely. Final body text should normally be black.

## B. Typography and Color Inconsistencies

### B1. Chapter-title font mismatch

**Location:** Chapter I p. 1 versus Chapter II p. 1.

- Chapter I title: NimbusRoman-Bold, approximately 17.9 pt, red.
- Chapter II title: Tinos-Bold, approximately 17.9 pt, red.

**Suggested update:** use the same chapter-title style and font family for both chapters.

### B2. Body-text color changes by chapter and revision block

Chapter I is predominantly encoded as navy (`#000080`), while Chapter II is mainly black and changes to dark blue (`#1F4E79`) in later revised sections. Even when the colors look similar on screen, the underlying styles differ and will reproduce differently in print.

**Suggested update:** body, captions, lists, and table content should use one black text style. Reserve red only for chapter titles if the template requires it. Do not use navy to indicate newly revised content in the final PDF.

### B3. Heading hierarchy changes inside Chapter II

Earlier third-level headings such as `1.1.1` use approximately 11.5 pt, while `2.2.1-2.2.4` use approximately 12.4 pt. From `2.5.5` onward, headings also switch to dark blue. Section `2.6` is dark blue while earlier same-level headings are black.

**Suggested update:** define one fixed style per hierarchy level and apply it throughout:

- chapter title;
- numbered section (`1.`, `2.`);
- subsection (`1.1`, `2.4`);
- subsubsection (`1.1.1`, `2.5.1`).

### B4. Caption typography varies

Figure and table captions alternate between approximately 10.9 pt and 12 pt. The continuation caption for Table 1.4 is smaller than its initial caption.

**Suggested update:** use one caption size, weight, alignment, and spacing for all tables and figures, including continuation captions.

### B5. Accidental Times-Roman spans occur in Chapter II

**Locations:** Chapter II p. 8 contains one ordinary body line in Times-Roman; p. 10 contains several WBS entries in Times-Roman at 7.6-9.2 pt while neighboring rows use Tinos.

**Suggested update:** normalize those spans to the chosen table/body style. These are paste-format defects, not semantic emphasis.

## C. Table Presentation Problems

### C1. Multi-page tables lack continuation captions

Affected tables include:

- Table 2.1, pp. 1-2;
- Table 2.3, pp. 4-5;
- Table 2.5, pp. 9-12;
- Table 2.7, pp. 15-16;
- Table 2.8, pp. 18-19.

Column headers repeat, but the later pages do not say `(continued)`. Chapter I Table 1.4 already demonstrates the desired continuation convention.

**Suggested update:** use automatic multi-page table handling with repeated headers and `Table 2.x (continued)` on every continuation page.

### C2. Tables 2.1 and 2.3 are split after only one partial block

Table 2.1 leaves one member on the next page. Table 2.3 places only its first communication row on p. 4 and the rest on p. 5.

**Suggested update:** move the table start to the next page when insufficient space remains, or rebalance row heights and surrounding prose. Avoid starting a table when only one row fits.

### C3. Table 2.5 is too long for the main narrative

The 90-item WBS occupies four pages and uses smaller 10 pt text. It interrupts the management narrative and forces the Gantt chart into the remaining space on p. 12.

**Suggested update:** keep a summarized WBS in Chapter II and move the full 90-item ledger to an appendix, or retain it as a true multi-page long table with explicit continuation captions.

### C4. Table typography is not fully standardized

Table text varies among approximately 10.9 pt, 10.5 pt, and 10 pt. Density can justify a smaller table font, but the variation currently follows individual pasted tables rather than a documented rule.

**Suggested update:** define a default table font and one approved compact-table exception. Use consistent cell padding, vertical alignment, header height, border weight, and peach header color.

## D. Figure Presentation Problems

### D1. Figure numbering is out of order

**Location:** Figure 2.6 appears on Chapter II p. 20, while Figure 2.5 appears later on p. 23.

**Suggested update:** either move the change-management figure before the artifact-flow figure or renumber them according to first appearance. Cross-references and the List of Figures must follow the corrected order.

### D2. Figure 2.4 is not readable at normal page scale

**Location:** Chapter II p. 12.

The Gantt chart is compressed beneath the four-page WBS table. Task labels, dates, and legend items are too small for comfortable reading; embedded labels reach approximately 4.65 pt.

**Suggested update:** place the Gantt chart on its own landscape page or move a high-resolution full version to an appendix. Keep only a simplified milestone chart in the main chapter if needed.

### D3. Figure 2.5 is also dense for portrait placement

**Location:** Chapter II p. 23.

Its seven process cards remain understandable, but secondary bullet text is small.

**Suggested update:** enlarge it, simplify the internal bullet text, or use a landscape/full-width page.

### D4. Unexplained horizontal rule on Chapter II p. 8

A horizontal line appears near the top of the page without a corresponding heading, table, or figure boundary.

**Suggested update:** remove the residual border/shape unless it is part of an explicitly defined header style.

## E. Pagination and Page-Balance Problems

### E1. Chapter I p. 9 is severely underfilled

Only the final limitation paragraphs occupy the top portion of the page; most of the page is blank.

**Suggested update:** rebalance the Table 1.4 continuation and Section 6.2 text, or intentionally start Chapter II on a fresh page after a better Chapter I break. Do not solve this by shrinking all text.

### E2. Chapter II p. 28 is also underfilled

This is less serious because it is the chapter's final page, but the yellow highlights make the sparse page look unfinished.

**Suggested update:** removing revision highlights and correcting the displaced Section 2.3 content will naturally repaginate this area. Reassess afterward.

### E3. Page breaks separate captions or introductions from their objects

Figure 2.6 is introduced at the bottom of p. 19 and displayed on p. 20. Multi-page tables similarly lose their caption context on continuation pages.

**Suggested update:** keep a figure with its caption and at least one explanatory paragraph where possible. Use explicit float barriers and multi-page-table continuation logic in the later LaTeX implementation.

## F. Minor Consistency Updates

1. Standardize `Project Scope & Limitations` to `Project Scope and Limitations` if the template uses formal prose rather than ampersands.
2. Standardize `Team Work` to `Teamwork` or `Team Organization and Responsibilities`.
3. Use `×` rather than `x` in `2 × NVIDIA Tesla T4`.
4. Use one spelling/capitalization policy for `real-world`, `post-defense`, `WBS`, `Definition of Done`, and model/runtime terminology.
5. Use nonbreaking spaces or protected tokens for model names such as `YOLO26m-cls`, `WSSV_BG`, `TensorFlow Lite/LiteRT`, and `CA-to-SimAM` so they are not split awkwardly.
6. Review excessive automatic hyphenation in narrow table cells and highly technical terms.

## G. Items That Already Work Well

- A4 geometry and principal margins are consistent across both split chapters.
- Page numbers are centered consistently; only the sequence is wrong at the chapter boundary.
- Table captions are above tables and figure captions are below figures.
- Tables use a consistent peach header concept and generally clear border structure.
- Chapter I Table 1.4 demonstrates a useful continuation caption and repeated-header pattern.
- Figures 1.1, 1.2, 2.1, 2.2, 2.3, and 2.6 are visually clear at normal scale.
- Section numbering within Chapter I is coherent.

## H. Proposed Approval Checklist Before LaTeX Generation

Please approve or revise these decisions before implementation:

- [ ] Remove all yellow highlighting from the final report.
- [ ] Use continuous Arabic pagination; Chapter II follows Chapter I without repeating p. 9.
- [ ] Use one font family throughout; confirm Tinos versus Times New Roman.
- [ ] Use black body text and captions; retain red only for chapter titles if required.
- [ ] Reconstruct Chapter II Sections 2.3-2.4 before typesetting.
- [ ] Restore and title Section 2.3.2.
- [ ] Renumber/reorder Figures 2.5 and 2.6.
- [ ] Use automatic continued captions and repeated headers for multi-page tables.
- [ ] Decide whether the full WBS moves to an appendix.
- [ ] Render the Gantt chart on a landscape/full page or replace it with a simplified main-text version.
- [ ] Normalize heading and caption sizes.
- [ ] Rebalance Chapter I p. 9 after structural corrections.

## Readiness Verdict

Chapter I is structurally coherent but still visibly marked as a revision draft. Chapter II has both draft-formatting defects and a genuine content-order failure around Sections 2.3-2.4. The chapters should **not** be converted mechanically from the current PDF into LaTeX. The structure and style decisions above must be accepted first, after which LaTeX can enforce them consistently.
