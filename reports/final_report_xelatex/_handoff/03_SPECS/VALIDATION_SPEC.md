# STRICT VALIDATION SPEC

The validator is part of the deliverable, not an optional helper.

## PASS definition
A stage may be committed only when:
- source lint: PASS;
- compilation: PASS;
- compile log: PASS;
- qpdf: PASS;
- font embedding: PASS;
- text corruption scan: PASS;
- figure/table manifest: PASS;
- footer/page-number scan: PASS;
- layout heuristic scan: PASS;
- render generation: PASS;
- high-risk visual review: PASS;
- no unresolved reviewer finding.

If any item is FAIL, the agent must repair and rerun the complete relevant gate.

## Environment gate
Required:
- xelatex;
- qpdf;
- pdfinfo;
- pdffonts;
- pdftoppm;
- Git;
- authenticated gh;
- Python 3.11+;
- pymupdf;
- Pillow;
- pypdf;
- PyYAML;
- Times New Roman files;
- required LaTeX packages.

latexmk is preferred but not mandatory because the supplied build script has a XeLaTeX fallback.
Perl is only mandatory if latexmk is used.

## Source hard failures
- raw `$$`;
- `eqnarray`;
- `pdfpages` used to assemble report body;
- duplicate labels;
- missing includegraphics asset;
- unsafe soft hyphen / zero-width characters;
- known malformed tokens;
- monospace around known filenames;
- tiny/scriptsize table workaround;
- large negative vspace;
- manual hard-coded equation numbers;
- unresolved visible placeholders.

## Compile-log hard failures
Undefined control sequence, Emergency stop, Fatal error, LaTeX Error, Overfull hbox/vbox, Float too large, Too many unprocessed floats, undefined references/citations, multiply-defined labels/destinations, Missing character.

## PDF hard failures
qpdf error, encryption, non-A4 page, missing embedded fonts, no Times-family text font, malformed text tokens, soft-hyphen/zero-width/replacement-character corruption, footer junk, body sentence in footer zone, blank non-intentional page, required figure/table ID missing, high-risk figure caption without nearby visual content, caption-only figure page, extreme prose-only whitespace, materially compressed body line-height, broken numeric citation pattern, bad FPT logo rendering, or missing Figure 6.21(a)/(b)/(c).

## Visual gate
Render every page and create contact sheets.
Inspect high-risk pages individually.

The agent must not claim visual PASS if it did not actually inspect the renders with an available image/viewer capability.
If local Codex cannot visually inspect image files, stop before final delivery and request human visual approval; do not fake the gate.
