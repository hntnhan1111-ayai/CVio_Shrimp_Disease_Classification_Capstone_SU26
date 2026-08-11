# Subagent policy

Use subagents to reduce context pollution and improve review quality, but keep writes serialized.

## Main-agent role
The main agent is the only LaTeX writer/integrator.

## Review subagents
After a section draft compiles, spawn read-only reviewers:

### Reviewer A — evidence and academic claims
Check metrics/provenance, dataset/split wording, fabrication risk, causal overclaim, and citation support.

### Reviewer B — XeLaTeX and typography
Check semantic LaTeX, equation notation, table construction, float behavior, font/spacing rules, references/labels.

### Reviewer C — PDF/render/preflight
Check validator outputs, log failures, missing figures, page numbering, whitespace/leading, high-risk rendered pages.

Wait for all reviewers.
Collect findings with exact file/line/page references.
The main agent fixes them.
Rerun validation.

Parallel read-heavy review is encouraged.
Parallel write-heavy editing of the same report source is prohibited.
