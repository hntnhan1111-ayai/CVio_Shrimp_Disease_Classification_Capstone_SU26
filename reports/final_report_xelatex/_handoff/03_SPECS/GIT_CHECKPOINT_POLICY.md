# Git checkpoint policy

## Isolation
Never edit the audited dirty checkout directly.
Create a clean Git worktree from:
`origin/paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp`

Recommended branch:
`report/final-xelatex-e2e-2026-08-11`
with a collision-safe suffix when needed.

## Commit cadence
During Chapter VI:
- one successful source + validation commit after EACH accepted section;
- push immediately after that commit.

At chapter boundaries:
- generate a chapter preview PDF;
- commit source + chapter validation + chapter preview;
- push immediately.

At final thesis completion:
- commit full source;
- final PDF;
- validation reports;
- build README;
- manifest/hash;
- push.

## Do not bloat history
Do not commit a complete 155-page PDF after every small section.
The commit itself is the checkpoint.
Use small section/chapter previews during intermediate stages.
Commit the complete final thesis PDF only at the final accepted stage.

## Commit naming examples
- `report(ch6-s04): repair main classification section`
- `report(ch6-s09): validate explainability section`
- `report(ch6): complete strict validated results chapter`
- `report(ch4): repair semantic equations and methodology layout`
- `report(final): pass full thesis XeLaTeX preflight`

## Gate before commit
A checkpoint script must refuse to commit if validation is not PASS.

Never force-push.
Never discard unrelated history.
