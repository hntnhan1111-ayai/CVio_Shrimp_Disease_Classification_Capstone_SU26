# CVio checkpoint release audit execution plan

This living plan governs the checkpoint audit, academic documentation redesign,
verification, and gated publication on
`paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp`.

## Compatibility boundary

The work may change documented model identity, checkpoint download contracts,
and reproduction commands. Existing runtime APIs will not be changed unless an
evidence-backed audit utility is required. No checkpoint will be assigned an
official label based on its filename alone. Dataset regime, class order,
architecture, checkpoint hash, and evaluation evidence are distinct release
boundaries.

## Milestones

- [x] Preserve pre-existing work on a local WIP branch and enter the paper branch.
- [x] Record Git, worktree, remote, LFS, and WIP provenance.
- [x] Inventory checkpoint binaries, archive members, Git/LFS objects, and textual references.
- [x] Quarantine and inspect every candidate with trusted-load controls.
- [x] Verify or reject the SDI-4 candidate architecture and fixed-test metrics.
- [x] Find and verify the SDI-4 CE and both exact-hash EXT-3 checkpoints.
- [x] Publish metadata and only verified binaries through an approved storage path.
- [x] Rebuild the README, supporting explanations, and deterministic figures.
- [x] Run repository, checkpoint, hash, link, and documentation QA.
- [x] Create coherent commits and perform the non-force push gate.

## Progress

- 2026-08-05: Saved 49 pre-existing changed files to local WIP commit
  `669cef5fd0e0a915f1bac057f92e877637a1dc9a`.
- 2026-08-05: Switched to the target paper branch at `137f54f` and recorded
  `preflight.json`.
- 2026-08-05: Inventoried 293 candidate paths and inspected 10 in-scope unique
  checkpoint/export hashes.
- 2026-08-05: Rejected `4305e491...38c1` as a historical CE architecture after
  exact 173-image evaluation with two Ultralytics runtimes.
- 2026-08-05: Verified and staged only the two exact-hash EXT-3 checkpoints
  through Git LFS; no SDI-4 weight was published.
- 2026-08-05: Rebuilt the academic README, method documentation, figures,
  checkpoint registry, model card, and evidence-scoped result narrative.
- 2026-08-05: Completed formatting, lint, tests, smoke, import, checkpoint,
  figure, link, hash, LFS, and whitespace QA.
- 2026-08-05: Passed the final upstream-divergence and credential gates, uploaded
  both verified LFS objects (43 MB), and pushed the paper branch without force.

## Surprises and discoveries

- The target branch ignores fewer local preview artifacts than the preserved WIP
  branch. Those cache files remain on disk and are locally excluded from status.
- LFS is installed, but the current endpoint reports `auth=none`; this is deferred
  to the final publication gate.
- The proposed-named SDI-4 candidate contains no late attention path and scores
  below both official targets; runtime pinning does not change its predictions.
- Neither exact official-final SDI-4 binary exists in the audited sources.
- The combined-regime selected artifact was historically mislabeled as proposed,
  but its retained source metadata and architecture identify CE.

## Decision log

- Use a quarantine directory outside tracked repository paths for any pickle load.
- Define SDI-4 reproduction tolerance before evaluation as absolute error
  `1e-6` for metrics recomputed from 173 integer predictions. A result outside
  this tolerance is not an exact reproduction of the official selected result.
- Do not treat historical `0.9054` evidence or filename claims as official-final
  evidence.
- Keep deployment exports conceptually separate from training checkpoints and
  do not label an export as verified until its source checkpoint is verified.
- Publish the two verified EXT-3 binaries via Git LFS because the repository has
  LFS support; retain unresolved SDI-4 slots as metadata-only entries.
- Treat corruption and XAI evidence tied to the historical 0.905448 package as
  qualitative/controlled secondary evidence, never as official-final evidence.

## Outcomes and retrospective

The scientific release gate worked as intended: a convincing filename was
rejected by architecture and metric evidence, while exact-hash EXT-3 binaries
were separated from their results ZIP and safely published. The local audit,
documentation, artifacts, tests, LFS upload, and non-force branch push are
complete.
