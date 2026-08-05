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
- [ ] Inventory checkpoint binaries, archive members, Git/LFS objects, and textual references.
- [ ] Quarantine and inspect every candidate with trusted-load controls.
- [ ] Verify or reject the SDI-4 candidate architecture and fixed-test metrics.
- [ ] Find and verify the SDI-4 CE and both exact-hash EXT-3 checkpoints.
- [ ] Publish metadata and only verified binaries through an approved storage path.
- [ ] Rebuild the README, supporting explanations, and deterministic figures.
- [ ] Run repository, checkpoint, hash, link, and documentation QA.
- [ ] Create coherent commits and perform the non-force push gate.

## Progress

- 2026-08-05: Saved 49 pre-existing changed files to local WIP commit
  `669cef5fd0e0a915f1bac057f92e877637a1dc9a`.
- 2026-08-05: Switched to the target paper branch at `137f54f` and recorded
  `preflight.json`.

## Surprises and discoveries

- The target branch ignores fewer local preview artifacts than the preserved WIP
  branch. Those cache files remain on disk and are locally excluded from status.
- LFS is installed, but the current endpoint reports `auth=none`; this is deferred
  to the final publication gate.

## Decision log

- Use a quarantine directory outside tracked repository paths for any pickle load.
- Define SDI-4 reproduction tolerance before evaluation as absolute error
  `1e-6` for metrics recomputed from 173 integer predictions. A result outside
  this tolerance is not an exact reproduction of the official selected result.
- Do not treat historical `0.9054` evidence or filename claims as official-final
  evidence.
- Keep deployment exports conceptually separate from training checkpoints and
  do not label an export as verified until its source checkpoint is verified.

## Outcomes and retrospective

Pending.
