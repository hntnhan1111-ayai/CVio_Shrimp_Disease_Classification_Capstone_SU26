# Repository Baseline Audit

Audit date: 2026-07-29

Repository: `hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26`

Branch: `main`

HEAD: `969544bd80658c90d093ba4c2b99b9cd798aeb86`

## Baseline

- The repository root contained two tracked files: `README.md` and `.gitignore`.
- The README contained only `# CVio_Shrimp_Disease_Classification_Capstone_SU26`.
- The existing `.gitignore` contained Flutter, Dart, Android, iOS, desktop, coverage, and editor rules. Those rules are preserved.
- No source code, tests, datasets, checkpoints, model weights, experiment configs, metrics, documentation assets, workflows, or community files were present on `main`.
- No internal or external README links existed, so no broken links were detected.
- No license was declared.
- No CI configuration was present; current CI status was therefore not applicable.
- Repository visibility was not inferred from the checkout. Documentation must not depend on visibility.
- `git ls-files` exposed no likely secret file names and no large tracked files.
- `git status --ignored --short` exposed no ignored files in the checkout.
- Gitleaks was not installed, so no content-level secret scan was performed. No secret values were printed during this audit.

## Evidence available outside `main`

Remote branches were inspected read-only to understand project scope. They contain classification, segmentation, mobile/LiteRT, experiment, and paper-oriented work. Because those branches are not merged into `main`, their measurements and implementation details are not presented as verified `main` results.

Git authors on remote branches identify Nhan, Phong, and Nhu. No reliable repository evidence identified the fourth student member. The public landing page therefore keeps that person as `TBD`.

## Files protected from modification

- `.git/` and all Git history.
- Existing `.gitignore` rules.
- All remote branch contents.
- Any research artifacts, datasets, checkpoints, or measured values not present on `main`.

## Missing information

- Institution, course code, capstone dates, and formal supervisor identity.
- Fourth student name.
- Dataset cards, licenses, access terms, counts, splits, and hashes accepted for the integrated capstone.
- Released baseline, proposed-method, and final-checkpoint definitions.
- Verified integrated experiment results and metric provenance.
- Training, evaluation, export, and benchmark entry points on `main`.
- Source-code, model-weight, dataset, and third-party asset licenses.
- Supported Python/framework versions and deployment targets.

## Baseline conclusion

`main` is an empty landing surface rather than a reproducible research release. The repository can safely receive documentation, original visual assets, governance templates, and local validation tooling, provided every scientific value remains explicitly pending until evidence is merged and audited.
