# GitHub delivery policy

- Repository: `hntnhan1111-ayai/CVio_Shrimp_Disease_Classification_Capstone_SU26`.
- Clone/fetch first; never initialize an unrelated repository over existing project history.
- Prefer base branch `paper/asl-ldam-simam-dcfr-yolo-noisy-shrimp` when present.
- Create a fresh `report/final-xelatex-council-repair-2026-08-11` branch (suffix if collision).
- Never force-push and never rewrite unrelated history.
- Do not commit raw datasets, caches, temporary page renders, TeX auxiliary files, or huge handoff evidence archives.
- Commit final PDF, native XeLaTeX source, required final figures, reproducibility scripts, and compact validation reports.
- Before push: `git diff --check`, `git status`, build PASS, automated validation PASS, full visual review PASS.
- Use `gh auth status`; if unauthenticated, ask the user to run `gh auth login`.
