# Migration boundary

This directory is additive. The following remain untouched:

- `yolov11n_attention/` notebooks, sources, YAML files and historical outputs.
- `LiteRT-for-Android/`.
- the vendored `yolov11n_attention/ultralytics/` tree.
- raw dataset material and the user-created new-data notebooks.

Only the baseline and five report-selected candidates appear in this new
catalog. Other exploratory modules remain available in the legacy tree but are
not promoted into the reproducibility workflow.

## Cleanup policy applied here

Only untracked, generated temporary material is removed. The raw expanded-dataset export,
its extracted train folder, Git-tracked `.tmp/`, all legacy notebooks and all
dataset folders are preserved. This avoids breaking either dataset workflow.
