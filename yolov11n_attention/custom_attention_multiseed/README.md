# Top-4 Custom Attention Multiseed Pipeline

This folder contains standalone per-direction notebooks plus optional
orchestration scripts for rerunning only the four retained YOLO11n-seg custom
attention modules across three seeds, evaluating confidence and NMS thresholds,
and aggregating mean +/- std results.

The pipeline is intentionally narrow:

- It does not create new attention modules.
- It does not modify the clean baseline notebook.
- It does not rebuild or reshuffle the grouped/no-leakage split.
- It does not change class names or dataset contents.
- It skips existing run directories so old checkpoints are not overwritten.

## Source Protocol

The training and evaluation protocol was checked against:

- `yolov11n_attention/yolov11n_grouped_attention/aip491-01-yolo-seg-11n-clean-baseline-fix-leakage.ipynb`
- `custom_attention/res/triplet-attention-segment-head.ipynb`
- `custom_attention/res/cesa-lite-segment-head.ipynb`
- `custom_attention/res/context-suppression-gate-lite.ipynb`
- `custom_attention/res/sge-eca-head-gate.ipynb`

The baseline seed is `42`, so the three configured seeds are:

```python
SEEDS = [42, 3407, 2026]
```

The healthy-aware score follows the existing notebook formula:

```text
score = labeled_test_mask_map50
        - 0.05 * mask_count_mae
        - 0.15 * disease_box_miss_rate
        - 0.10 * healthy_mask_fp_rate
```

The same weights are used for every module, seed, confidence threshold, and NMS
threshold.

## Directory Layout

```text
custom_attention_multiseed/
|-- README.md
|-- run_top4_multiseed.py
|-- evaluate_top4_threshold_sweep.py
|-- aggregate_multiseed_results.py
|-- configs/
|   |-- top4_modules.yaml
|   `-- seeds.yaml
|-- results/
|   `-- .gitkeep
|-- notebooks/
|   `-- top4_multiseed_summary.ipynb
|-- triplet_attention_segment_head/
|   |-- README.md
|   |-- notes.md
|   `-- triplet_attention_segment_head_multiseed.ipynb
|-- cesa_lite_segment_head/
|   |-- README.md
|   |-- notes.md
|   `-- cesa_lite_segment_head_multiseed.ipynb
|-- context_suppression_gate_lite/
|   |-- README.md
|   |-- notes.md
|   `-- context_suppression_gate_lite_multiseed.ipynb
`-- sge_eca_head_gate/
    |-- README.md
    |-- notes.md
    `-- sge_eca_head_gate_multiseed.ipynb
```

The four module folders mirror the organization used by `custom_attention/`.
Each folder contains one standalone train/eval notebook copied from the
matching source module and patched for the three seeds. These notebooks do not
call the shared Python runner scripts and do not define new attention modules.

## Top-4 Module Map

| Key | Display name | Model YAML |
|---|---|---|
| `triplet_attention_segment_head` | Triplet Attention Segment Head | `custom_attention/triplet_attention_segment_head/yolov11n_triplet_segment.yaml` |
| `cesa_lite_segment_head` | CESA-Lite Segment Head | `custom_attention/cesa_lite_segment_head/yolov11n_cesa_lite.yaml` |
| `context_suppression_gate_lite` | Context-Suppression Gate Lite | `custom_attention/context_suppression_gate_lite/model.yaml` |
| `sge_eca_head_gate` | SGE-ECA Head Gate | `custom_attention/sge_eca_head_gate/model.yaml` |

## Expected Dataset State

The scripts expect an already prepared grouped/no-leakage dataset split with:

```text
train/images, train/labels
valid/images, valid/labels
test/images, test/labels
```

By default the config points to the Kaggle baseline location:

```text
/kaggle/working/shrimpDisHandSegV2-1
```

If your local path is different, pass both:

```bash
--base-data-dir <prepared_dataset_dir> --data-yaml <prepared_dataset_dir>/data.yaml
```

The CLI training script copies the prepared split into a per-module/per-seed
snapshot under `custom_attention_multiseed/results/datasets/`. The standalone
notebooks follow the original notebook style and copy the prepared split under:

```text
/kaggle/working/custom_attention_multiseed_<module>/<module>/seed_<seed>/dataset
```

Both paths keep training/evaluation code from mutating the source dataset.

## Run Order

For the notebook-first workflow, open exactly one notebook for the direction
you want to rerun and execute its cells:

```text
custom_attention_multiseed/triplet_attention_segment_head/triplet_attention_segment_head_multiseed.ipynb
custom_attention_multiseed/cesa_lite_segment_head/cesa_lite_segment_head_multiseed.ipynb
custom_attention_multiseed/context_suppression_gate_lite/context_suppression_gate_lite_multiseed.ipynb
custom_attention_multiseed/sge_eca_head_gate/sge_eca_head_gate_multiseed.ipynb
```

Each of these notebooks runs only its own module across:

```python
SEEDS = [42, 3407, 2026]
```

After notebook training creates the checkpoints, run threshold sweep and
aggregation from the repository root:

```bash
python custom_attention_multiseed/evaluate_top4_threshold_sweep.py
python custom_attention_multiseed/aggregate_multiseed_results.py
```

The CLI training runner is still available when you want batch execution
instead of notebooks:

```bash
python custom_attention_multiseed/run_top4_multiseed.py --dry-run
python custom_attention_multiseed/run_top4_multiseed.py
python custom_attention_multiseed/run_top4_multiseed.py --module sge_eca_head_gate --seed 42 --smoke
```

## Training Outputs

Training outputs are written to:

```text
runs/custom_attention_multiseed/<module_name>/seed_<seed>/
```

The training run log CSV is:

```text
custom_attention_multiseed/results/top4_multiseed_train_runs.csv
```

If a run directory already exists, the script skips it. If `weights/best.pt`
already exists, the run is recorded as `skipped_existing`.

## Threshold Sweep

The evaluation script sweeps:

```python
CONF_LIST = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
IOU_LIST = [0.50, 0.60, 0.70]
```

For each module/seed/conf/iou it evaluates:

- full validation set;
- full test set, including healthy images;
- labeled-only diseased test set;
- healthy-only test images for false-positive counting.

The main sweep CSV is:

```text
custom_attention_multiseed/results/top4_threshold_sweep.csv
```

Missing checkpoints and evaluation errors are logged to:

```text
custom_attention_multiseed/results/top4_threshold_sweep_errors.csv
```

## Aggregation Outputs

The aggregation script writes:

```text
custom_attention_multiseed/results/top4_multiseed_summary.csv
custom_attention_multiseed/results/top4_threshold_best_by_module.csv
custom_attention_multiseed/results/top4_final_ranking.md
```

`top4_final_ranking.md` reports:

- best performance model;
- best stable model;
- best deployment candidate.

It does not claim a final winner unless all top-4 modules have all three seeds
at the default threshold.

## Notebook

Open:

```text
custom_attention_multiseed/notebooks/top4_multiseed_summary.ipynb
```

The notebook reads the aggregate CSV files, displays mean +/- std tables, plots
test mAP50, mask mAP50-95, healthy FP, disease miss, healthy-aware score, and
shows the best threshold per module.
