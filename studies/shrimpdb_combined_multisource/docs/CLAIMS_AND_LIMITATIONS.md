# Claims and Limitations

This document states the claims and limitations of the **shrimpdb_combined_multisource**
study. The wording below is intentionally conservative and reflects only what is supported by
the verified artifacts.

## 1. Claims

The following claims are supported by the reported fixed-seed results:

1. The ASL-LDAM loss combined with SimAM-DCFR attention was implemented and trained on the
   specified datasets using YOLO26m-cls.
2. The ShrimpDB-3 model (trained on three classes from ShrimpDB) achieved 91.49% accuracy
   and 91.29% macro-F1 on the ShrimpDB test split (47 images, seed 42).
3. The Combined-4 model (trained on four classes from both sources) achieved 83.18% accuracy
   and 82.18% macro-F1 on the Combined-4 test split (220 images, seed 42).
4. The Combined-4 model's accuracy on the ShrimpDB test domain (same 47 images, three classes)
   is lower than the ShrimpDB-only model's accuracy on the same images.

## 2. Limitations

The following limitations must be acknowledged when interpreting these results:

### 2.1 Fixed Seed Only

All reported results are from a **single fixed-seed (seed 42) run**. The study does not
report mean ± standard deviation across multiple random seeds, nor does it claim statistical
significance. Observed differences may be due to the particular random initialization and
data ordering rather than systematic effects.

### 2.2 Image-Level Split

The dataset split is performed at the **image level**. Specimen or animal identity was not
established or tracked. Images originating from the same physical specimen may appear in both
training and test partitions, potentially inflating performance estimates.

### 2.3 Exact-Duplicate Detection Scope

Exact-duplicate detection confirms the removal of byte-identical images within and across
label groups. It does not prove the removal of all near-duplicate, augmented, or transformed
variants. Residual duplicate content may be present.

### 2.4 Class Space Mismatch

The ShrimpDB-3 experiment uses a three-class space (`Healthy`, `BG`, `WSSV`), while the
Combined-4 experiment uses a four-class space (`Healthy`, `BG`, `WSSV`, `WSSV_BG`). Direct
comparison of overall metrics between experiments is confounded by the different class spaces.
The matched-domain comparison on the ShrimpDB test set (three classes) partially mitigates
this issue, but the Combined-4 model's output includes a fourth class that has no ground truth
in the ShrimpDB test set.

### 2.5 Training Duration Difference

The two experiments operated under different training durations.

- ShrimpDB-3 executed through epoch 21 (one-based indexing; epoch 1 is the first pass
  through the dataset). The validation-selected `best.pt` corresponds to epoch 6, per
  the Ultralytics EarlyStopping log message: "Best results observed at epoch 6, best
  model saved as best.pt." With `patience=15`, training stopped after 15 subsequent
  epochs without an improvement in the tracked validation metric.
- Combined-4 executed through epoch 30, completing the full 30-epoch budget.
  `patience=15` was not triggered; the validation-selected `best.pt` corresponds to
  epoch 26 (minimum `val/loss` across epochs 1-30).

Detailed epoch accounting and log evidence are recorded in
`artifacts/final_application_model/effective_epochs.json`. This duration difference
is not a methodological flaw but is relevant when comparing learning dynamics between
the two experiments.

### 2.6 Calibration

Expected calibration error (ECE) is 19.30% for ShrimpDB-3 and 33.46% for Combined-4.
These values indicate that the model's predicted probabilities are not well calibrated.
The model's confidence estimates should not be interpreted as reliable probabilities without
additional calibration.

### 2.7 Small Test Sets

The ShrimpDB-3 test set contains 47 images. The Combined-4 test set contains 220 images.
Bootstrap 95% confidence intervals (2000 samples) are wide, reflecting the limited sample
sizes. Results should be treated as estimates with substantial uncertainty.

### 2.8 Source-Domain Asymmetry

The ShrimpDB source contributes 47 images (21.4% of the Combined-4 test set), while
ShrimpDiseaseDB contributes 173 images (78.6%). Source-domain metrics for ShrimpDB are
therefore based on a small sample and carry higher variance.

### 2.9 Checkpoint Selection

The `last.pt` checkpoint for Combined-4 exhibited higher test metrics (88.18% accuracy,
87.61% macro-F1) than the validation-selected `best.pt` (83.18% accuracy, 82.18% macro-F1).
Using `last.pt` based on test-set observation is not valid for deployment. The
validation-selected `best.pt` is the designated checkpoint. Future reruns should define
checkpoint selection using validation metrics before test evaluation.

### 2.10 Research Code Only

The code, models, and results in this repository are research artifacts. The model is not a
certified veterinary diagnostic system. It has not been evaluated on clinical populations and
should not be used for diagnostic purposes.

## 3. What These Results Do NOT Prove

The results do **not** prove or imply the following:

- That ASL-LDAM + SimAM-DCFR is superior to other loss functions or attention mechanisms on
  shrimp disease classification in a generalisable sense.
- That the performance observed on ShrimpDB and ShrimpDiseaseDB will generalise to unseen
  datasets, imaging devices, lighting conditions, or shrimp species.
- That the model's predicted probabilities are calibrated or suitable for downstream
  probabilistic reasoning.
- That the image-level split is specimen-safe; animal-level generalisation has not been
  established.
- That the multi-source training protocol improves performance relative to single-source
  training in a statistically significant manner; the observed cross-domain reduction on the
  ShrimpDB test set is noted but not claimed as a systematic advantage or disadvantage.

## 4. Historical Reference Warning

Some configuration files include a reference to a prior fixed-seed result
(`reported_processed_only_reference`) from an earlier run of the reviewed repository. This
value was supplied as a historical reference and was **not** re-executed in the current
Kaggle Notebook runtime. It should not be compared directly with the current experimental
results without accounting for any differences in dataset versions, software versions, or
runtime conditions.
