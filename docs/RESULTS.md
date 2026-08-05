# Results and evidence levels

## The headline SDI-4 result is reported, but its binaries remain unresolved

The official selected seed-42 result is Accuracy `0.913295` and Macro-F1
`0.910137` on 173 test images. The corresponding official-final checkpoint was
not found, so these values are reported-result evidence rather than a
reproducible checkpoint claim.

| Metric | CE | ASL-LDAM + SimAM-DCFR | Delta |
|---|---:|---:|---:|
| Accuracy | 0.890200 | 0.913295 | +0.023095 |
| Macro-F1 | 0.890200 | 0.910137 | +0.019937 |

![Grouped bars comparing reported SDI-4 CE and proposed accuracy and macro-F1](assets/results/clean_results.png)

*Figure 1. Reported official clean results on the fixed 173-image partition.
The focused axis makes the small difference legible; the binary gap prevents
checkpoint-level reproduction.*

Source: [`paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv`](../artifacts/tables/improvements/paper_key_yolo26m_best_method_vs_stage1_ce_seed42.csv).
The Cohen's Kappa values retained in that row are 0.850500 and 0.881593, but
they were not part of the two official metrics supplied as hard evidence.

### No official-final class-wise or confusion-matrix claim

The retained class-wise table and confusion matrices describe a historical
clean result with Macro-F1 `0.905448`, not the official `0.910137` result. They
remain available for provenance, but this page does not present them as the
official-final matrix or report. The newly generated matrices under
`artifacts/release_audit/evaluation/4305e491.../` describe the rejected
`0.863062` candidate only.

## EXT-3-Original exact-hash package results

The 47-image results package contains no binary, but it references hashes that
match the two separately located checkpoint files. Architecture, class order,
stored training metadata, predictions, reports, matrices, and package hash audit
agree.

| Metric | CE (`055e22...c8ea`) | Proposed (`ce0352...fb36`) |
|---|---:|---:|
| Accuracy | 0.702128 | 0.914894 |
| Macro-F1 | 0.722990 | 0.912937 |
| Cohen's Kappa | 0.550239 | 0.869444 |

The local EXT-3 source images were unavailable for a second inference run.
Consequently, this is exact-hash/package verification, not an independent
rerun. Both outputs have exactly three classes: Healthy, BG, WSSV.

## Controlled corruptions belong to a historical SDI-4 package

The retained corruption tables are internally tied to a historical clean
Macro-F1 `0.905448` package. They are useful as a controlled sensitivity study,
but they do not establish robustness for the official-final `0.910137`
checkpoint.

![Horizontal bars comparing mean macro-F1 under five corruptions](assets/results/corruption_results.png)

*Figure 2. Mean Macro-F1 across severities 1–3. The historical method package
outperforms its CE comparator for all five selected corruptions; source-checkpoint
identity prevents transferring that claim to the official-final model.*

| Corruption | CE mean Macro-F1 | Historical method mean | Delta |
|---|---:|---:|---:|
| Impulse noise | 0.4850 | 0.5582 | +0.0732 |
| Gaussian noise | 0.7479 | 0.7850 | +0.0371 |
| Contrast reduction | 0.8207 | 0.8810 | +0.0603 |
| Defocus blur | 0.8300 | 0.8866 | +0.0567 |
| Low light | 0.8315 | 0.9022 | +0.0707 |

Sources: [`top5_noise_baseline_vs_best_mean_summary.csv`](../artifacts/tables/noise/top5_noise_baseline_vs_best_mean_summary.csv)
and [`best_method_top5_noise_by_severity.csv`](../artifacts/tables/noise/best_method_top5_noise_by_severity.csv).

## Adding EXT-3 reverses the method ordering

![Horizontal bars comparing CE and proposed macro-F1 across three dataset regimes](assets/results/regime_comparison.png)

*Figure 3. Dataset-regime comparison. The proposed method is higher for the
reported SDI-4 row and verified EXT-3 row, but CE is higher in the audited
combined regime.*

| Regime | CE Macro-F1 | Proposed Macro-F1 | Evidence |
|---|---:|---:|---|
| SDI-4 | 0.890200 | 0.910137 | reported official metrics; binaries unresolved |
| EXT-3-Original | 0.722990 | 0.912937 | exact hashes and package verified |
| SDI-4 + EXT-3-Original | 0.865105 | 0.821801 | audited additional-regime package |

The combined package's selected CE model was historically display-labeled as
ASL-LDAM + SimAM-DCFR. Audit evidence identifies its actual source method as CE;
this page uses the source method and explicitly rejects the old display label.

Source: [`verified_regime_results.csv`](../artifacts/release_audit/verified_regime_results.csv).

## Interpretation boundary

All results are single-seed, fixed-split descriptive comparisons. No confidence
interval, multi-seed mean, statistical significance, animal-level independence,
field generalization, or causal superiority is claimed.
