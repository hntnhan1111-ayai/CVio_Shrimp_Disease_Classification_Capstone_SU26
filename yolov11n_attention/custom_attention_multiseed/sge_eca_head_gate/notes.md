# SGE-ECA Head Gate Multiseed Notes

## Why keep this direction

Single-seed result showed a pragmatic balance:

- low healthy false-positive rate;
- low disease miss rate;
- low segmentation-loss gap;
- small parameter overhead.

## What to watch

- `healthy_test_healthy_mask_fp_rate_mean`
- `labeled_test_disease_box_miss_rate_mean`
- `healthy_aware_score_mean`
- `full_test_fps_mean`
- `params_mean`

## Risk

The module may be too conservative and underperform Triplet/CESA on raw mask
mAP. Threshold sweep should check whether it offers a better deployment point
when false positives are weighted heavily.
