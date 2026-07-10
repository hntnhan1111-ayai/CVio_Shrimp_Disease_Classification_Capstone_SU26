# Triplet Attention Segment Head Multiseed Notes

## Why keep this direction

Single-seed result was the strongest overall candidate for instance
segmentation quality:

- highest labeled diseased test mask mAP50 among the custom modules;
- highest full-test mask mAP50;
- highest healthy-aware score.

## What to watch

- `healthy_aware_score_mean` and `labeled_test_mask_map50_mean`
- `healthy_aware_score_std` and `labeled_test_mask_map50_std`
- `healthy_test_healthy_mask_fp_rate_mean`
- `labeled_test_disease_box_miss_rate_mean`

## Risk

The single-seed run had a large segmentation-loss gap, so this direction should
not be claimed as final unless the three-seed standard deviation is acceptable.
