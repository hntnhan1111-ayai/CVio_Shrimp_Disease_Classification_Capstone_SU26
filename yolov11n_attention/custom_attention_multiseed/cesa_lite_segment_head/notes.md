# CESA-Lite Segment Head Multiseed Notes

## Why keep this direction

Single-seed result was the second strongest mask-quality candidate and directly
targets the feature maps consumed by the segmentation head.

## What to watch

- mean labeled diseased test mask mAP50;
- full-test mask mAP50;
- disease miss rate;
- healthy-aware score;
- stability versus Triplet.

## Risk

CESA-style attention can amplify disease-like texture. The threshold sweep
should verify whether false positives can be controlled without losing too much
mask recall.
