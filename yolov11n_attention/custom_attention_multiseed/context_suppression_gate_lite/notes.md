# Context-Suppression Gate Lite Multiseed Notes

## Why keep this direction

Single-seed result suggested good test generalization despite modest validation
mAP. This direction is useful if the final model needs stable deployment
behavior rather than peak single-run mAP.

## What to watch

- healthy-aware score mean and std;
- test mAP mean and std;
- disease miss rate after threshold tuning;
- whether confidence sweep can reduce healthy FP without large recall loss.

## Risk

Context suppression can suppress true disease evidence when healthy and diseased
shrimp share similar visual context.
