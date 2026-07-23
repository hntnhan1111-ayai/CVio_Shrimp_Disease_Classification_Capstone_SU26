# Limitations

1. The frozen winner is reported for one seed; multi-seed confirmation is required for a strong statistical claim.
2. Architecture variants were explored before freezing the winner; a new untouched holdout or external dataset is preferred for final journal claims.
3. Corruption severity levels are ordered stress tests, not independent random replicates.
4. Clean recall is slightly lower than the matched baseline.
5. Some selected corruptions have negative mean mAP75 differences despite positive mAP50-95 differences.
6. Qualitative samples use one shared WSSV image and are controlled illustrations, not per-image maxima.
7. Standalone validation/test plot directories were absent from the collected evidence; numeric metrics and regeneration scripts are present.
8. Mobile latency, energy, and thermal behavior are not established by the RTX 4090 evaluation.
