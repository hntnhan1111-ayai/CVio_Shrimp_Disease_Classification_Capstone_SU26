# Items still required before a full rerun

1. **Legacy dataset inputs:** the original `data.yaml`, fixed split manifest,
   preprocessing details and dataset location for the 129-image test protocol.
2. **Expanded-dataset final split:** one versioned grouped near-duplicate split manifest,
   including how ambiguous/cross-label duplicate groups were resolved.
3. **Runner parity:** extract each selected custom layer from its notebook into
   a Python module and prove a matching forward pass/model YAML build.
4. **WIoU v3 on the expanded dataset:** no WIoU-v3 notebook exists in either current expanded-dataset
   notebook track; it is the only top-five candidate not yet ported there.
5. **Matched expanded-dataset outputs:** the current output does not contain a baseline
   strong or LKA→SimAM training result, so neither can be used in the current
   expanded-dataset comparison table.
6. **Held-out evaluation:** add the final test manifest, threshold and
   healthy-FP/disease-only metrics for the selected expanded-dataset checkpoints. The
   imported results.csv values are validation metrics.
7. **Multi-seed evidence:** all report-selected values are single-run results;
   run seeds 42, 43 and 44 (or another pre-registered set) for the baseline and
   each candidate.
8. **Config runtime dependency:** install PyYAML from requirements.txt in the
   training environment before a YAML-driven runner is added; the current
   Python environment does not include it.
