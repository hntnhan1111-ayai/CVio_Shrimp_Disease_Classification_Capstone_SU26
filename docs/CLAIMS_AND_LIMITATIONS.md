# Claims, evidence, and limitations

## Claims supported by retained evidence

- On the fixed 173-image SDI-4 seed-42 test, the **reported official** selected
  result has Accuracy 0.913295 and Macro-F1 0.910137 versus 0.890200/0.890200
  for CE. Neither official-final binary was located.
- The two released EXT-3-Original checkpoints match exact referenced hashes,
  have three outputs in order Healthy/BG/WSSV, and match the audited package's
  CE and proposed roles.
- The EXT-3 proposed checkpoint contains the intended late SimAM-DCFR path;
  its CE counterpart does not.
- The audited combined SDI-4 + EXT-3-Original package reverses the ordering:
  CE Macro-F1 0.865105 exceeds proposed Macro-F1 0.821801.
- A historical SDI-4 package reports improved mean Macro-F1 over its CE
  comparator for five controlled corruptions. That evidence is not transferred
  to the unresolved official-final checkpoint.

## Claims explicitly unsupported

- SHA `4305e491...38c1` is not the official proposed model. It is a
  `HISTORICAL_NONFINAL` CE architecture and reproduces Macro-F1 0.863062.
- The historical Macro-F1 0.905448 class-wise report and confusion matrix are
  not the official-final 0.910137 report.
- The current FP32/FP16 TFLite files are not verified official-proposed exports.
- EXT-3 checkpoints are not four-class SDI-4 checkpoints.
- The method is not shown to be universally better: the combined-regime result
  is a direct counterexample in the retained evidence.

## Threats to validity

- **Single seed and split.** No multi-seed mean, variance, confidence interval,
  or statistical significance test is available.
- **Unit independence.** The split is image-level and is not proven to prevent
  shrimp/specimen-level leakage.
- **Checkpoint gap.** Reported official SDI-4 metrics cannot currently be bound
  to a stable binary hash.
- **EXT-3 rerun gap.** The 47 source images were not present for an independent
  inference rerun; verification used exact hashes and the retained result
  package.
- **Domain shift.** Controlled corruptions approximate acquisition failures but
  do not establish robustness in farms, laboratories, devices, or regions.
- **Preprocessing.** Background removal can suppress or introduce visual cues.
- **Library/runtime variation.** Training may vary with GPU kernels, package
  versions, pretrained assets, and Ultralytics internals.
- **Method design space.** The LDAM → single-label ASL order is not proven
  optimal against all alternative formulations or hyperparameters.
- **Qualitative XAI.** Four selected explanation panels cannot validate causal
  reasoning, feature faithfulness, or general model behavior.

## Intended-use boundary

This is research software for reproducibility and method evaluation. It is not
a clinical, veterinary, biosecurity, or autonomous treatment system. Outputs
require domain-expert review and must not replace laboratory confirmation.
