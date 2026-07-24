# CVio Shrimp Disease Classification

Research code and paper artifacts for shrimp disease image classification.

## Study

The main paper-oriented study is [`studies/shrimpdb_combined_multisource`](studies/shrimpdb_combined_multisource/README.md).
It contains installation, dataset protocol, reproducible training commands, quantitative
results, figures, checkpoint hashes, citation, and limitations.

## Selected results

The selected results are fixed seed-42 observations. ShrimpDB-3 uses the ASL-LDAM +
SimAM-DCFR source method. The selected Combined-4 metrics use the CE Baseline checkpoint;
the common merged display label is retained as presentation metadata only.

| Dataset | Actual source method | Accuracy | Macro-F1 |
|---|---|---:|---:|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | 91.49% | 91.29% |
| Combined-4 | CE Baseline | 87.73% | 86.51% |

See the [study README](studies/shrimpdb_combined_multisource/README.md),
[CITATION.cff](CITATION.cff), and [LICENSE](LICENSE).
