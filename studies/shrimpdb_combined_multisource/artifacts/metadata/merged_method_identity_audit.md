# Merged Method-Identity Audit

Status: **verified**.

The merged package is authoritative for the selected best-by-dataset presentation. Its
display label is retained separately from `actual_source_method`.

| Selected dataset | Display label | Verified source method | Label/source match | Confidence |
|---|---|---|---|---|
| ShrimpDB-3 | ASL-LDAM + SimAM-DCFR | ASL-LDAM + SimAM-DCFR | true | high |
| Combined-4 | ASL-LDAM + SimAM-DCFR | CE Baseline | false | high |

Evidence was cross-checked against `FINAL_BEST_RESULTS.json`, the selected provenance JSON,
the selected raw metrics, and the selected checkpoint SHA-256 values. The historical
`method_identity_audit.json` is preserved unchanged and documents why the old standalone
package presentation could not support the requested cross-assignment. No raw metrics,
predictions, confusion matrices, or checkpoints were edited.
