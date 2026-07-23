# Dataset EDA source provenance

## Archive hashes

| File | SHA-256 |
|---|---|
| `dataset_eda.zip` | see `dataset_eda/manifest.json` |
| `cvio_yolo_dataset_eda_no_raw_images.zip` | see below |

## Merged canonical EDA package

The canonical EDA package in `dataset_eda/` was merged from both ZIP archives. After comparing file lists and content, all required figures, tables, `summary.json`, `manifest.json`, `eda_report.html`, and `README.md` are present in the canonical directory.

The original ZIP archives are preserved for traceability but are not part of the public research narrative.

## Source dataset

**Canonical dataset root:** `shrimp_yolo_canonical_bg_wssv_2cls_split70_15_15`

**traintest split:** 70.11% / 15.01% / 14.88%

**Image resolution:** 2048 × 2048 (all images)

**Classes:** 0:BG, 1:WSSV

**Total images:** 746
**Total boxes:** 5,569
**Audit issues:** 0
**Empty/missing-label images:** 0

## Validation

All expected totals match:
- Train images: 523 ✓
- Val images: 112 ✓
- Test images: 111 ✓
- BG boxes: 2,121 ✓
- WSSV boxes: 3,448 ✓
- Train boxes: 3,815 ✓
- Val boxes: 836 ✓
- Test boxes: 918 ✓
- Mean boxes per image: 7.465 ✓
- Median boxes per image: 6 ✓
- Tiny boxes: 3,933 (70.62%) ✓
- Small boxes: 1,598 (28.69%) ✓
- Medium boxes: 38 (0.68%) ✓
