# Dataset card

## Dataset: CVio Shrimp Disease Detection

### Overview

Individual shrimp photographed onshore using consumer mobile phones. Two classes of disease manifestations:
- **BG** (class 0)
- **WSSV** (class 1)

### Intended use

Research object detection of shrimp disease manifestations. Not for autonomous veterinary diagnosis, underwater deployment, or unvalidated species/diseases.

### Split

| Split | Images | Boxes |
|---|---:|---:|
| Train | 523 | 3,815 |
| Validation | 112 | 836 |
| Test | 111 | 918 |

### Classes and distribution

| Class | Boxes | Percentage |
|---|---:|---:|
| BG | 2,121 | 38.09% |
| WSSV | 3,448 | 61.91% |

### Characteristics

- All images: **2048 × 2048**
- Mean boxes per image: 7.465
- Median boxes per image: 6
- Tiny objects (< 0.1% area): 3,933 boxes (70.62%)
- Small objects (0.1–1% area): 1,598 boxes (28.69%)
- Medium objects (1–5% area): 38 boxes (0.68%)
- Mean box area: 0.0997%
- Audit issues: 0
- Missing labels: 0

### Limitations

- Mobile-phone images only (not underwater).
- Small-object dominated: 99.31% of boxes are tiny or small.
- One geographic/farm origin; generalizability is unverified.
- Binary classification: BG vs WSSV only.
