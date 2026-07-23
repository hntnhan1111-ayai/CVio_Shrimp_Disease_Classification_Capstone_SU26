# Path 15 Training Log Analysis: Fourier G4 x Strong Aug x SimAM+CA

Source notebook: `C:\Users\Admin\Downloads\notebookae2e3c4caa.ipynb`

## Run Contract

| Item | Value |
|---|---|
| Notebook title | ONLY Fourier Path 15 - Fourier G4 x SimAM+CA Interaction Seed 42 |
| Split | stratified grouped-specimen |
| Seed | 42 |
| Split fingerprint | `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb` |
| Model base | `yolo11n-seg.pt` |
| Ultralytics | `8.4.62` |
| Image size | 640 |
| Epochs requested | 100 |
| Patience | 30 |
| Hidden Ultralytics hook | off |
| Strong augmentation | on |
| Fourier G4 | high-pass boost, sigma 50, alpha 0.10 |
| Attention variant | SimAM + Coordinate Attention before segment head |

## Strong Augmentation Policy

```python
{
    "auto_augment": None,
    "erasing": 0.15,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.3,
    "hsv_h": 0.05,
    "hsv_s": 0.50,
    "hsv_v": 0.40,
    "degrees": 10.0,
    "translate": 0.10,
    "scale": 0.50,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": False,
    "bgr": 0.0,
}
```

## Enabled Modes

Mode A was defined but disabled, so this notebook does **not** include the critical no-Fourier strong-augmentation baseline architecture control.

| Mode | Fourier | Strong aug | Hook | Attention | Architecture | Meaning |
|---|---|---|---|---|---|---|
| B | yes | yes | off | no | baseline YOLO11n-seg | Fourier G4 + strong aug |
| C | no | yes | off | yes | SimAM+CA | strong aug + attention |
| D | yes | yes | off | yes | SimAM+CA | Fourier G4 + strong aug + attention |

## Main Results

| Rank | Mode | Fourier | Attention | mAP50 | mAP50-95 | Healthy FP | FP/img | Miss rate | Count MAE | Healthy-aware test | Notes |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | B | yes | no | 0.577126 | 0.208817 | 0.317073 | 0.365854 | 0.147727 | 0.435606 | 0.501480 | Best mAP50 so far |
| 2 | D | yes | yes | 0.556110 | 0.198244 | 0.268293 | 0.268293 | 0.113636 | 0.354167 | 0.494527 | Lower mAP50 than B, better FP/miss/count |
| 3 | C | no | yes | 0.493754 | 0.171782 | 0.365854 | 0.414634 | 0.125000 | 0.314394 | 0.422699 | Attention alone does not beat baseline |

## Comparisons

| Comparison | mAP50 delta | Interpretation |
|---|---:|---|
| B vs prior Mode G verified best (`0.531771`) | +0.045355 | Strong augmentation with Fourier is the new best direction. |
| D vs C | +0.062356 | Within SimAM+CA architecture, Fourier G4 adds a large mAP50 gain. |
| D vs B | -0.021016 | Attention reduces peak mAP50 under Fourier, but improves healthy FP, miss rate, and count MAE. |
| B vs hook-on clean baseline anchor (`0.511424`) | +0.065702 | This is a meaningful jump over the old anchor. |

## Interpretation

The headline result is Mode B: Fourier G4 plus strong YOLO augmentation, hook off, baseline YOLO11n-seg architecture. It reaches `0.577126` labeled test mask mAP50, which is the first verified result in this thread that clearly exceeds the `0.54` target and reaches the desired `0.56-0.57` range.

The attention result is mixed. SimAM+CA with Fourier still performs strongly at `0.556110`, but it trails the baseline architecture Fourier run by about `0.021` mAP50. However, it has better healthy FP, disease miss rate, and count MAE than Mode B. This suggests attention may make the model more conservative and better calibrated, but not best for raw mAP50.

The missing control is Mode A: no Fourier, strong augmentation, hook off, baseline architecture. Without Mode A, we cannot fully separate how much of Mode B's gain comes from Fourier and how much comes from strong augmentation. Mode C is not enough as the no-Fourier control because it also changes the architecture.

## Decision

Promote Mode B as the current best single seed-42 candidate:

- Fourier G4 high-pass, sigma 50, alpha 0.10
- strong YOLO augmentation
- hidden/default hook off
- baseline YOLO11n-seg architecture

Do not promote SimAM+CA as the main best-mAP method yet. Keep it as a secondary/calibration candidate because it improves FP/miss/count behavior relative to Mode B but costs mAP50.

## Recommended Next Runs

1. Run missing Mode A:
   - no Fourier
   - strong augmentation
   - hook off
   - baseline YOLO11n-seg architecture
   - this isolates Fourier's contribution under the strong augmentation policy.

2. Rerun Mode B across seeds:
   - seeds 42, 123, 3407
   - if Mode B stays above the clean baseline and near/above 0.54, it becomes paper-grade.

3. Tune Mode B locally:
   - alpha: `0.05`, `0.08`, `0.10`, `0.12`
   - sigma: `35`, `40`, `50`, `60`
   - keep the strong augmentation policy fixed.

4. Ablate strong augmentation policy around Mode B:
   - remove `flipud=0.3`
   - reduce HSV from `0.05/0.50/0.40` to a middle setting
   - remove `erasing=0.15`
   - reduce `scale=0.50` to `0.35`

5. Keep SimAM+CA as a secondary branch:
   - its best use may be lower FP or better count metrics rather than top mAP50.
