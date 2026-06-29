# Path 15 Mode A Analysis: Strong Aug Baseline Without Fourier

Source notebook: `C:\Users\Admin\Downloads\fouirer-hev-aug-attention.ipynb`

## Result

Mode A is the missing no-Fourier strong-augmentation baseline:

| Mode | Fourier | Strong aug | Hook | Attention | Architecture | Labeled test mAP50 | mAP50-95 | Healthy FP | FP/img | Miss rate | Count MAE | Healthy-aware |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | no | yes | off | no | baseline YOLO11n-seg | 0.595095 | 0.211258 | 0.341463 | 0.365854 | 0.090909 | 0.293561 | 0.532634 |

Full test mask mAP50 from the YOLO validation table is `0.565`, while labeled-only test mask mAP50 is `0.595095`.

## Validity Check

| Check | Evidence |
|---|---|
| Split policy | stratified grouped-specimen |
| Split fingerprint | `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb` |
| Leakage check | `Shrimp-level leakage check passed.` |
| Train images | 905 |
| Valid images | 115 |
| Test images | 129 |
| Labeled-only test images | 88 |
| Healthy-only test images | 41 |
| Fourier | disabled; images remained original |
| Hidden Ultralytics Albumentations hook | disabled for this run |
| Model | `yolo11n-seg.pt` |
| Ultralytics | `8.4.62` |
| Epochs | 100 completed |
| Patience | 30 |

The `0.595095` value is not an obvious invalid metric. It is the labeled-only test mAP50 from the grouped-specimen evaluation.

## Training Config

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

## Interpretation

This run changes the main conclusion. Strong augmentation alone is now the best seed-42 recipe.

Matched comparison against Path 15 Mode B:

| Comparison | Fourier off | Fourier on | Delta |
|---|---:|---:|---:|
| Strong aug + hook off + baseline YOLO11n-seg | 0.595095 | 0.577126 | -0.017969 |

Therefore, under this strong augmentation baseline, Fourier G4 does **not** improve mAP50. It slightly hurts raw labeled test mAP50 and also has worse miss/count behavior than Mode A:

| Metric | Mode A no Fourier | Mode B Fourier G4 |
|---|---:|---:|
| mAP50 | 0.595095 | 0.577126 |
| mAP50-95 | 0.211258 | 0.208817 |
| Healthy FP | 0.341463 | 0.317073 |
| Disease miss rate | 0.090909 | 0.147727 |
| Count MAE | 0.293561 | 0.435606 |
| Healthy-aware | 0.532634 | 0.501480 |

The only clear advantage of Mode B over Mode A is lower healthy FP. For the paper's current mAP50-first goal, Mode A is better.

## Why It Looked Weird

The earlier Path 15 notebook was missing Mode A, so Mode B looked like the best strong-augmentation result. Once Mode A is added, the high score shows that the main boost comes from the strong YOLO augmentation policy, not from Fourier.

This is not explained by patience: Mode A completed all 100 epochs, so early stopping did not control the result.

This is not explained by leakage from the visible log: the grouped split fingerprint matches the previous runs and the shrimp-level leakage check passed.

## Updated Paper Claim

Safe claim:

> Fourier improves weak/no-augmentation baselines and the clean-light hook-on baseline, but under the strongest augmentation policy tested, the no-Fourier strong-augmentation baseline is currently best.

Do not claim that Fourier caused the `0.595095` result. It did not; Mode A has Fourier off.

## Next Required Check

Run Mode A and Mode B across multiple seeds. If Mode A remains ahead, the final method should be strong augmentation, and Fourier should be presented as a useful but regime-dependent preprocessing factor rather than the final best method.
