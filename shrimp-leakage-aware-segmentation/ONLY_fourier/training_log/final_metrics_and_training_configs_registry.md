# ONLY_fourier Final Metrics And Training Config Registry

This registry collects the metric tables and training configurations that matter for the current paper direction. The single-run seed-42 target has been satisfied: Path 15 Mode A reached `0.595095` labeled-only test mask mAP50.

## Common Evaluation Contract

| Item | Value |
|---|---|
| Dataset | Roboflow `shrimpdishandsegv2`, YOLO segmentation format |
| Model family | YOLOv11n-seg |
| Primary split | stratified grouped-specimen |
| Primary seed | 42 |
| Split fingerprint | `1ffd4a250deb11598f09a0a25d1cd1522811b6e010b0228c3c60176029ff9fcb` |
| Train groups/images | 331 groups / 905 images |
| Valid groups/images | 40 groups / 115 images |
| Test groups/images | 45 groups / 129 images |
| Labeled-only test images | 88 |
| Healthy-only test images | 41 |
| Primary metric | labeled-only test mask mAP50 on the 88 diseased/labeled test images |
| Required companion metric | full test mask mAP50 on all 129 test images, including 41 healthy-empty images |
| Secondary diagnostics | labeled-only mAP50-95, full-test mAP50-95, healthy FP, FP/img, disease miss rate, count MAE, healthy-aware score |

Metric naming rule: all leaderboard `mAP` values are segmentation-mask mAP values, not box mAP. `Full test` uses the full 129-image grouped test set. `Labeled test` uses the 88-image labeled-only diseased subset and is the metric used for ranking unless stated otherwise. Some historical rows came only from earlier markdown summaries, so their full-test columns are left blank until the exact original CSV/log is available.

## Current Leaderboard

| Rank | Run family | Mode / Run | Core config | Full test mAP50 | Full test mAP50-95 | Labeled test mAP50 | Labeled test mAP50-95 | Healthy FP | FP/img | Miss rate | Count MAE | Healthy-aware | Status |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | Path 15 strong aug baseline | A | No Fourier + strong aug + hook off + baseline YOLO11n-seg | 0.565* | 0.201* | 0.595095 | 0.211258 | 0.341463 | 0.365854 | 0.090909 | 0.293561 | 0.532634 | Current best labeled-only mAP50 |
| 2 | Path 15 Fourier G4 x strong aug x attention | B | Fourier G4 + strong aug + hook off + baseline YOLO11n-seg | 0.559* | 0.203* | 0.577126 | 0.208817 | 0.317073 | 0.365854 | 0.147727 | 0.435606 | 0.501480 | Fourier hurts labeled-only mAP50 vs matched Mode A |
| 3 | Path 15 Fourier G4 x strong aug x attention | D | Fourier G4 + strong aug + hook off + SimAM+CA | 0.535* | 0.191* | 0.556110 | 0.198244 | 0.268293 | 0.268293 | 0.113636 | 0.354167 | 0.494527 | Strong secondary |
| 4 | Factorial A-H | G | Fourier high-pass s50 a0.10 + clean-light aug + hook on | 0.503098 | 0.161328 | 0.531771 | 0.170904 | 0.243902 | 0.292683 | 0.102273 | 0.437500 | 0.470165 | Fourier helps clean baseline |
| 5 | Factorial A-H | D | No Fourier + clean-light aug + hook on | 0.481532 | 0.158036 | 0.512002 | 0.167785 | 0.243902 | 0.268293 | 0.079545 | 0.321970 | 0.459581 | Accepted clean baseline |
| 6 | Historical summary | Hook-on clean baseline anchor | No Fourier + clean-light YOLO aug + hook on | - | - | 0.511424 | 0.163868 | 0.243902 | 0.268293 | 0.079545 | 0.331439 | 0.458530 | Historical anchor row; prefer Factorial D for paired full/labeled metrics |
| 7 | Path 15 Fourier G4 x strong aug x attention | C | No Fourier + strong aug + hook off + SimAM+CA | 0.451* | 0.158* | 0.493754 | 0.171782 | 0.365854 | 0.414634 | 0.125000 | 0.314394 | 0.422699 | Reproduction mismatch vs teammate |
| 8 | Mode G local high-pass sweep | T1 | Mode G with high-pass alpha 0.08 sigma 35 | 0.449433 | 0.157764 | 0.486738 | 0.169787 | 0.414634 | 0.463415 | 0.056818 | 0.371212 | 0.418192 | Local sweep |
| 9 | Factorial A-H | C | No Fourier + clean-light YOLO aug + hook off | 0.446849 | 0.150375 | 0.473408 | 0.158613 | 0.390244 | 0.439024 | 0.090909 | 0.518939 | 0.394800 | Hook-off control |
| 10 | Factorial A-H | B | Fourier high-pass only, no YOLO aug, hook off | 0.288431 | 0.072203 | 0.345105 | 0.088484 | 0.634146 | 0.926829 | 0.193182 | 0.839015 | 0.210763 | Best Fourier-only no-aug |
| 11 | Factorial A-H | A | No Fourier, no YOLO aug, hook off | 0.206948 | 0.053365 | 0.235374 | 0.060555 | 0.390244 | 0.560976 | 0.227273 | 0.429924 | 0.140762 | Lower-bound baseline |

`*` Path 15 full-test values were recovered from the YOLO validation table printed in the notebook log, which reports three decimal places. The labeled-only and healthy-aware columns come from the compact paper row and retain six-decimal precision.

## Paper Reporting Sequence

This is the cleanest way to report the work as a Fourier on/off progression. The key idea is to compare matched regimes first, then explain the sweeps as failed attempts to improve beyond the first useful Fourier setting.

| Step | Question | Notebook / source | Fourier off labeled test mAP50 | Fourier on labeled test mAP50 | Delta labeled mAP50 | Full-test companion | What it shows |
|---:|---|---|---:|---:|---:|---|---|
| 1 | What does YOLO11n-seg do on bare grouped data? | `001_clean-aug-off-hook-off-baseline.ipynb` / Factorial A | 0.235374 | - | - | Full off: 0.206948 | Bare grouped-data lower bound |
| 2 | Does the first Fourier config help bare data? | `01_fourier-highpass-alpha01sigma50.ipynb` / Factorial A vs B | 0.235374 | 0.345105 | +0.109731 | Full off/on: 0.206948 -> 0.288431 | Fourier high-pass F1 strongly improves mAP50 without YOLO aug, but healthy FP rises |
| 3 | What is the clean working baseline? | `12_only_fourier_12_factorial_A_H_highpass_s50_a0p10.ipynb` mode D | 0.512002 | - | - | Full off: 0.481532 | Clean-light aug + default hook is a strong baseline |
| 4 | Does F1 help the clean working baseline? | factorial A-H mode D vs G | 0.512002 | 0.531771 | +0.019769 | Full off/on: 0.481532 -> 0.503098 | Fourier F1 still helps under the clean-light hook-on baseline |
| 5 | Can other Fourier configs beat F1 on the clean baseline? | `013`, `014`, high-pass local sweep | 0.512002 / 0.531771 reference | best local sweep below 0.531771 | no | Best local full: 0.449433 | Other tested Fourier settings did not beat F1 with the clean baseline |
| 6 | What is the stronger augmentation / attention base? | teammate strong-aug notebook and Path 15 mode C | see note | 0.493754 for Path 15 C | - | Path 15 C full: 0.451* | Strong-aug/attention reproduction is inconsistent; Path 15 C underperforms teammate report |
| 7 | Does F1 help the stronger augmentation / attention regime? | `015_fouirer-hev-aug-attention.ipynb` / Path 15 C vs D | 0.493754 with SimAM+CA | 0.556110 with SimAM+CA | +0.062356 | Full off/on: 0.451* -> 0.535* | Fourier clearly helps inside the SimAM+CA strong-aug branch |
| 8 | What is the best current recipe? | Path 15 mode A vs B | 0.595095 | 0.577126 | -0.017969 | Full off/on: 0.565* -> 0.559* | Strong aug + baseline architecture without Fourier is current best; Fourier is not beneficial in this matched strong-aug baseline |

Important nuance for Step 6: the teammate-reported strong-aug SimAM+CA result was about `0.55`, but Path 15 Mode C got `0.493754`. The likely cause is reproduction mismatch, especially Ultralytics version and patch/YAML details. This should be treated as a reproducibility issue, not a final conclusion that SimAM+CA is weak.

Important update for Step 8: Mode A is no longer missing. It outperforms Fourier Mode B. This means the best seed-42 recipe is currently strong augmentation without Fourier, while Fourier remains useful in weaker or clean-light regimes.

## Matched Fourier On/Off Ablations

This table is the clearest view of the Fourier effect. Each row compares a Fourier-off run and a Fourier-on run under the same split, seed, model family, augmentation policy, hook policy, and architecture where available. The ranking metric remains labeled-only test mask mAP50; full-test mAP50 is included to show whether the same direction holds when healthy-empty test images are included.

| Regime | Fourier off run | Fourier on run | Fourier config | Full test off | Full test on | Delta full | Labeled test off | Labeled test on | Delta labeled | Healthy FP off | Healthy FP on | Interpretation |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bare YOLO, no explicit aug, hook off | Factorial A | Factorial B | high-pass s50 a0.10 | 0.206948 | 0.288431 | +0.081483 | 0.235374 | 0.345105 | +0.109731 | 0.390244 | 0.634146 | Fourier strongly improves diseased mAP, but substantially increases healthy FP. |
| No clean-light aug, default hook on | Factorial H | Factorial F | high-pass s50 a0.10 | 0.282207 | 0.269136 | -0.013071 | 0.306111 | 0.299415 | -0.006696 | 0.365854 | 0.414634 | Fourier does not help when only the default hook is active. |
| Clean-light aug, hook off | Factorial C | Factorial E | high-pass s50 a0.10 | 0.446849 | 0.349002 | -0.097847 | 0.473408 | 0.408162 | -0.065246 | 0.390244 | 0.365854 | Fourier hurts this regime despite a small healthy-FP improvement. |
| Clean-light aug, default hook on | Factorial D | Factorial G | high-pass s50 a0.10 | 0.481532 | 0.503098 | +0.021566 | 0.512002 | 0.531771 | +0.019769 | 0.243902 | 0.243902 | Best clean-light Fourier effect: small but consistent full/labeled gain with unchanged healthy FP. |
| Strong aug, hook off, baseline YOLO11n-seg | Path 15 A | Path 15 B | G4 high-pass s50 a0.10 | 0.565* | 0.559* | -0.006* | 0.595095 | 0.577126 | -0.017969 | 0.341463 | 0.317073 | Fourier slightly reduces mAP50 in the strongest baseline-architecture regime while improving healthy FP. |
| Strong aug, hook off, SimAM+CA | Path 15 C | Path 15 D | G4 high-pass s50 a0.10 | 0.451* | 0.535* | +0.084* | 0.493754 | 0.556110 | +0.062356 | 0.365854 | 0.268293 | Fourier clearly helps the attention branch and improves healthy FP. |

Takeaway: Fourier is not uniformly beneficial. It is strongest in the weak/bare regime and in the reproduced SimAM+CA branch, gives a modest positive effect for the clean-light hook-on baseline, and is negative for the current strongest no-attention strong-augmentation baseline.

## Notebook Coverage Audit

| Notebook / log | Role in story | Keep in main paper? |
|---|---|---|
| `001_clean-aug-off-hook-off-baseline.ipynb` | Step 1 bare YOLO baseline | yes |
| `00_clean-aug-hook-off.ipynb` / `only_fourier_01_light_aug_hook_off_baseline.ipynb` | clean-light hook-off control | maybe, as ablation/context |
| `01_fourier-highpass-alpha01sigma50.ipynb` | Step 2 initial F1 on bare data | yes |
| `02_fourier-highpass-alpha03sigma50.ipynb` | stronger F1 alpha control; worse than alpha 0.10 | appendix/sweep |
| `03`-`07` Fourier transform notebooks | band-pass, damping, low-frequency flatten, homomorphic | appendix/sweep, supports stopping broad Fourier-only search |
| `08`-`10` train-only / train-copy Fourier augmentation variants | early Fourier augmentation attempts | appendix/sweep, not main result |
| `11` alpha sweep | high-pass alpha 0.05/0.15/0.20 | appendix/sweep |
| `12` factorial A-H | Steps 3-4, clean-light/hook/Fourier interaction | yes |
| `013-mode-g-fourier-tuning-seed42.ipynb` | Fourier type/config tuning around Mode G | appendix/sweep |
| `014-mode-g-highpass-local-sweep-seed42.ipynb` | local alpha/sigma sweep around Mode G | yes/appendix, supports "F1 stayed best" |
| `015_fouirer-hev-aug-attention.ipynb` / Path 15 | Steps 6-8, Fourier Mode B `0.577126` and no-Fourier Mode A `0.595095` | yes |
| `u2net_bgrem_clean_aug_hook_on.ipynb` | background removal negative result | no main paper, maybe mention dropped |
| `99_noise-ablation-baseline.ipynb` | robustness/noise side study | separate paper/appendix only |

## Main Training Configs

### Config A: Strict No-Aug Hook-Off Baseline

Purpose: lower-bound control for "model + grouped data only."

| Item | Value |
|---|---|
| Fourier | off |
| Explicit YOLO augmentation | off |
| Hidden/default Ultralytics hook | off |
| Model | `yolo11n-seg.pt` |
| Epochs / patience | 100 / 40 |
| Train images | 905 |
| Result | `0.235374` labeled test mask mAP50 |

```python
{
    "auto_augment": None,
    "erasing": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": False,
    "bgr": 0.0,
    "close_mosaic": 0,
}
```

### Config B: Clean-Light YOLO Augmentation

Purpose: accepted baseline-style augmentation before strong augmentation and Path 15.

| Item | Value |
|---|---|
| Fourier | off unless factorial mode enables it |
| Explicit YOLO augmentation | clean-light |
| Hidden/default Ultralytics hook | variable; best baseline keeps hook on |
| Model | `yolo11n-seg.pt` |
| Epochs / patience | usually 100 / 40 |
| Best reference result | `0.512002` labeled test mask mAP50 for Factorial D |

```python
{
    "auto_augment": None,
    "erasing": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.35,
    "hsv_v": 0.20,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.20,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": False,
    "bgr": 0.0,
    "close_mosaic": 0,
}
```

### Config C: Fourier G4

Purpose: best reusable Fourier preprocessing currently identified.

| Item | Value |
|---|---|
| Transform | high-pass boost |
| Sigma | 50 |
| Alpha | 0.10 |
| Application | train/valid/test images for fixed preprocessing runs |
| GPU status in Path 15 | train 905 GPU images, valid 115 GPU images, test 129 GPU images; CPU fallback 0 |

```python
FOURIER_TRANSFORM = "highpass_boost"
FOURIER_SIGMA = 50
FOURIER_ALPHA = 0.10
FOURIER_BATCH_SIZE = 4
USE_GPU_FOURIER = True
```

### Config D: Strong YOLO Augmentation

Purpose: teammate-style stronger augmentation that produced the current best result when combined with Fourier G4.

| Item | Value |
|---|---|
| Hidden/default Ultralytics hook | off |
| Mosaic / mixup / copy-paste | all off |
| Main differences vs clean-light | erasing, flipud, stronger HSV, degrees, translate, larger scale |
| Best result with Fourier G4 | Path 15 Mode B, `0.577126` |
| Best result without Fourier | Path 15 Mode A, `0.595095` |

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

### Config E: SimAM + Coordinate Attention

Purpose: lightweight attention branch around YOLO11n-seg. Not the best mAP50 method, but may improve calibration-style diagnostics.

| Item | Value |
|---|---|
| Architecture | custom YOLO11n-seg YAML with CoordAtt then SimAM before the segment head |
| Pretraining | loads `yolo11n-seg.pt` into custom architecture |
| Path 15 no-Fourier result | Mode C, `0.493754` mAP50 |
| Path 15 Fourier result | Mode D, `0.556110` mAP50 |
| Interpretation | attention reduced mAP50 relative to Mode B but improved healthy FP, miss rate, and count MAE |

```text
P3 branch: CoordAtt -> SimAM
P4 branch: CoordAtt -> SimAM
P5 branch: CoordAtt -> SimAM
Segment head consumes the three attention-refined feature maps.
```

## Experiment Family Tables

### Path 15: Fourier G4 x Strong Aug x SimAM+CA

| Mode | Fourier | Strong aug | Hook | Attention | Architecture | mAP50 | mAP50-95 | Healthy FP | FP/img | Miss rate | Count MAE | Healthy-aware |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | no | yes | off | no | baseline YOLO11n-seg | 0.595095 | 0.211258 | 0.341463 | 0.365854 | 0.090909 | 0.293561 | 0.532634 |
| B | yes | yes | off | no | baseline YOLO11n-seg | 0.577126 | 0.208817 | 0.317073 | 0.365854 | 0.147727 | 0.435606 | 0.501480 |
| D | yes | yes | off | yes | SimAM+CA | 0.556110 | 0.198244 | 0.268293 | 0.268293 | 0.113636 | 0.354167 | 0.494527 |
| C | no | yes | off | yes | SimAM+CA | 0.493754 | 0.171782 | 0.365854 | 0.414634 | 0.125000 | 0.314394 | 0.422699 |

Matched baseline-architecture comparison: Fourier G4 changes `0.595095` to `0.577126`, a `-0.017969` mAP50 delta.

### Factorial A-H: Fourier x Clean-Light Aug x Hook

| Mode | Fourier | Clean-light aug | Hook | mAP50 | mAP50-95 | Healthy FP | Miss rate | Count MAE | Healthy-aware |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| G | yes | yes | on | 0.531771 | 0.170904 | 0.243902 | 0.102273 | 0.437500 | 0.470165 |
| D | no | yes | on | 0.512002 | 0.167785 | 0.243902 | 0.079545 | 0.321970 | 0.459581 |
| C | no | yes | off | 0.473408 | 0.158613 | 0.390244 | 0.090909 | 0.518939 | 0.394800 |
| E | yes | yes | off | 0.408162 | 0.124771 | 0.365854 | 0.125000 | 0.357955 | 0.334929 |
| B | yes | no | off | 0.345105 | 0.088484 | 0.634146 | 0.193182 | 0.839015 | 0.210763 |
| H | no | no | on | 0.306111 | 0.077802 | 0.365854 | 0.272727 | 0.626894 | 0.197272 |
| F | yes | no | on | 0.299415 | 0.082874 | 0.414634 | 0.306818 | 0.732955 | 0.175281 |
| A | no | no | off | 0.235374 | 0.060555 | 0.390244 | 0.227273 | 0.429924 | 0.140762 |

### Mode G High-Pass Local Sweep

All rows keep Mode G-style interaction fixed and vary high-pass alpha/sigma locally.

| Mode | Alpha | Sigma | mAP50 | mAP50-95 | Healthy FP | Miss rate | Count MAE | Healthy-aware |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.08 | 35 | 0.486738 | 0.169787 | 0.414634 | 0.056818 | 0.371212 | 0.418192 |
| T6 | 0.12 | 40 | 0.484870 | 0.162447 | 0.341463 | 0.125000 | 0.397727 | 0.412087 |
| T3 | 0.12 | 35 | 0.476721 | 0.165574 | 0.414634 | 0.102273 | 0.342803 | 0.402776 |
| T7 | 0.10 | 45 | 0.468098 | 0.142472 | 0.219512 | 0.193182 | 0.448864 | 0.394726 |
| T8 | 0.12 | 45 | 0.461169 | 0.158254 | 0.317073 | 0.079545 | 0.306818 | 0.402189 |
| T2 | 0.10 | 35 | 0.436085 | 0.134966 | 0.341463 | 0.068182 | 0.488636 | 0.367280 |
| T5 | 0.10 | 40 | 0.435122 | 0.143280 | 0.512195 | 0.102273 | 0.543561 | 0.341384 |
| T4 | 0.08 | 40 | 0.429584 | 0.145919 | 0.341463 | 0.215909 | 0.587121 | 0.333695 |
| T9 | 0.10 | 55 | 0.400008 | 0.136314 | 0.292683 | 0.181818 | 0.337121 | 0.326610 |

### Historical Fourier-Only / No-Aug Runs

| Run | Policy | mAP50 | mAP50-95 | Healthy FP | FP/img | Miss rate | Count MAE | Healthy-aware |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| High-pass a0.1 s50 | fixed all-splits high-pass | 0.345105 | 0.088484 | 0.634146 | 0.926829 | 0.193182 | 0.839015 | 0.210763 |
| Train-only high-pass a0.1 | train transformed, valid/test original | 0.336343 | 0.091158 | 0.609756 | 0.902439 | 0.193182 | 0.873106 | 0.202734 |
| High-frequency damping a0.1 s50 | fixed all-splits damping | 0.321119 | 0.106764 | 0.390244 | 0.414634 | 0.522727 | 0.592803 | 0.174045 |
| High-pass a0.05 s50 | fixed all-splits high-pass | 0.308774 | 0.079226 | 0.341463 | 0.463415 | 0.250000 | 0.562500 | 0.209003 |
| Band-pass l12 h60 a0.2 | fixed all-splits band-pass | 0.295219 | 0.083854 | 0.268293 | 0.268293 | 0.431818 | 0.630682 | 0.172083 |
| High-pass a0.20 s50 | fixed all-splits high-pass | 0.291338 | 0.082667 | 0.292683 | 0.341463 | 0.329545 | 0.539773 | 0.185649 |
| Band-pass l20 h80 a0.2 | fixed all-splits band-pass | 0.286980 | 0.088123 | 0.195122 | 0.195122 | 0.522727 | 0.621212 | 0.157998 |
| Random high-pass train aug a0.05-0.20 | original + random Fourier train copies | 0.283641 | 0.081329 | 0.146341 | 0.170732 | 0.602273 | 0.685606 | 0.144385 |
| Low-frequency flatten b0.3 s100 | fixed all-splits lowfreq flatten | 0.278693 | 0.085208 | 0.146341 | 0.219512 | 0.465909 | 0.551136 | 0.166616 |
| Homomorphic gl0.70 gh1.20 s50 | fixed all-splits homomorphic | 0.261197 | 0.091581 | 0.414634 | 0.731707 | 0.238636 | 0.666667 | 0.150605 |
| High-pass a0.15 s50 | fixed all-splits high-pass | 0.248861 | 0.072006 | 0.146341 | 0.170732 | 0.511364 | 0.630682 | 0.125988 |
| Mixed original + high-pass a0.1 train aug | original + fixed Fourier train copies | 0.242476 | 0.073790 | 0.146341 | 0.170732 | 0.454545 | 0.541667 | 0.132577 |
| High-pass a0.3 s50 | fixed all-splits high-pass | 0.242205 | 0.080400 | 0.146341 | 0.146341 | 0.522727 | 0.647727 | 0.116776 |
| Strict no-aug hook-off baseline | no Fourier, no YOLO aug, hook off | 0.235374 | 0.060555 | 0.390244 | 0.560976 | 0.227273 | 0.429924 | 0.140762 |

## Evidence Interpretation

The main result has changed from "Fourier can improve a weak no-augmentation baseline" to "strong YOLO augmentation can exceed the target mAP50." The strongest evidence is Path 15 Mode A.

The current paper story should not claim that Fourier causes the best result. With the missing Mode A now run, Fourier G4 slightly reduces mAP50 under the strongest baseline-architecture augmentation regime.

> Fourier improves weak/no-augmentation and clean-light baselines, but the strongest tested seed-42 recipe is strong augmentation without Fourier. Fourier's effect is regime-dependent.

## Open Controls Needed Before Paper Claim

| Priority | Control | Why it matters |
|---:|---|---|
| 1 | Multi-seed Mode A and Mode B: seeds 42, 123, 3407 | Confirms whether strong augmentation remains better than Fourier+strong augmentation |
| 2 | Reproduce teammate SimAM+CA strong-aug row with Ultralytics `8.4.67` | Explains why Path 15 Mode C underperformed the teammate-reported ~0.55 |
| 3 | Threshold sweep for Mode A, B, and D | Determines whether Fourier/attention helps deployment-oriented FP or count metrics despite lower mAP50 |
| 4 | Multi-seed clean baseline / Factorial D | Gives fair confidence interval for the old baseline |
| 5 | Decide paper framing | If Mode A remains best, the paper is about augmentation-first optimization with Fourier as a regime-dependent ablation, not a Fourier-best method |

## Source Files

| Artifact | Path |
|---|---|
| Running metrics report | `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/only_fourier_running_metrics_report.md` |
| Path 15 analysis | `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/15_fourier_G4_strong_aug_attention_log_analysis.md` |
| Path 15 Mode A analysis | `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/15_mode_A_strong_aug_baseline_log_analysis.md` |
| Historical Fourier summary | `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/training_log_summary.md` |
| Factorial A-H summary CSV | `shrimp-leakage-aware-segmentation/ONLY_fourier/training_log/only_fourier_factorial_A_H_highpass_s50_a0p10_summary.csv` |
| High-pass local sweep summary CSV | `shrimp-leakage-aware-segmentation/ONLY_fourier/highpass_local_sweep/training_log/only_fourier_mode_G_highpass_local_sweep_seed42_summary.csv` |
| Path 15 source notebook | `C:\Users\Admin\Downloads\notebookae2e3c4caa.ipynb` |
