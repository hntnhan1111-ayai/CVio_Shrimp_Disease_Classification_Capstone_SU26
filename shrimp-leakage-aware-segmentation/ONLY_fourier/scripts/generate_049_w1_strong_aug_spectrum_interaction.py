import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ONLY_FOURIER = ROOT / "shrimp-leakage-aware-segmentation" / "ONLY_fourier"
SOURCE = ONLY_FOURIER / "evidence" / "intake_downloads" / "fourier-visualization.ipynb"
OUTPUT = ONLY_FOURIER / "049_w1_strong_aug_spectrum_interaction_kaggle_seed42.ipynb"


def source_text(cell):
    return "".join(cell.get("source", []))


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        cell["source"][-1] += "\n"


def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"][:11]

set_source(
    cells[0],
    """# 049 - W1 and Strong-Augmentation Spectrum Interaction

This Kaggle notebook generates quantitative and visual evidence for the seed-42 Path 15 observation:

- no augmentation: labeled mask mAP50 `0.235374 -> 0.345105` with W1;
- strong augmentation, hook off: labeled mask mAP50 `0.595095 -> 0.577126` with W1.

It does **not** retrain a model. It analyzes what happens to spatial images and frequency spectra when the recorded W1 high-pass preprocessing is combined with the active image transformations of the strong policy.

The paired paths are:

1. original;
2. W1 only;
3. strong augmentation only;
4. W1 then strong augmentation, matching Path 15 training order;
5. strong augmentation then W1, a counterfactual order.

All comparisons use shared random augmentation parameters. Results are aggregated across healthy, BG, WSSV, and co-infection examples instead of relying on one selected image.
""",
)

set_source(
    cells[1],
    """## Run Controls and Primary Outputs

The notebook exports a sample manifest, per-realization measurements, aggregated spectral statistics, radial profiles, representative spatial/spectrum figures, configuration audits, and an automatically generated interpretation summary.

The strong-policy replica covers the segmentation-relevant transformations recorded in Path 15: letterbox/resize, rotation, translation, scale, HSV, horizontal flip, and vertical flip. Mosaic, MixUp, CutMix, and Copy-Paste were zero. The hidden Albumentations hook was disabled. `erasing=0.15` is audited separately because Ultralytics applies erasing in its classification pipeline rather than the standard segmentation transform builder.
""",
)

set_source(
    cells[2],
    """from pathlib import Path

SEED = 42
SAMPLE_SPLIT = 'test'
SAMPLES_PER_STRATUM = 6
MAX_IMAGES_TO_SHOW = 24
AUGMENTATION_REALIZATIONS = 12
IMGSZ = 640
RADIAL_BINS = 96

W1_SIGMA = 50.0
W1_ALPHA = 0.10

if Path('/kaggle/working').exists():
    WORK_DIR = Path('/kaggle/working')
elif Path('/content').exists():
    WORK_DIR = Path('/content')
else:
    WORK_DIR = Path.cwd()

DATASET_DIR = WORK_DIR / 'shrimpDisHandSegV2-1'
OUTPUT_DIR = WORK_DIR / '049_w1_strong_aug_spectrum_interaction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('WORK_DIR:', WORK_DIR)
print('DATASET_DIR:', DATASET_DIR)
print('OUTPUT_DIR:', OUTPUT_DIR)
print('Analysis split:', SAMPLE_SPLIT)
print('Samples per stratum:', SAMPLES_PER_STRATUM)
print('Augmentation realizations:', AUGMENTATION_REALIZATIONS)
""",
)

imports = source_text(cells[4])
imports = imports.replace(
    "('cv2', 'opencv-python-headless'),",
    "('cv2', 'opencv-python-headless'),\n    ('ultralytics', 'ultralytics==8.4.62'),",
)
imports = imports.replace(
    "for import_name, package_name in REQUIRED_IMPORTS:\n"
    "    if importlib.util.find_spec(import_name) is None:\n"
    "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package_name])\n",
    "for import_name, package_name in REQUIRED_IMPORTS:\n"
    "    if import_name == 'ultralytics':\n"
    "        if installed_version('ultralytics') != '8.4.62':\n"
    "            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'ultralytics==8.4.62'])\n"
    "    elif importlib.util.find_spec(import_name) is None:\n"
    "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package_name])\n",
)
imports = imports.replace(
    "import yaml\n",
    "import yaml\nimport inspect\nimport ultralytics\n\nif ultralytics.__version__ != '8.4.62':\n"
    "    raise RuntimeError(f'Expected ultralytics 8.4.62, got {ultralytics.__version__}')\n",
)
set_source(cells[4], imports)

controls = source_text(cells[10])
controls = controls.replace("per_disease=SAMPLES_PER_DISEASE", "per_disease=SAMPLES_PER_STRATUM")
controls = controls.replace("selected_images = sample_images()", "selected_images = sample_images()")
set_source(cells[10], controls)


audit_cell = r'''from ultralytics.data import augment as ultra_augment

STRONG_AUG_CONFIG = {
    'auto_augment': None,
    'erasing': 0.15,
    'mosaic': 0.0,
    'mixup': 0.0,
    'cutmix': 0.0,
    'copy_paste': 0.0,
    'fliplr': 0.5,
    'flipud': 0.3,
    'hsv_h': 0.05,
    'hsv_s': 0.50,
    'hsv_v': 0.40,
    'degrees': 10.0,
    'translate': 0.10,
    'scale': 0.50,
    'shear': 0.0,
    'perspective': 0.0,
    'multi_scale': False,
    'bgr': 0.0,
}

v8_builder = getattr(ultra_augment, 'v8_transforms', None)
classification_builder = getattr(ultra_augment, 'classify_augmentations', None)
if v8_builder is None:
    raise RuntimeError('Ultralytics 8.4.62 does not expose v8_transforms; cannot audit the segmentation pipeline.')
v8_source = inspect.getsource(v8_builder)
classification_source = inspect.getsource(classification_builder) if classification_builder is not None else ''
audit = {
    'ultralytics_version': ultralytics.__version__,
    'recorded_strong_config': STRONG_AUG_CONFIG,
    'active_replica_operations': [
        'letterbox_resize', 'random_rotation', 'random_translation', 'random_scale',
        'random_hsv', 'random_vertical_flip', 'random_horizontal_flip',
    ],
    'zero_disabled_operations': ['mosaic', 'mixup', 'cutmix', 'copy_paste', 'shear', 'perspective'],
    'hidden_albumentations_hook': 'disabled in Path 15',
    'erasing_in_v8_segmentation_builder_source': 'erasing' in v8_source,
    'classification_builder_found': classification_builder is not None,
    'erasing_in_classification_builder_source': 'erasing' in classification_source,
    'replica_scope': (
        'Image-only deterministic replica of Ultralytics 8.4.62 transform equations and order. '
        'It is intended for paired spectral analysis, not as a replacement training loader.'
    ),
    'recorded_training_order': 'offline W1 on train/valid/test, then online train-only strong augmentation',
}

(OUTPUT_DIR / 'ultralytics_augmentation_contract.json').write_text(
    json.dumps(audit, indent=2), encoding='utf-8'
)
(OUTPUT_DIR / 'ultralytics_v8_transforms_source.txt').write_text(v8_source, encoding='utf-8')
print(json.dumps(audit, indent=2))

if audit['erasing_in_v8_segmentation_builder_source']:
    print('WARNING: erasing appears in v8_transforms source; inspect the saved source before reporting it as inactive.')
else:
    print('Audit result: erasing is not part of v8_transforms; it is not included in this segmentation image replica.')
'''


transform_cell = r'''def w1_highpass_bgr(image_bgr, sigma=W1_SIGMA, alpha=W1_ALPHA):
    """Path 15 W1: per-channel FFT Gaussian low-pass reconstruction and residual boost."""
    image = image_bgr.astype(np.float32)
    height, width = image.shape[:2]
    y = np.arange(height, dtype=np.float32) - height / 2.0
    x = np.arange(width, dtype=np.float32) - width / 2.0
    yy, xx = np.meshgrid(y, x, indexing='ij')
    lowpass = np.exp(-(xx * xx + yy * yy) / (2.0 * float(sigma) ** 2))[..., None]
    freq = np.fft.fftshift(np.fft.fft2(image, axes=(0, 1)), axes=(0, 1))
    low = np.fft.ifft2(
        np.fft.ifftshift(freq * lowpass, axes=(0, 1)), axes=(0, 1)
    ).real
    enhanced = image + float(alpha) * (image - low)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def letterbox_image(image, new_shape=IMGSZ, color=(114, 114, 114), interpolation=cv2.INTER_LINEAR):
    height, width = image.shape[:2]
    ratio = min(new_shape / height, new_shape / width)
    resized_w = int(round(width * ratio))
    resized_h = int(round(height * ratio))
    pad_w = new_shape - resized_w
    pad_h = new_shape - resized_h
    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
    bordered = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return bordered, {'ratio': ratio, 'left': left, 'top': top}


def sample_strong_parameters(seed, width=IMGSZ, height=IMGSZ):
    """Draw once so every processing path receives the same augmentation parameters."""
    py_rng = random.Random(int(seed))
    np_rng = np.random.default_rng(int(seed) + 1000003)
    angle = py_rng.uniform(-STRONG_AUG_CONFIG['degrees'], STRONG_AUG_CONFIG['degrees'])
    scale = py_rng.uniform(1.0 - STRONG_AUG_CONFIG['scale'], 1.0 + STRONG_AUG_CONFIG['scale'])
    tx = py_rng.uniform(0.5 - STRONG_AUG_CONFIG['translate'], 0.5 + STRONG_AUG_CONFIG['translate']) * width
    ty = py_rng.uniform(0.5 - STRONG_AUG_CONFIG['translate'], 0.5 + STRONG_AUG_CONFIG['translate']) * height

    center = np.eye(3, dtype=np.float32)
    center[0, 2] = -width / 2.0
    center[1, 2] = -height / 2.0
    rotation = np.eye(3, dtype=np.float32)
    rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)
    translation = np.eye(3, dtype=np.float32)
    translation[0, 2] = tx
    translation[1, 2] = ty
    matrix = translation @ rotation @ center

    hsv_draw = np_rng.uniform(-1.0, 1.0, 3).astype(np.float32)
    return {
        'angle': float(angle),
        'scale': float(scale),
        'translate_x_px': float(tx - width / 2.0),
        'translate_y_px': float(ty - height / 2.0),
        'matrix': matrix,
        'hue_delta': float(hsv_draw[0] * STRONG_AUG_CONFIG['hsv_h'] * 180.0),
        'sat_gain': float(1.0 + hsv_draw[1] * STRONG_AUG_CONFIG['hsv_s']),
        'val_gain': float(1.0 + hsv_draw[2] * STRONG_AUG_CONFIG['hsv_v']),
        'flipud': py_rng.random() < STRONG_AUG_CONFIG['flipud'],
        'fliplr': py_rng.random() < STRONG_AUG_CONFIG['fliplr'],
    }


def apply_random_hsv_bgr(image_bgr, params):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    x = np.arange(256, dtype=np.float32)
    lut_hue = ((x + params['hue_delta']) % 180).astype(np.uint8)
    lut_sat = np.clip(x * params['sat_gain'], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * params['val_gain'], 0, 255).astype(np.uint8)
    lut_sat[0] = 0
    h, s, v = cv2.split(hsv)
    transformed = cv2.merge((cv2.LUT(h, lut_hue), cv2.LUT(s, lut_sat), cv2.LUT(v, lut_val)))
    return cv2.cvtColor(transformed, cv2.COLOR_HSV2BGR)


def apply_strong_replica(letterboxed_bgr, params, is_mask=False):
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    border_value = 0 if is_mask else (114, 114, 114)
    output = cv2.warpAffine(
        letterboxed_bgr,
        params['matrix'][:2],
        dsize=(IMGSZ, IMGSZ),
        flags=interpolation,
        borderValue=border_value,
    )
    if not is_mask:
        output = apply_random_hsv_bgr(output, params)
    if params['flipud']:
        output = np.flipud(output)
    if params['fliplr']:
        output = np.fliplr(output)
    return np.ascontiguousarray(output)


def polygon_mask_for_image(image_path, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for line in read_label_lines(image_path):
        values = line.split()
        coords = np.asarray([float(value) for value in values[1:]], dtype=np.float32)
        if len(coords) < 6 or len(coords) % 2:
            continue
        points = coords.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask


def build_processing_paths(image_bgr, params):
    original_lb, _ = letterbox_image(image_bgr)
    w1_original = w1_highpass_bgr(image_bgr)
    w1_lb, _ = letterbox_image(w1_original)
    strong_only = apply_strong_replica(original_lb, params)
    w1_then_strong = apply_strong_replica(w1_lb, params)
    strong_then_w1 = w1_highpass_bgr(strong_only)
    return {
        'original': original_lb,
        'w1_only': w1_lb,
        'strong_only': strong_only,
        'w1_then_strong': w1_then_strong,
        'strong_then_w1': strong_then_w1,
    }
'''


metrics_cell = r'''MODE_ORDER = ['original', 'w1_only', 'strong_only', 'w1_then_strong', 'strong_then_w1']
MODE_LABELS = {
    'original': 'Original',
    'w1_only': 'W1 only',
    'strong_only': 'Strong aug only',
    'w1_then_strong': 'W1 -> strong aug',
    'strong_then_w1': 'Strong aug -> W1',
}


def image_stratum(image_path):
    ids = class_ids_for_image(image_path)
    if not ids:
        return 'Healthy'
    if len(ids) >= 2:
        return 'WSSV_BG'
    return class_names[ids[0]] if ids[0] < len(class_names) else str(ids[0])


def centered_power(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    return np.abs(spectrum) ** 2


def normalized_radius(shape):
    height, width = shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    yy -= (height - 1) / 2.0
    xx -= (width - 1) / 2.0
    radius = np.sqrt(xx * xx + yy * yy)
    return radius / max(float(radius.max()), 1.0)


RADIUS = normalized_radius((IMGSZ, IMGSZ))
BAND_MASKS = {
    'low': RADIUS <= 0.15,
    'mid': (RADIUS > 0.15) & (RADIUS <= 0.40),
    'high': RADIUS > 0.40,
}


def radial_profile(power, bins=RADIAL_BINS):
    indexes = np.minimum((RADIUS * bins).astype(np.int32), bins - 1)
    sums = np.bincount(indexes.ravel(), weights=power.ravel(), minlength=bins)
    counts = np.bincount(indexes.ravel(), minlength=bins)
    return sums / np.maximum(counts, 1)


def measurement_row(image_bgr, lesion_mask, power):
    total = float(power.sum()) + 1e-12
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    lesion_pixels = lesion_mask > 0
    background_pixels = ~lesion_pixels
    row = {
        'mean_intensity': float(gray.mean()),
        'pixel_std': float(gray.std()),
        'clipping_fraction': float(np.mean((image_bgr <= 0) | (image_bgr >= 255))),
        'laplacian_variance': float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        'gradient_mean': float(gradient.mean()),
        'lesion_gradient_mean': float(gradient[lesion_pixels].mean()) if lesion_pixels.any() else np.nan,
        'background_gradient_mean': float(gradient[background_pixels].mean()) if background_pixels.any() else np.nan,
    }
    for band_name, band_mask in BAND_MASKS.items():
        energy = float(power[band_mask].sum())
        row[f'{band_name}_energy'] = energy
        row[f'{band_name}_energy_share'] = energy / total
    return row
'''


run_cell = r'''records = []
radial_records = []
sample_rows = []
representatives = {}

for image_index, image_path in enumerate(selected_images):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f'Could not read {image_path}')
    height, width = image_bgr.shape[:2]
    raw_mask = polygon_mask_for_image(image_path, height, width)
    mask_lb, _ = letterbox_image(raw_mask, interpolation=cv2.INTER_NEAREST, color=0)
    stratum = image_stratum(image_path)
    sample_rows.append({
        'image_index': image_index,
        'filename': image_path.name,
        'stratum': stratum,
        'original_height': height,
        'original_width': width,
        'label_instances': len(read_label_lines(image_path)),
    })

    for realization in range(AUGMENTATION_REALIZATIONS):
        draw_seed = SEED * 1_000_000 + image_index * 1_000 + realization
        params = sample_strong_parameters(draw_seed)
        paths = build_processing_paths(image_bgr, params)
        transformed_mask = apply_strong_replica(mask_lb, params, is_mask=True)

        if realization == 0 and stratum not in representatives:
            representatives[stratum] = {
                'filename': image_path.name,
                'paths': {key: value.copy() for key, value in paths.items()},
            }

        powers = {mode: centered_power(image) for mode, image in paths.items()}
        for mode in MODE_ORDER:
            mask_for_mode = transformed_mask if mode in {'strong_only', 'w1_then_strong', 'strong_then_w1'} else mask_lb
            row = {
                'filename': image_path.name,
                'stratum': stratum,
                'realization': realization,
                'draw_seed': draw_seed,
                'mode': mode,
                'angle': params['angle'],
                'scale': params['scale'],
                'translate_x_px': params['translate_x_px'],
                'translate_y_px': params['translate_y_px'],
                'hue_delta': params['hue_delta'],
                'sat_gain': params['sat_gain'],
                'val_gain': params['val_gain'],
                'flipud': params['flipud'],
                'fliplr': params['fliplr'],
            }
            row.update(measurement_row(paths[mode], mask_for_mode, powers[mode]))
            records.append(row)

            radial = radial_profile(powers[mode])
            radial_total = float(radial.sum()) + 1e-12
            for bin_index, value in enumerate(radial):
                radial_records.append({
                    'filename': image_path.name,
                    'stratum': stratum,
                    'realization': realization,
                    'mode': mode,
                    'normalized_radius': (bin_index + 0.5) / RADIAL_BINS,
                    'radial_power': float(value),
                    'radial_power_share': float(value / radial_total),
                })

measurements_df = pd.DataFrame(records)
radial_df = pd.DataFrame(radial_records)
sample_manifest_df = pd.DataFrame(sample_rows)

# Paired deltas quantify the incremental W1 effect under matched augmentation.
pair_index = ['filename', 'stratum', 'realization']
wide = measurements_df.pivot(index=pair_index, columns='mode')
paired_rows = []
for index_values in wide.index:
    base = dict(zip(pair_index, index_values))
    for metric in [
        'low_energy_share', 'mid_energy_share', 'high_energy_share', 'clipping_fraction',
        'laplacian_variance', 'gradient_mean', 'lesion_gradient_mean', 'background_gradient_mean',
    ]:
        values = wide[metric].loc[index_values]
        paired_rows.append({
            **base,
            'metric': metric,
            'w1_effect_without_aug': values['w1_only'] - values['original'],
            'w1_effect_after_matched_aug': values['w1_then_strong'] - values['strong_only'],
            'order_effect_strong_then_w1_minus_w1_then_strong': values['strong_then_w1'] - values['w1_then_strong'],
        })
paired_df = pd.DataFrame(paired_rows)

metric_columns = [
    'mean_intensity', 'pixel_std', 'clipping_fraction', 'laplacian_variance', 'gradient_mean',
    'lesion_gradient_mean', 'background_gradient_mean', 'low_energy_share', 'mid_energy_share',
    'high_energy_share',
]
aggregate_df = measurements_df.groupby(['stratum', 'mode'])[metric_columns].agg(['mean', 'std', 'count'])
aggregate_df.columns = ['_'.join(column) for column in aggregate_df.columns]
aggregate_df = aggregate_df.reset_index()

paired_aggregate_df = paired_df.groupby(['stratum', 'metric'])[
    ['w1_effect_without_aug', 'w1_effect_after_matched_aug', 'order_effect_strong_then_w1_minus_w1_then_strong']
].agg(['mean', 'std', 'count'])
paired_aggregate_df.columns = ['_'.join(column) for column in paired_aggregate_df.columns]
paired_aggregate_df = paired_aggregate_df.reset_index()

sample_manifest_df.to_csv(OUTPUT_DIR / '049_sample_manifest.csv', index=False)
measurements_df.to_csv(OUTPUT_DIR / '049_per_realization_spectrum_metrics.csv', index=False)
aggregate_df.to_csv(OUTPUT_DIR / '049_aggregate_spectrum_metrics.csv', index=False)
paired_df.to_csv(OUTPUT_DIR / '049_paired_w1_effects.csv', index=False)
paired_aggregate_df.to_csv(OUTPUT_DIR / '049_paired_w1_effects_aggregate.csv', index=False)
radial_df.to_csv(OUTPUT_DIR / '049_radial_power_profiles.csv', index=False)

print('Samples:', len(sample_manifest_df))
print('Per-realization rows:', len(measurements_df))
display(sample_manifest_df)
display(paired_aggregate_df)
'''


figures_cell = r'''def log_power_for_display(image_bgr):
    return 10.0 * np.log10(centered_power(image_bgr) + 1e-12)


for stratum in ['Healthy', 'BG', 'WSSV', 'WSSV_BG']:
    if stratum not in representatives:
        print('No representative available for', stratum)
        continue
    item = representatives[stratum]
    paths = item['paths']
    display_spectra = {mode: log_power_for_display(paths[mode]) for mode in MODE_ORDER}
    all_values = np.concatenate([value.ravel() for value in display_spectra.values()])
    vmin, vmax = np.percentile(all_values, [3.0, 99.7])

    difference_reference = {
        'original': 'original',
        'w1_only': 'original',
        'strong_only': 'original',
        'w1_then_strong': 'strong_only',
        'strong_then_w1': 'strong_only',
    }
    signed_differences = {
        mode: display_spectra[mode] - display_spectra[difference_reference[mode]] for mode in MODE_ORDER
    }
    nonzero_differences = np.concatenate(
        [np.abs(signed_differences[mode]).ravel() for mode in MODE_ORDER if mode != 'original']
    )
    difference_limit = max(float(np.percentile(nonzero_differences, 99.0)), 1e-6)

    fig, axes = plt.subplots(3, len(MODE_ORDER), figsize=(19, 10.5), constrained_layout=True)
    for column, mode in enumerate(MODE_ORDER):
        axes[0, column].imshow(cv2.cvtColor(paths[mode], cv2.COLOR_BGR2RGB))
        axes[0, column].set_title(MODE_LABELS[mode], fontsize=11)
        axes[0, column].axis('off')
        axes[1, column].imshow(display_spectra[mode], cmap='magma', vmin=vmin, vmax=vmax)
        axes[1, column].axis('off')
        axes[2, column].imshow(
            signed_differences[mode], cmap='coolwarm', vmin=-difference_limit, vmax=difference_limit
        )
        axes[2, column].axis('off')
    axes[0, 0].set_ylabel('Spatial image', fontsize=11)
    axes[1, 0].set_ylabel('FFT log power\n(shared scale)', fontsize=11)
    axes[2, 0].set_ylabel('Signed dB difference\nvs paired reference', fontsize=11)
    fig.suptitle(f'{stratum}: {item["filename"]}', fontsize=14)
    stem = stratum.lower()
    fig.savefig(OUTPUT_DIR / f'049_{stem}_spatial_spectrum_orders.png', dpi=220, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / f'049_{stem}_spatial_spectrum_orders.pdf', bbox_inches='tight')
    plt.show()

radial_summary = radial_df.groupby(['mode', 'normalized_radius'])['radial_power'].agg(['mean', 'std', 'count']).reset_index()
radial_summary['se'] = radial_summary['std'] / np.sqrt(radial_summary['count'].clip(lower=1))
radial_summary.to_csv(OUTPUT_DIR / '049_radial_power_profile_aggregate.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
colors = {
    'original': '#222222', 'w1_only': '#d62728', 'strong_only': '#1f77b4',
    'w1_then_strong': '#2ca02c', 'strong_then_w1': '#9467bd',
}
for mode in MODE_ORDER:
    subset = radial_summary[radial_summary['mode'] == mode]
    x = subset['normalized_radius'].to_numpy()
    y = 10.0 * np.log10(subset['mean'].to_numpy() + 1e-12)
    ax.plot(x, y, label=MODE_LABELS[mode], color=colors[mode], linewidth=2)
ax.axvline(0.15, color='#777777', linestyle='--', linewidth=1)
ax.axvline(0.40, color='#777777', linestyle='--', linewidth=1)
ax.set_xlabel('Normalized radial frequency')
ax.set_ylabel('Mean radial power (dB)')
ax.set_title('Average radial spectrum across paired images and realizations')
ax.grid(alpha=0.25)
ax.legend()
fig.savefig(OUTPUT_DIR / '049_radial_power_comparison.png', dpi=220, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / '049_radial_power_comparison.pdf', bbox_inches='tight')
plt.show()

band_delta = paired_df[paired_df['metric'].isin(['low_energy_share', 'mid_energy_share', 'high_energy_share'])]
band_plot = band_delta.groupby('metric')[['w1_effect_without_aug', 'w1_effect_after_matched_aug']].agg(['mean', 'std'])
band_plot.to_csv(OUTPUT_DIR / '049_band_share_w1_delta_summary.csv')

x = np.arange(3)
width = 0.36
labels = ['Low', 'Mid', 'High']
metrics = ['low_energy_share', 'mid_energy_share', 'high_energy_share']
without = [band_plot.loc[m, ('w1_effect_without_aug', 'mean')] for m in metrics]
after = [band_plot.loc[m, ('w1_effect_after_matched_aug', 'mean')] for m in metrics]
fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
ax.bar(x - width / 2, without, width, label='W1 effect without augmentation', color='#d62728')
ax.bar(x + width / 2, after, width, label='Incremental W1 effect after matched augmentation', color='#2ca02c')
ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(x, labels)
ax.set_ylabel('Change in total spectral-energy share')
ax.set_title('Does strong augmentation reduce or redistribute W1 spectral emphasis?')
ax.legend()
ax.grid(axis='y', alpha=0.25)
fig.savefig(OUTPUT_DIR / '049_w1_band_energy_interaction.png', dpi=220, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / '049_w1_band_energy_interaction.pdf', bbox_inches='tight')
plt.show()
'''


summary_cell = r'''overall = paired_df.groupby('metric')[[
    'w1_effect_without_aug', 'w1_effect_after_matched_aug',
    'order_effect_strong_then_w1_minus_w1_then_strong',
]].agg(['mean', 'std', 'count'])


def mean_value(metric, column):
    return float(overall.loc[metric, (column, 'mean')])


high_without = mean_value('high_energy_share', 'w1_effect_without_aug')
high_after = mean_value('high_energy_share', 'w1_effect_after_matched_aug')
clip_without = mean_value('clipping_fraction', 'w1_effect_without_aug')
clip_after = mean_value('clipping_fraction', 'w1_effect_after_matched_aug')
gradient_without = mean_value('gradient_mean', 'w1_effect_without_aug')
gradient_after = mean_value('gradient_mean', 'w1_effect_after_matched_aug')

interaction_metrics = pd.DataFrame([
    {
        'regime': 'No augmentation, hook off',
        'no_fourier_labeled_map50': 0.235374,
        'w1_labeled_map50': 0.345105,
        'w1_delta': 0.109731,
        'no_fourier_healthy_fp': 0.390244,
        'w1_healthy_fp': 0.634146,
    },
    {
        'regime': 'Strong augmentation, hook off',
        'no_fourier_labeled_map50': 0.595095,
        'w1_labeled_map50': 0.577126,
        'w1_delta': -0.017969,
        'no_fourier_healthy_fp': 0.341463,
        'w1_healthy_fp': 0.317073,
    },
])
interaction_contrast = float(interaction_metrics.iloc[1]['w1_delta'] - interaction_metrics.iloc[0]['w1_delta'])
interaction_metrics['seed'] = SEED
interaction_metrics.to_csv(OUTPUT_DIR / '049_recorded_training_interaction_metrics.csv', index=False)

summary_lines = [
    '# W1 and Strong-Augmentation Spectrum Interaction',
    '',
    '## Recorded training observation',
    '',
    '| Regime | No Fourier labeled mAP50 | W1 labeled mAP50 | W1 delta |',
    '|---|---:|---:|---:|',
]
for row in interaction_metrics.to_dict('records'):
    summary_lines.append(
        f"| {row['regime']} | {row['no_fourier_labeled_map50']:.6f} | "
        f"{row['w1_labeled_map50']:.6f} | {row['w1_delta']:+.6f} |"
    )
summary_lines += [
    '',
    f'The seed-42 difference-in-differences interaction contrast is `{interaction_contrast:+.6f}` '
    f'({interaction_contrast * 100:+.2f} percentage points).',
    '',
    '## Measured image-spectrum interaction',
    '',
    f'- W1 high-band energy-share change without augmentation: `{high_without:+.8f}`.',
    f'- Incremental W1 high-band energy-share change after matched strong augmentation: `{high_after:+.8f}`.',
    f'- W1 clipping-fraction change without augmentation: `{clip_without:+.8f}`.',
    f'- Incremental W1 clipping-fraction change after matched strong augmentation: `{clip_after:+.8f}`.',
    f'- W1 mean-gradient change without augmentation: `{gradient_without:+.8f}`.',
    f'- Incremental W1 mean-gradient change after matched strong augmentation: `{gradient_after:+.8f}`.',
    '',
    '## Interpretation boundary',
    '',
    'These measurements show how the image distribution and spectrum change under paired preprocessing paths. '
    'They support or challenge a proposed mechanism, but they do not prove that the measured spectral change caused '
    'the model metric difference. The performance rows are seed-42 observations; multi-seed training is required '
    'before describing the negative interaction as statistically confirmed.',
    '',
    'The strong-policy visualization excludes erasing because the Ultralytics 8.4.62 segmentation transform builder '
    'does not use the classification erasing setting. Consult `ultralytics_augmentation_contract.json` and the saved '
    '`v8_transforms` source before final report wording.',
]

summary_path = OUTPUT_DIR / '049_w1_strong_aug_spectrum_interaction_summary.md'
summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
display(interaction_metrics)
display(overall)
print(summary_path.read_text(encoding='utf-8'))
print('All outputs saved under:', OUTPUT_DIR)
'''


cells.extend(
    [
        markdown_cell("## Audit the Recorded Strong-Augmentation Contract\n"),
        code_cell(audit_cell),
        markdown_cell(
            "## W1 and Paired Strong-Augmentation Replica\n\n"
            "The implementation below follows the image-operation order used by Ultralytics 8.4.62 when mosaic and mix transforms are disabled: letterbox/resize, random affine geometry, HSV, vertical flip, and horizontal flip. Parameters are sampled once and reused for every paired path.\n"
        ),
        code_cell(transform_cell),
        markdown_cell("## Frequency and Spatial Measurements\n"),
        code_cell(metrics_cell),
        markdown_cell("## Run the Paired Analysis\n"),
        code_cell(run_cell),
        markdown_cell("## Report-Ready Spatial and Frequency-Domain Figures\n"),
        code_cell(figures_cell),
        markdown_cell("## Recorded Training Interaction and Evidence Summary\n"),
        code_cell(summary_cell),
    ]
)

notebook["cells"] = cells
notebook.setdefault("metadata", {}).setdefault("kernelspec", {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
})
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
for index, cell in enumerate(notebook["cells"]):
    if cell.get("cell_type") == "code":
        ast.parse(source_text(cell), filename=f"{OUTPUT.name}:cell_{index}")
print(f"Generated {OUTPUT}")
