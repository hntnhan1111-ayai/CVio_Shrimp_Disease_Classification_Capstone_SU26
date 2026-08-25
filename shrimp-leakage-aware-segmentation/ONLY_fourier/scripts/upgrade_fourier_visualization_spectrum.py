import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = (
    ROOT
    / "shrimp-leakage-aware-segmentation"
    / "ONLY_fourier"
    / "evidence"
    / "intake_downloads"
    / "fourier-visualization.ipynb"
)


def source_text(cell):
    return "".join(cell.get("source", []))


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    if text and not text.endswith("\n"):
        cell["source"][-1] += "\n"


def markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
cells = notebook["cells"]
cells = [
    cell
    for cell in cells
    if "## Defense-Ready W1 Spatial and Frequency-Domain Comparison" not in source_text(cell)
    and "# Generate a compact, thesis-defense-ready W1 figure" not in source_text(cell)
]

title = source_text(cells[0])
title = title.replace(
    "# Fourier Transform Visualization Playground",
    "# Fourier Transform Visualization: Spatial and Frequency-Domain Evidence",
)
set_source(cells[0], title)

dependency_cell = source_text(cells[4])
dependency_cell = dependency_cell.replace(
    "import importlib.metadata\nimport subprocess",
    "import importlib.metadata\nimport importlib.util\nimport subprocess",
)
old_dependency_loop = """for package_name in ['roboflow', 'pandas', 'pyyaml', 'matplotlib', 'opencv-python-headless']:
    if installed_version(package_name) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package_name])
"""
new_dependency_loop = """REQUIRED_IMPORTS = [
    ('roboflow', 'roboflow'),
    ('pandas', 'pandas'),
    ('yaml', 'pyyaml'),
    ('matplotlib', 'matplotlib'),
    ('cv2', 'opencv-python-headless'),
]
for import_name, package_name in REQUIRED_IMPORTS:
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package_name])
"""
if old_dependency_loop in dependency_cell:
    dependency_cell = dependency_cell.replace(old_dependency_loop, new_dependency_loop)
set_source(cells[4], dependency_cell)

# Normalize Roboflow's class-name representation before sampling by class id.
class_cell = source_text(cells[10])
class_cell = class_cell.replace(
    "class_names = data_config.get('names', [])\nprint('Mask classes:', class_names)",
    """class_names_obj = data_config.get('names', [])
if isinstance(class_names_obj, dict):
    class_names = [class_names_obj[key] for key in sorted(class_names_obj, key=lambda value: int(value))]
else:
    class_names = list(class_names_obj)
if len(class_names) < 2:
    raise RuntimeError(f'Expected at least BG and WSSV mask classes, got: {class_names}')
print('Mask classes:', class_names)""",
)
set_source(cells[10], class_cell)

functions = source_text(cells[12])
old_fft = r"""def fft_magnitude_image(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    lo, hi = np.percentile(mag, [1, 99])
    mag = np.clip((mag - lo) / max(1e-6, hi - lo), 0, 1)
    return mag
"""
new_fft = r'''def fft_power_spectrum(img_rgb):
    """Return shifted FFT, linear power, and log-power in dB for grayscale luminance."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    shifted = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(shifted) ** 2
    power_db = 10.0 * np.log10(power + 1e-12)
    if not np.isfinite(power_db).all():
        raise RuntimeError('Non-finite values found in FFT log-power spectrum')
    return shifted, power, power_db


def shared_spectrum_limits(original_db, transformed_db, lower_percentile=1.0, upper_percentile=99.5):
    """Use one robust display scale so spectrum brightness is directly comparable."""
    low = min(
        float(np.percentile(original_db, lower_percentile)),
        float(np.percentile(transformed_db, lower_percentile)),
    )
    high = max(
        float(np.percentile(original_db, upper_percentile)),
        float(np.percentile(transformed_db, upper_percentile)),
    )
    if not high > low:
        high = low + 1.0
    return low, high


def normalized_frequency_radius(shape_hw):
    """Radius 1.0 corresponds to Nyquist frequency along either image axis."""
    height, width = shape_hw
    fy = (np.arange(height, dtype=np.float64) - height / 2.0) / max(height / 2.0, 1.0)
    fx = (np.arange(width, dtype=np.float64) - width / 2.0) / max(width / 2.0, 1.0)
    xx, yy = np.meshgrid(fx, fy)
    return np.sqrt(xx * xx + yy * yy)


def radial_power_profile(power, bins=128):
    """Average linear spectral power in concentric radial-frequency bins."""
    radius = normalized_frequency_radius(power.shape)
    edges = np.linspace(0.0, 1.0, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.full(bins, np.nan, dtype=np.float64)
    for index in range(bins):
        selected = (radius >= edges[index]) & (radius < edges[index + 1])
        if selected.any():
            profile[index] = float(np.mean(power[selected]))
    profile_db = 10.0 * np.log10(profile + 1e-12)
    return centers, profile_db


def frequency_band_energy(original_power, transformed_power):
    """Summarize energy redistribution without using display-normalized spectra."""
    radius = normalized_frequency_radius(original_power.shape)
    valid = radius <= 1.0
    original_total = float(original_power[valid].sum())
    transformed_total = float(transformed_power[valid].sum())
    bands = [
        ('low', 0.00, 0.15),
        ('mid', 0.15, 0.40),
        ('high', 0.40, 1.00),
    ]
    rows = []
    for name, lower, upper in bands:
        selected = valid & (radius >= lower) & (radius < upper)
        original_energy = float(original_power[selected].sum())
        transformed_energy = float(transformed_power[selected].sum())
        rows.append({
            'band': name,
            'normalized_radius_min': lower,
            'normalized_radius_max': upper,
            'original_energy_percent': 100.0 * original_energy / max(original_total, 1e-12),
            'transformed_energy_percent': 100.0 * transformed_energy / max(transformed_total, 1e-12),
            'absolute_energy_gain_db': 10.0 * np.log10(
                (transformed_energy + 1e-12) / (original_energy + 1e-12)
            ),
        })
    return pd.DataFrame(rows)
'''
if old_fft in functions:
    functions = functions.replace(old_fft, new_fft)
elif "def fft_power_spectrum(" in functions:
    fft_start = functions.index("def fft_power_spectrum(")
    fft_end = functions.index("def absolute_difference_heatmap(")
    functions = functions[:fft_start] + new_fft + "\n\n" + functions[fft_end:]
else:
    raise RuntimeError("Could not locate either the original or upgraded FFT functions")

visualize_start = functions.index("def visualize_transform(")
visualize_replacement = r"""def visualize_transform(image_paths, transform_fn, title, max_images=MAX_IMAGES_TO_SHOW, show_masks=True, save_dir=None):
    rows = min(len(image_paths), max_images)
    if rows == 0:
        print('No images selected.')
        return
    fig, axes = plt.subplots(rows, 6, figsize=(22, 3.8 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, image_path in enumerate(image_paths[:rows]):
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f'Could not read selected image: {image_path}')
        img_rgb = ensure_rgb(img_bgr)
        transformed = transform_fn(img_rgb)
        if transformed.shape != img_rgb.shape:
            raise RuntimeError(
                f'Transform changed image geometry for {image_path.name}: '
                f'{img_rgb.shape} -> {transformed.shape}'
            )
        labels = read_label_lines(image_path)
        original_show = overlay_yolo_masks(img_rgb, labels) if show_masks else img_rgb
        transformed_show = overlay_yolo_masks(transformed, labels) if show_masks else transformed
        spatial_diff = np.abs(img_rgb.astype(np.float32) - transformed.astype(np.float32)).mean(axis=2)
        spatial_vmax = max(float(np.percentile(spatial_diff, 99.5)), 1e-6)

        _, _, original_db = fft_power_spectrum(img_rgb)
        _, _, transformed_db = fft_power_spectrum(transformed)
        spectrum_vmin, spectrum_vmax = shared_spectrum_limits(original_db, transformed_db)
        spectral_delta_db = transformed_db - original_db
        delta_limit = max(float(np.percentile(np.abs(spectral_delta_db), 99.5)), 1e-6)

        _, disease, _, _ = parse_shrimp_group_key(image_path.name)
        row_title = f'{disease} | {image_path.name}'
        panels = [
            (original_show, 'Original + mask' if show_masks else 'Original', None, {}),
            (transformed_show, 'Transformed + mask' if show_masks else 'Transformed', None, {}),
            (spatial_diff, 'Spatial abs diff', 'magma', {'vmin': 0.0, 'vmax': spatial_vmax}),
            (original_db, 'Original log power (dB)', 'viridis', {'vmin': spectrum_vmin, 'vmax': spectrum_vmax}),
            (transformed_db, 'Transformed log power (dB)', 'viridis', {'vmin': spectrum_vmin, 'vmax': spectrum_vmax}),
            (spectral_delta_db, 'Spectral change: after - before (dB)', 'coolwarm', {'vmin': -delta_limit, 'vmax': delta_limit}),
        ]
        for col_idx, (panel, panel_title, cmap, display_kwargs) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            if cmap:
                ax.imshow(panel, cmap=cmap, **display_kwargs)
            else:
                ax.imshow(panel)
            ax.set_title(panel_title, fontsize=10)
            ax.axis('off')
        axes[row_idx, 0].set_ylabel(row_title, fontsize=9)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_title = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title).strip('_').lower()
        out_path = save_dir / f'{safe_title}.png'
        fig.savefig(out_path, dpi=180, bbox_inches='tight')
        print('Saved corrected shared-scale spectrum figure:', out_path)
    plt.show()
"""
functions = functions[:visualize_start] + visualize_replacement
set_source(cells[12], functions)

# Make the compact spatial comparison begin with a diseased image rather than Healthy.
comparison = source_text(cells[24])
comparison = comparison.replace(
    "COMPARE_IMAGE_INDEX = 0\nimage_path = selected_images[COMPARE_IMAGE_INDEX]",
    """COMPARE_IMAGE_INDEX = next(
    (index for index, path in enumerate(selected_images) if read_label_lines(path)),
    0,
)
image_path = selected_images[COMPARE_IMAGE_INDEX]""",
)
set_source(cells[24], comparison)

defense_markdown = r"""## Defense-Ready W1 Spatial and Frequency-Domain Comparison

This figure uses the selected W1 configuration (`sigma=50`, `alpha=0.10`) and computes the spectra from the raw images, without mask overlays. The original and transformed log-power spectra use one shared display scale. The signed difference is calculated before display normalization:

\[
\Delta P(u,v)=10\log_{10}(|F_{W1}(u,v)|^2+\epsilon)
-10\log_{10}(|F_{original}(u,v)|^2+\epsilon).
\]

Positive values indicate increased spectral power after W1. The radial profile and band table are calculated from linear spectral power, not from the displayed colors.
"""

defense_code = r'''# Generate a compact, thesis-defense-ready W1 figure and numerical spectral summary.
DEFENSE_W1_SIGMA = 50
DEFENSE_W1_ALPHA = 0.10
DEFENSE_PREFERRED_CLASS_IDS = [1, 0]  # Prefer a WSSV example, then BG.


def choose_defense_image(image_paths):
    for class_id in DEFENSE_PREFERRED_CLASS_IDS:
        for candidate in image_paths:
            ids = class_ids_for_image(candidate)
            if ids == [class_id]:
                return candidate
    for candidate in image_paths:
        if class_ids_for_image(candidate):
            return candidate
    if image_paths:
        return image_paths[0]
    raise RuntimeError('No sampled images are available for the defense figure')


defense_image_path = choose_defense_image(selected_images)
defense_bgr = cv2.imread(str(defense_image_path), cv2.IMREAD_COLOR)
if defense_bgr is None:
    raise FileNotFoundError(defense_image_path)
defense_original = ensure_rgb(defense_bgr)
defense_w1 = highpass_boost(
    defense_original,
    sigma=DEFENSE_W1_SIGMA,
    alpha=DEFENSE_W1_ALPHA,
)
if defense_w1.shape != defense_original.shape:
    raise RuntimeError('W1 unexpectedly changed image geometry')

labels = read_label_lines(defense_image_path)
original_overlay = overlay_yolo_masks(defense_original, labels)
w1_overlay = overlay_yolo_masks(defense_w1, labels)
spatial_difference = np.abs(
    defense_w1.astype(np.float32) - defense_original.astype(np.float32)
).mean(axis=2)

_, original_power, original_power_db = fft_power_spectrum(defense_original)
_, w1_power, w1_power_db = fft_power_spectrum(defense_w1)
power_vmin, power_vmax = shared_spectrum_limits(original_power_db, w1_power_db)
spectral_delta_db = w1_power_db - original_power_db
spectral_delta_limit = max(float(np.percentile(np.abs(spectral_delta_db), 99.5)), 1e-6)

frequency_gain = 1.0 + DEFENSE_W1_ALPHA * (
    1.0 - lowpass_filter(defense_original.shape[:2], DEFENSE_W1_SIGMA)
)
if not np.isclose(float(frequency_gain.min()), 1.0, atol=1e-4):
    raise RuntimeError(f'Unexpected W1 DC gain: {frequency_gain.min()}')
if float(frequency_gain.max()) > 1.0 + DEFENSE_W1_ALPHA + 1e-5:
    raise RuntimeError(f'Unexpected W1 maximum gain: {frequency_gain.max()}')

radial_frequency, original_radial_db = radial_power_profile(original_power)
_, w1_radial_db = radial_power_profile(w1_power)
valid_profile = np.isfinite(original_radial_db) & np.isfinite(w1_radial_db)
if valid_profile.sum() < 10:
    raise RuntimeError('Insufficient finite radial-profile bins')

band_energy_df = frequency_band_energy(original_power, w1_power)
band_energy_df.insert(0, 'image', defense_image_path.name)
band_energy_df.insert(1, 'fourier_method', 'W1_highpass_boost')
band_energy_df.insert(2, 'sigma', DEFENSE_W1_SIGMA)
band_energy_df.insert(3, 'alpha', DEFENSE_W1_ALPHA)

radial_profile_df = pd.DataFrame({
    'image': defense_image_path.name,
    'normalized_radial_frequency': radial_frequency,
    'original_power_db': original_radial_db,
    'w1_power_db': w1_radial_db,
    'w1_minus_original_db': w1_radial_db - original_radial_db,
})

band_csv = OUTPUT_DIR / 'w1_frequency_band_energy.csv'
radial_csv = OUTPUT_DIR / 'w1_radial_power_profile.csv'
band_energy_df.to_csv(band_csv, index=False)
radial_profile_df.to_csv(radial_csv, index=False)

fig, axes = plt.subplots(2, 4, figsize=(21, 10))

axes[0, 0].imshow(original_overlay)
axes[0, 0].set_title('Original image + mask')
axes[0, 1].imshow(w1_overlay)
axes[0, 1].set_title('W1 image + mask')

spatial_limit = max(float(np.percentile(spatial_difference, 99.5)), 1e-6)
spatial_plot = axes[0, 2].imshow(
    spatial_difference, cmap='magma', vmin=0.0, vmax=spatial_limit
)
axes[0, 2].set_title('Spatial absolute difference')
fig.colorbar(spatial_plot, ax=axes[0, 2], fraction=0.046, pad=0.04, label='Mean |pixel difference|')

gain_plot = axes[0, 3].imshow(
    frequency_gain,
    cmap='plasma',
    vmin=1.0,
    vmax=1.0 + DEFENSE_W1_ALPHA,
)
axes[0, 3].set_title('Theoretical W1 frequency gain')
fig.colorbar(gain_plot, ax=axes[0, 3], fraction=0.046, pad=0.04, label='Amplitude gain')

original_spectrum_plot = axes[1, 0].imshow(
    original_power_db, cmap='viridis', vmin=power_vmin, vmax=power_vmax
)
axes[1, 0].set_title('Original log-power spectrum')
fig.colorbar(original_spectrum_plot, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Power (dB)')

w1_spectrum_plot = axes[1, 1].imshow(
    w1_power_db, cmap='viridis', vmin=power_vmin, vmax=power_vmax
)
axes[1, 1].set_title('W1 log-power spectrum')
fig.colorbar(w1_spectrum_plot, ax=axes[1, 1], fraction=0.046, pad=0.04, label='Power (dB)')

delta_plot = axes[1, 2].imshow(
    spectral_delta_db,
    cmap='coolwarm',
    vmin=-spectral_delta_limit,
    vmax=spectral_delta_limit,
)
axes[1, 2].set_title('Spectral change: W1 - original')
fig.colorbar(delta_plot, ax=axes[1, 2], fraction=0.046, pad=0.04, label='Power change (dB)')

axes[1, 3].plot(
    radial_frequency[valid_profile],
    original_radial_db[valid_profile],
    label='Original',
    linewidth=2.0,
)
axes[1, 3].plot(
    radial_frequency[valid_profile],
    w1_radial_db[valid_profile],
    label='W1',
    linewidth=2.0,
)
axes[1, 3].set_title('Radial mean spectral power')
axes[1, 3].set_xlabel('Normalized radial frequency')
axes[1, 3].set_ylabel('Mean power (dB)')
axes[1, 3].set_xlim(0.0, 1.0)
axes[1, 3].grid(alpha=0.25)
axes[1, 3].legend()

for axis in axes.ravel()[:7]:
    axis.set_xticks([])
    axis.set_yticks([])

class_description = ', '.join(class_names[class_id] for class_id in class_ids_for_image(defense_image_path))
fig.suptitle(
    f'W1 spatial and frequency-domain effect | {class_description} | '
    f'sigma={DEFENSE_W1_SIGMA}, alpha={DEFENSE_W1_ALPHA}',
    fontsize=16,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])

defense_png = OUTPUT_DIR / 'w1_defense_spatial_frequency_comparison.png'
defense_pdf = OUTPUT_DIR / 'w1_defense_spatial_frequency_comparison.pdf'
fig.savefig(defense_png, dpi=220, bbox_inches='tight')
fig.savefig(defense_pdf, bbox_inches='tight')
plt.show()

print('Defense image:', defense_image_path)
print('W1 theoretical gain range:', float(frequency_gain.min()), 'to', float(frequency_gain.max()))
print('Saved defense PNG:', defense_png)
print('Saved defense PDF:', defense_pdf)
print('Saved band-energy CSV:', band_csv)
print('Saved radial-profile CSV:', radial_csv)
display(band_energy_df)
'''

cells.extend([markdown_cell(defense_markdown), code_cell(defense_code)])

# Outputs from the old independently normalized implementation must not remain embedded.
for cell in cells:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook["cells"] = cells
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Updated {NOTEBOOK_PATH}")
