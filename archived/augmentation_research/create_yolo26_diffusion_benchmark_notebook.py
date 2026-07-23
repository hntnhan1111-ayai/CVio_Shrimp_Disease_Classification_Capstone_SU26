import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("run-yolo26-diffusion-augmentation-ablation.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


cells = [
    md(
        """
# YOLO26 Diffusion Augmentation Ablation

This Kaggle-ready notebook runs the requested YOLO classification ablation on `yolo26n-cls` and `yolo26m-cls`:

- RandAugment
- RandAugment + class-sized weighted augmentation
- AugMix
- AugMix + class-sized weighted augmentation
- Diffusion augmentation
- Diffusion augmentation + class-sized weighted augmentation

Replication note from `diffusion_augmentation.pdf`: the faithful paper method is a segmentation augmentation pipeline. It uses SDXL inpainting with text prompts and spatial masks, then filters generated samples using a trained latent-space segmentation validator with IoU greater than 0.7. This shrimp YOLO notebook has class labels but no spatial masks or segmentation validator, so the diffusion path below is an approximation: low-strength image-to-image diffusion with class prompts, plus optional class-weighted synthetic oversampling for minority classes.
"""
    ),
    md("## 1. Install Dependencies"),
    code(
        """
import importlib.util
import subprocess
import sys

REQUIRED_PACKAGES = {
    "ultralytics>=8.3.0": "ultralytics",
    "diffusers>=0.30.0": "diffusers",
    "transformers>=4.42.0": "transformers",
    "accelerate>=0.31.0": "accelerate",
    "safetensors>=0.4.3": "safetensors",
}

missing = [pkg for pkg, module in REQUIRED_PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    print(f"Installing missing dependencies: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("All required dependencies are already installed.")
"""
    ),
    md("## 2. Configuration and Fixed Split"),
    code(
        """
import gc
import json
import math
import random
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from ultralytics import YOLO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMG_SIZE = 224
EPOCHS = 30
PATIENCE = 15
YOLO_BATCH_SIZE = 32
ENABLE_GPU_LOGGING = True

DATA_DIR = Path("/kaggle/input/datasets/uynnhy/processed-images/processed_images")
OUTPUT_DIR = Path("/kaggle/working/yolo26_diffusion_aug_ablation")
YOLO_DATA_ROOT = OUTPUT_DIR / "yolo_datasets"
YOLO_RUNS_DIR = OUTPUT_DIR / "yolo_runs"
REPORT_DIR = OUTPUT_DIR / "reports"

FORCE_REBUILD_DATASETS = False

# Diffusion defaults are deliberately conservative for label preservation and T4 practicality.
DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
HF_TOKEN = None  # Set to a Hugging Face token string if your selected model requires auth.
DIFFUSION_IMAGE_SIZE = 512
DIFFUSION_STRENGTH = 0.18
DIFFUSION_GUIDANCE_SCALE = 4.0
DIFFUSION_STEPS = 20

for directory in [YOLO_DATA_ROOT, YOLO_RUNS_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

CLASS_DIRS = ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]
CLASS_NAMES = ["Healthy", "BG", "WSSV", "WSSV_BG"]
CLASS_TO_IDX = {folder: idx for idx, folder in enumerate(CLASS_DIRS)}
NUM_CLASSES = len(CLASS_DIRS)

YOLO_MODELS = ["yolo26n-cls", "yolo26m-cls"]

RUN_CONFIGS = [
    {"key": "randaug", "name": "RandAugment", "auto_augment": "randaugment", "diffusion": False, "class_weighted": False},
    {"key": "randaug_class_weighted", "name": "RandAugment + Class-Weighted Aug", "auto_augment": "randaugment", "diffusion": False, "class_weighted": True},
    {"key": "augmix", "name": "AugMix", "auto_augment": "augmix", "diffusion": False, "class_weighted": False},
    {"key": "augmix_class_weighted", "name": "AugMix + Class-Weighted Aug", "auto_augment": "augmix", "diffusion": False, "class_weighted": True},
    {"key": "diffusion_aug", "name": "Diffusion Aug", "auto_augment": None, "diffusion": True, "class_weighted": False},
    {"key": "diffusion_aug_class_weighted", "name": "Diffusion Aug + Class-Weighted Aug", "auto_augment": None, "diffusion": True, "class_weighted": True},
]


def reset_random_state(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(".", "_").replace("+", "plus")


def count_params(model) -> float:
    return sum(param.numel() for param in model.parameters()) / 1e6


def gpu_status_text() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    text = f"torch CUDA memory allocated/reserved/max: {allocated:.1f}/{reserved:.1f}/{max_allocated:.1f} MB"

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            first_gpu = completed.stdout.strip().splitlines()[0]
            util, mem_used, mem_total, power, temp = [part.strip() for part in first_gpu.split(",")[:5]]
            text += f" | nvidia-smi util={util}% mem={mem_used}/{mem_total} MB power={power} W temp={temp} C"
    except Exception as exc:
        text += f" | nvidia-smi unavailable: {type(exc).__name__}: {exc}"

    return text


def print_gpu_status(label: str):
    if ENABLE_GPU_LOGGING:
        print(f"[GPU] {label}: {gpu_status_text()}")


print_gpu_status("startup")
"""
    ),
    md("## 3. Dataset Discovery"),
    code(
        """
def discover_processed_images(data_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in CLASS_DIRS:
        folder = data_dir / class_dir
        if not folder.exists():
            print(f"Warning: missing class folder: {folder}")
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                rows.append({"path": str(path), "class_dir": class_dir, "label": CLASS_TO_IDX[class_dir]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No images found under {data_dir}. Check the Kaggle input path.")
    return frame


df = discover_processed_images(DATA_DIR)
print(f"Loaded {len(df)} processed images from {DATA_DIR}")
display(df["class_dir"].value_counts().reindex(CLASS_DIRS).rename("count").to_frame())

train_df, tmp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED,
    shuffle=True,
)
val_df, test_df = train_test_split(
    tmp_df,
    test_size=0.50,
    stratify=tmp_df["label"],
    random_state=SEED,
    shuffle=True,
)

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{split_name}: {len(split_df)} images")
    print(split_df["class_dir"].value_counts().reindex(CLASS_DIRS).to_dict())

assert set(train_df["path"]).isdisjoint(set(val_df["path"]))
assert set(train_df["path"]).isdisjoint(set(test_df["path"]))
assert set(val_df["path"]).isdisjoint(set(test_df["path"]))
print("Image-level split overlap check passed.")
"""
    ),
    md("## 4. Offline Augmentation Builders"),
    code(
        """
def resize_crop_for_yolo(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = random.uniform(0.82, 1.0)
    aspect = random.uniform(0.90, 1.10)
    crop_area = width * height * scale
    crop_w = int(round((crop_area * aspect) ** 0.5))
    crop_h = int(round((crop_area / aspect) ** 0.5))
    crop_w = min(width, max(1, crop_w))
    crop_h = min(height, max(1, crop_h))
    left = random.randint(0, max(0, width - crop_w))
    top = random.randint(0, max(0, height - crop_h))
    image = image.crop((left, top, left + crop_w, top + crop_h))
    return image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)


def apply_class_weighted_pil_augment(image: Image.Image) -> Image.Image:
    image = resize_crop_for_yolo(image.convert("RGB"))
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    if random.random() < 0.25:
        image = ImageOps.autocontrast(image)
    angle = random.uniform(-12.0, 12.0)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(128, 128, 128))
    for enhancer_cls, low, high in [
        (ImageEnhance.Color, 0.85, 1.15),
        (ImageEnhance.Contrast, 0.85, 1.18),
        (ImageEnhance.Brightness, 0.88, 1.12),
        (ImageEnhance.Sharpness, 0.85, 1.20),
    ]:
        image = enhancer_cls(image).enhance(random.uniform(low, high))
    return image


def save_image_for_yolo(image: Image.Image, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(dst, quality=95)


def unique_image_name(src: Path, row_idx: int, prefix: str = "", suffix: str | None = None) -> str:
    suffix = suffix or src.suffix.lower() or ".jpg"
    return f"{prefix}{src.stem}_{row_idx:06d}{suffix}"


CLASS_PROMPTS = {
    "1. Healthy": "high quality close-up aquaculture inspection photo of a healthy shrimp, natural shell texture, clear body, realistic lighting",
    "2. BG": "high quality close-up aquaculture inspection photo of a shrimp with bacterial gill disease, darkened gill region, realistic shell texture",
    "3. WSSV": "high quality close-up aquaculture inspection photo of a shrimp with white spot syndrome virus, subtle white shell spots, realistic texture",
    "4. WSSV_BG": "high quality close-up aquaculture inspection photo of a shrimp with white spot syndrome and bacterial gill disease, subtle white spots and dark gill region, realistic texture",
}

NEGATIVE_PROMPT = "cartoon, illustration, drawing, text, watermark, logo, unrealistic, deformed, extra limbs, blurry, low quality, cropped body"
diffusion_pipe = None


def load_diffusion_pipeline():
    global diffusion_pipe
    if diffusion_pipe is not None:
        return diffusion_pipe

    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

    print(f"Loading diffusion model: {DIFFUSION_MODEL_ID}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype, "safety_checker": None, "requires_safety_checker": False}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    diffusion_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(DIFFUSION_MODEL_ID, **kwargs)
    diffusion_pipe.scheduler = DPMSolverMultistepScheduler.from_config(diffusion_pipe.scheduler.config)
    diffusion_pipe = diffusion_pipe.to(device)
    diffusion_pipe.enable_attention_slicing()
    if hasattr(diffusion_pipe, "enable_vae_slicing"):
        diffusion_pipe.enable_vae_slicing()
    return diffusion_pipe


def unload_diffusion_pipeline():
    global diffusion_pipe
    if diffusion_pipe is not None:
        del diffusion_pipe
        diffusion_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_diffusion_init_image(image: Image.Image) -> Image.Image:
    return ImageOps.fit(
        image.convert("RGB"),
        (DIFFUSION_IMAGE_SIZE, DIFFUSION_IMAGE_SIZE),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def apply_diffusion_augment(image: Image.Image, class_dir: str, seed: int) -> Image.Image:
    pipe = load_diffusion_pipeline()
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    result = pipe(
        prompt=CLASS_PROMPTS[class_dir],
        negative_prompt=NEGATIVE_PROMPT,
        image=prepare_diffusion_init_image(image),
        strength=DIFFUSION_STRENGTH,
        guidance_scale=DIFFUSION_GUIDANCE_SCALE,
        num_inference_steps=DIFFUSION_STEPS,
        generator=generator,
    ).images[0]
    return result.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BICUBIC)
"""
    ),
    md("## 5. YOLO Dataset Preparation"),
    code(
        """
def create_empty_yolo_dataset(dataset_dir: Path):
    for split_name in ["train", "val", "test"]:
        for class_dir in CLASS_DIRS:
            (dataset_dir / split_name / class_dir).mkdir(parents=True, exist_ok=True)


def copy_split_rows(split_df: pd.DataFrame, dataset_dir: Path, split_name: str):
    for row_idx, (_, row) in enumerate(tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copy {split_name}")):
        src = Path(row["path"])
        dst = dataset_dir / split_name / row["class_dir"] / unique_image_name(src, row_idx, suffix=src.suffix.lower() or ".jpg")
        shutil.copy2(src, dst)


def add_synthetic_row(row: pd.Series, dst: Path, use_diffusion: bool, seed: int):
    image = Image.open(row["path"]).convert("RGB")
    if use_diffusion:
        synthetic = apply_diffusion_augment(image, row["class_dir"], seed=seed)
    else:
        synthetic = apply_class_weighted_pil_augment(image)
    save_image_for_yolo(synthetic, dst)


def add_one_diffusion_variant_per_train_image(train_df: pd.DataFrame, dataset_dir: Path, counts: Counter):
    print("Adding one diffusion image-to-image variant per training image.")
    for row_idx, (_, row) in enumerate(tqdm(train_df.iterrows(), total=len(train_df), desc="Diffusion base synth")):
        src = Path(row["path"])
        dst = dataset_dir / "train" / row["class_dir"] / unique_image_name(src, row_idx, prefix="diff_", suffix=".jpg")
        add_synthetic_row(row, dst, use_diffusion=True, seed=SEED + 10_000 + row_idx)
        counts[row["class_dir"]] += 1


def add_class_weighted_variants(train_df: pd.DataFrame, dataset_dir: Path, counts: Counter, use_diffusion: bool):
    target_count = max(counts[class_dir] for class_dir in CLASS_DIRS)
    rng = np.random.default_rng(SEED)
    print(f"Balancing train folders to {target_count} images per class using {'diffusion' if use_diffusion else 'PIL'} augmentation.")

    for class_dir in CLASS_DIRS:
        needed = target_count - counts[class_dir]
        if needed <= 0:
            continue
        class_rows = train_df[train_df["class_dir"] == class_dir].reset_index(drop=True)
        sampled_indices = rng.integers(0, len(class_rows), size=needed)
        for extra_idx, source_idx in enumerate(tqdm(sampled_indices, desc=f"Weighted synth {class_dir}")):
            row = class_rows.iloc[int(source_idx)]
            src = Path(row["path"])
            prefix = "diff_weighted_" if use_diffusion else "weighted_"
            dst = dataset_dir / "train" / class_dir / unique_image_name(src, extra_idx, prefix=prefix, suffix=".jpg")
            add_synthetic_row(row, dst, use_diffusion=use_diffusion, seed=SEED + 20_000 + extra_idx + CLASS_TO_IDX[class_dir] * 100_000)
            counts[class_dir] += 1


def count_yolo_train_images(dataset_dir: Path) -> dict:
    counts = {}
    for class_dir in CLASS_DIRS:
        folder = dataset_dir / "train" / class_dir
        counts[class_dir] = len([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    return counts


def prepare_yolo_dataset_for_config(config: dict) -> Path:
    dataset_dir = YOLO_DATA_ROOT / config["key"]
    if dataset_dir.exists() and not FORCE_REBUILD_DATASETS:
        print(f"Using cached dataset for {config['name']}: {dataset_dir}")
        print("Train counts:", count_yolo_train_images(dataset_dir))
        return dataset_dir

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    create_empty_yolo_dataset(dataset_dir)

    reset_random_state(SEED)
    copy_split_rows(train_df, dataset_dir, "train")
    copy_split_rows(val_df, dataset_dir, "val")
    copy_split_rows(test_df, dataset_dir, "test")

    counts = Counter(train_df["class_dir"].tolist())

    if config["diffusion"]:
        add_one_diffusion_variant_per_train_image(train_df, dataset_dir, counts)

    if config["class_weighted"]:
        add_class_weighted_variants(train_df, dataset_dir, counts, use_diffusion=config["diffusion"])

    if config["diffusion"]:
        unload_diffusion_pipeline()

    final_counts = count_yolo_train_images(dataset_dir)
    print(f"Prepared YOLO dataset for {config['name']}: {dataset_dir}")
    print("Train counts:", final_counts)
    return dataset_dir
"""
    ),
    md("## 6. YOLO Training and Evaluation"),
    code(
        """
def yolo_device_arg():
    return 0 if torch.cuda.is_available() else "cpu"


def evaluate_yolo_model(yolo_model, eval_df: pd.DataFrame, timed=False):
    names = yolo_model.names
    name_to_idx = {value: int(key) for key, value in names.items()}
    source_paths = eval_df["path"].tolist()

    if timed:
        _ = yolo_model.predict(source=source_paths[:1], imgsz=IMG_SIZE, device=yolo_device_arg(), verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
    else:
        start = None

    preds = yolo_model.predict(
        source=source_paths,
        imgsz=IMG_SIZE,
        batch=YOLO_BATCH_SIZE,
        device=yolo_device_arg(),
        verbose=False,
    )

    if timed and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start if timed else None

    y_true = [name_to_idx[class_dir] for class_dir in eval_df["class_dir"].tolist()]
    y_pred = [int(result.probs.top1) for result in preds]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "elapsed": elapsed,
        "labels": y_true,
        "preds": y_pred,
    }


YOLO_AUGMENTATION_BASE_ARGS = {
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "crop_fraction": 1.0,
}


def yolo_augmentation_args(config: dict) -> dict:
    args = YOLO_AUGMENTATION_BASE_ARGS.copy()
    args["auto_augment"] = config["auto_augment"]
    return args


def train_yolo_model(model_name: str, config: dict, dataset_dir: Path) -> dict:
    augmentation_name = config["name"]
    print("\\n" + "=" * 90)
    print(f"Training YOLO classifier: {model_name} | Augmentation: {augmentation_name}")
    print("=" * 90)

    weights_name = f"{model_name}.pt"
    run_name = f"{config['key']}_{sanitize_name(model_name)}"
    train_start = time.time()
    yolo = YOLO(weights_name)
    print_gpu_status(f"{augmentation_name} / {model_name} before YOLO train")
    yolo.train(
        data=str(dataset_dir),
        task="classify",
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=YOLO_BATCH_SIZE,
        patience=PATIENCE,
        seed=SEED,
        project=str(YOLO_RUNS_DIR),
        name=run_name,
        exist_ok=True,
        device=yolo_device_arg(),
        verbose=True,
        **yolo_augmentation_args(config),
    )
    print_gpu_status(f"{augmentation_name} / {model_name} after YOLO train")
    train_time = time.time() - train_start

    best_path = YOLO_RUNS_DIR / run_name / "weights" / "best.pt"
    if not best_path.exists():
        candidates = sorted((YOLO_RUNS_DIR / run_name).glob("**/best.pt"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate YOLO best checkpoint for {model_name} / {augmentation_name}")
        best_path = candidates[-1]

    best_yolo = YOLO(str(best_path))
    print_gpu_status(f"{augmentation_name} / {model_name} before YOLO eval")
    val_metrics = evaluate_yolo_model(best_yolo, val_df, timed=False)
    test_metrics = evaluate_yolo_model(best_yolo, test_df, timed=True)
    print_gpu_status(f"{augmentation_name} / {model_name} after YOLO eval")
    inf_time = test_metrics["elapsed"]
    params_m = count_params(best_yolo.model)

    result = {
        "Augmentation": augmentation_name,
        "Augmentation Key": config["key"],
        "Model": model_name,
        "Backend Name": weights_name,
        "Parameters (M)": round(params_m, 2),
        "Training Time (s)": round(train_time, 1),
        "Val F1-Score": round(val_metrics["macro_f1"], 4),
        "Test Accuracy": round(test_metrics["accuracy"], 4),
        "Test F1-Score": round(test_metrics["macro_f1"], 4),
        "Cohen Kappa": round(test_metrics["cohen_kappa"], 4),
        "Inference Time (s)": round(inf_time, 2),
        "FPS": round(len(test_df) / inf_time, 1) if inf_time else np.nan,
        "Latency (ms)": round((inf_time / len(test_df)) * 1000, 2) if inf_time else np.nan,
        "Train Counts": json.dumps(count_yolo_train_images(dataset_dir)),
        "Checkpoint Path": str(best_path),
        "Export Note": (
            f"auto_augment={config['auto_augment']}; "
            f"diffusion={config['diffusion']}; "
            f"class_weighted={config['class_weighted']}; "
            "YOLO augmentation knobs otherwise disabled."
        ),
    }

    del yolo, best_yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result
"""
    ),
    md("## 7. Sequential Benchmark Loop"),
    code(
        """
comparison_results = []

for config in RUN_CONFIGS:
    print("\\n" + "#" * 90)
    print(f"Starting YOLO augmentation benchmark: {config['name']}")
    print("#" * 90)
    dataset_dir = prepare_yolo_dataset_for_config(config)

    for model_name in YOLO_MODELS:
        try:
            reset_random_state(SEED)
            result = train_yolo_model(model_name, config, dataset_dir)
            comparison_results.append(result)
            print("Recorded result:")
            display(pd.DataFrame([result]))
        except Exception as exc:
            print(f"ERROR while running {config['name']} / {model_name}: {type(exc).__name__}: {exc}")
            comparison_results.append(
                {
                    "Augmentation": config["name"],
                    "Augmentation Key": config["key"],
                    "Model": model_name,
                    "Backend Name": "",
                    "Parameters (M)": np.nan,
                    "Training Time (s)": np.nan,
                    "Val F1-Score": np.nan,
                    "Test Accuracy": np.nan,
                    "Test F1-Score": np.nan,
                    "Cohen Kappa": np.nan,
                    "Inference Time (s)": np.nan,
                    "FPS": np.nan,
                    "Latency (ms)": np.nan,
                    "Train Counts": json.dumps(count_yolo_train_images(dataset_dir)) if dataset_dir.exists() else "{}",
                    "Checkpoint Path": "",
                    "Export Note": f"Run failed: {type(exc).__name__}: {exc}",
                }
            )

        partial_df = pd.DataFrame(comparison_results)
        partial_df.to_csv(REPORT_DIR / "yolo26_diffusion_aug_ablation_partial.csv", index=False)
        partial_df.to_json(REPORT_DIR / "yolo26_diffusion_aug_ablation_partial.json", orient="records", indent=2)
"""
    ),
    md("## 8. Combined Metrics"),
    code(
        """
df_summary = pd.DataFrame(comparison_results)
metric_columns = [
    "Augmentation",
    "Model",
    "Parameters (M)",
    "Training Time (s)",
    "Val F1-Score",
    "Test Accuracy",
    "Test F1-Score",
    "Cohen Kappa",
    "Inference Time (s)",
    "FPS",
    "Latency (ms)",
    "Train Counts",
]

if not df_summary.empty:
    run_order = {config["key"]: idx for idx, config in enumerate(RUN_CONFIGS)}
    model_order = {model: idx for idx, model in enumerate(YOLO_MODELS)}
    df_summary["Run Order"] = df_summary["Augmentation Key"].map(run_order)
    df_summary["Model Order"] = df_summary["Model"].map(model_order)
    df_summary = df_summary.sort_values(["Run Order", "Model Order"], na_position="last").reset_index(drop=True)

    print("\\n" + "=" * 90)
    print("YOLO26 DIFFUSION AUGMENTATION ABLATION: SHRIMP DISEASE CLASSIFICATION")
    print("=" * 90)
    display(df_summary[metric_columns])

    summary_csv = REPORT_DIR / "yolo26_diffusion_aug_ablation_summary.csv"
    summary_json = REPORT_DIR / "yolo26_diffusion_aug_ablation_summary.json"
    metrics_csv = REPORT_DIR / "yolo26_diffusion_aug_ablation_metrics_table.csv"
    df_summary.to_csv(summary_csv, index=False)
    df_summary.to_json(summary_json, orient="records", indent=2)
    df_summary[metric_columns].to_csv(metrics_csv, index=False)
    print(f"Saved full summary CSV: {summary_csv}")
    print(f"Saved full summary JSON: {summary_json}")
    print(f"Saved metrics-only CSV: {metrics_csv}")

    print("\\nScore-sorted view:")
    display(df_summary.sort_values(by="Test F1-Score", ascending=False, na_position="last")[metric_columns])
else:
    print("No model results were produced.")
"""
    ),
    md("## 9. Optional Archive Outputs"),
    code(
        """
zip_base = "/kaggle/working/yolo26_diffusion_aug_ablation_yolo_runs"
if YOLO_RUNS_DIR.exists():
    archive_path = shutil.make_archive(zip_base, "zip", YOLO_RUNS_DIR)
    print(f"Zipped YOLO runs at: {archive_path}")
else:
    print("No YOLO runs directory found to zip.")
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
