# %% [markdown]
# # Shrimp Disease Full Lightweight Model Comparison
#
# Trains the requested lightweight classifiers and reports the same comparison
# metrics as the baseline table, plus Cohen Kappa. Training/evaluation is the
# priority; export and LiteRT/TFLite conversion are optional sections at the end.

# %%
from google.colab import drive
drive.mount("/content/drive")

# %% [markdown]
# ## 1. Install dependencies

# %%
import importlib.util
import subprocess
import sys

packages = [
    "timm>=1.0.0",
    "torchmetrics",
    "ultralytics>=8.3.0",
    "onnx",
    "onnxscript",
]


def import_name_for(package_spec: str) -> str:
    package = package_spec.split(">=")[0].split("==")[0].split("<")[0]
    return {
        "ultralytics": "ultralytics",
        "torchmetrics": "torchmetrics",
        "onnxscript": "onnxscript",
    }.get(package, package.replace("-", "_"))


missing = [pkg for pkg in packages if importlib.util.find_spec(import_name_for(pkg)) is None]
if missing:
    print("Installing missing dependencies:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *missing])
else:
    print("All training dependencies are already available.")

# %% [markdown]
# ## 2. Configuration and fixed split

# %%
import gc
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from ultralytics import YOLO

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMG_SIZE = 224
EPOCHS = 30
PATIENCE = 5
PAPER_BATCH_SIZE = 128
MICRO_BATCH_SIZE = 32
ACCUMULATION_STEPS = max(1, PAPER_BATCH_SIZE // MICRO_BATCH_SIZE)
EVAL_BATCH_SIZE = PAPER_BATCH_SIZE
YOLO_BATCH_SIZE = MICRO_BATCH_SIZE
WARMUP_EPOCHS = 5
WARMUP_HEAD_LR = 1e-3
BACKBONE_FINETUNE_LR = 2e-5
HEAD_FINETUNE_LR = 1e-4
STEP_SIZE = 3
STEP_GAMMA = 0.9
NUM_WORKERS = 2

DATA_DIR = Path("/content/drive/MyDrive/shrimp_disease_images/Processed_Rembg_Images")
OUTPUT_DIR = Path("/content/drive/MyDrive/shrimp_disease_images/full_lightweight_model_comparison")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EXPORT_DIR = OUTPUT_DIR / "exports"
YOLO_DATA_DIR = OUTPUT_DIR / "yolo_dataset"
YOLO_RUNS_DIR = OUTPUT_DIR / "yolo_runs"
REPORT_DIR = OUTPUT_DIR / "reports"
CONVERSION_ROOT = OUTPUT_DIR / "litert_conversions"

for directory in [CHECKPOINT_DIR, EXPORT_DIR, YOLO_DATA_DIR, YOLO_RUNS_DIR, REPORT_DIR, CONVERSION_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

CLASS_DIRS = ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]
CLASS_NAMES = ["Healthy", "BG", "WSSV", "WSSV_BG"]
CLASS_TO_IDX = {folder: idx for idx, folder in enumerate(CLASS_DIRS)}
NUM_CLASSES = len(CLASS_DIRS)

TIMM_MODELS = [
    "convnext_tiny_in22k",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "repvgg_a0",
    "efficientnet_v2_s",
    "shufflenet_v2_x1_0",
    "fastvit_t8",
    "edgenext_xx_small",
    "mobileone_s0",
    "mobilevit_s",
    "mobilenetv4_conv_small",
    "mnasnet_100",
    "ghostnetv2_100",
    "rexnet_100",
    "squeezenet1_1",
    "mobilenetv4_hybrid_medium",
    "efficientvit_m1",
]
YOLO_MODELS = ["yolo26n-cls", "yolo26m-cls", "yolo11n-cls", "yolo11m-cls"]
ALL_MODELS = TIMM_MODELS + YOLO_MODELS

# Keep evaluation as the priority. Set True after a successful run if you want
# export to happen automatically in the same execution.
RUN_EXPORT_AFTER_TRAINING = False

# %%
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
        raise RuntimeError(f"No images found under {data_dir}. Run preprocessing from the baseline first.")
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

# %% [markdown]
# ## 3. TIMM data pipeline and helpers

# %%
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.RandomResizedCrop(
        IMG_SIZE,
        scale=(0.82, 1.0),
        ratio=(0.90, 1.10),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BICUBIC),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    # Matches torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms().
    transforms.Resize(236, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ShrimpFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


train_loader = DataLoader(
    ShrimpFrameDataset(train_df, train_transform),
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
val_loader = DataLoader(
    ShrimpFrameDataset(val_df, eval_transform),
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
test_loader = DataLoader(
    ShrimpFrameDataset(test_df, eval_transform),
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

# %%
TIMM_ALIASES = {
    "convnext_tiny_in22k": ["convnext_tiny.fb_in22k", "convnext_tiny.fb_in1k", "convnext_tiny"],
    "mobilenet_v3_large": ["mobilenetv3_large_100.ra_in1k", "mobilenetv3_large_100", "tf_mobilenetv3_large_100"],
    "efficientnet_b0": ["efficientnet_b0.ra_in1k", "tf_efficientnet_b0.ns_jft_in1k", "efficientnet_b0"],
    "repvgg_a0": ["repvgg_a0.rvgg_in1k", "repvgg_a0"],
    "efficientnet_v2_s": ["tf_efficientnetv2_s.in21k_ft_in1k", "tf_efficientnetv2_s", "efficientnetv2_rw_s.ra2_in1k"],
    "shufflenet_v2_x1_0": ["shufflenet_v2_x1_0"],
    "fastvit_t8": ["fastvit_t8.apple_in1k", "fastvit_t8"],
    "edgenext_xx_small": ["edgenext_xx_small.in1k", "edgenext_xx_small"],
    "mobileone_s0": ["mobileone_s0.apple_in1k", "mobileone_s0"],
    "mobilevit_s": ["mobilevit_s.cvnets_in1k", "mobilevit_s"],
    "mobilenetv4_conv_small": ["mobilenetv4_conv_small.e2400_r224_in1k", "mobilenetv4_conv_small"],
    "mnasnet_100": ["mnasnet_100.rmsp_in1k", "mnasnet_100"],
    "ghostnetv2_100": ["ghostnetv2_100.in1k", "ghostnetv2_100"],
    "rexnet_100": ["rexnet_100.nav_in1k", "rexnet_100"],
    "squeezenet1_1": ["squeezenet1_1"],
    "mobilenetv4_hybrid_medium": ["mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k", "mobilenetv4_hybrid_medium"],
    "efficientvit_m1": ["efficientvit_m1.r224_in1k", "efficientvit_m1"],
}


def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace(".", "_")


def count_params(model) -> float:
    return sum(param.numel() for param in model.parameters()) / 1e6


def resolve_timm_name(display_name: str) -> str:
    candidates = TIMM_ALIASES.get(display_name, [display_name])
    available = set(timm.list_models(pretrained=False))
    for candidate in candidates:
        if candidate in available:
            return candidate
    pattern_hits = []
    for candidate in candidates:
        pattern_hits.extend(timm.list_models(candidate + "*", pretrained=False))
    if pattern_hits:
        return sorted(pattern_hits)[0]
    raise ValueError(f"No TIMM model found for {display_name}. Tried: {candidates}")


class TimmShrimpXNet(nn.Module):
    def __init__(self, timm_name: str, pretrained: bool):
        super().__init__()
        try:
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        except TypeError:
            self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)

        # TIMM's num_features can describe the pre-classifier channel count,
        # while num_classes=0 can return a post-head embedding for some models
        # such as MobileNetV3. Infer the actual forward output to avoid head
        # shape mismatches.
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            features = self.backbone(sample)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            num_features = features.flatten(1).shape[1]
        self.backbone.train(was_training)

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
        x = torch.flatten(x, 1)
        return self.classifier(x)


def create_timm_classifier(display_name: str):
    timm_name = resolve_timm_name(display_name)
    try:
        model = TimmShrimpXNet(timm_name, pretrained=True)
        pretrained = True
    except Exception as pretrained_error:
        print(
            f"Pretrained weights failed for {display_name} ({timm_name}): "
            f"{type(pretrained_error).__name__}: {pretrained_error}"
        )
        model = TimmShrimpXNet(timm_name, pretrained=False)
        pretrained = False
    return model.to(device), timm_name, pretrained


def classifier_parameters(model):
    if hasattr(model, "classifier"):
        return list(model.classifier.parameters())

    params = []
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        params = list(classifier.parameters())
    elif isinstance(classifier, (list, tuple, nn.ModuleList)):
        for module in classifier:
            if isinstance(module, nn.Module):
                params.extend(list(module.parameters()))

    if not params:
        head_tokens = ("classifier", "head", "fc")
        params = [
            param
            for name, param in model.named_parameters()
            if any(token in name.lower() for token in head_tokens)
        ]

    if not params:
        raise RuntimeError("Could not identify classifier/head parameters for warmup.")
    return params


def freeze_backbone_for_warmup(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in classifier_parameters(model):
        param.requires_grad = True


def make_warmup_optimizer(model):
    return optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=WARMUP_HEAD_LR,
    )


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True


def unfreeze_final_backbone_portion(model, display_name: str):
    backbone = model.backbone if hasattr(model, "backbone") else model
    trainable_modules = []

    for param in model.parameters():
        param.requires_grad = False

    for param in classifier_parameters(model):
        param.requires_grad = True

    if hasattr(backbone, "features") and isinstance(backbone.features, (nn.Sequential, nn.ModuleList, list, tuple)):
        features = backbone.features
        if "convnext" in display_name.lower() and len(features) > 5:
            selected = list(features[5:])
        else:
            start = max(0, len(features) - max(1, len(features) // 3))
            selected = list(features[start:])
        trainable_modules.extend(selected)

    elif hasattr(backbone, "stages") and isinstance(backbone.stages, (nn.Sequential, nn.ModuleList, list, tuple)):
        stages = backbone.stages
        start = max(0, len(stages) - max(1, len(stages) // 3))
        trainable_modules.extend(list(stages[start:]))

    elif hasattr(backbone, "blocks") and isinstance(backbone.blocks, (nn.Sequential, nn.ModuleList, list, tuple)):
        blocks = backbone.blocks
        start = max(0, len(blocks) - max(1, len(blocks) // 3))
        trainable_modules.extend(list(blocks[start:]))

    else:
        excluded = {"classifier", "head", "fc", "global_pool", "pool", "avgpool"}
        children = [
            child for name, child in backbone.named_children()
            if name not in excluded and not name.startswith("head")
        ]
        trainable_modules.extend(children[-2:] if len(children) >= 2 else children)

    for attr in ["norm", "norm_head", "head_norm", "pre_head", "final_conv"]:
        module = getattr(backbone, attr, None)
        if isinstance(module, nn.Module):
            trainable_modules.append(module)

    for module in trainable_modules:
        unfreeze_module(module)

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def make_finetune_optimizer(model):
    head_param_ids = {id(param) for param in classifier_parameters(model)}
    backbone_params = []
    head_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in head_param_ids:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": BACKBONE_FINETUNE_LR})
    if head_params:
        param_groups.append({"params": head_params, "lr": HEAD_FINETUNE_LR})
    return optim.Adam(param_groups)


def predict_pytorch(model, loader, criterion=None, timed=False):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []

    if timed:
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        with torch.no_grad():
            for _ in range(5):
                model(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        start = time.time()
    else:
        start = None

    with torch.no_grad():
        for ims, gts in loader:
            ims = ims.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            logits = model(ims)
            if criterion is not None:
                total_loss += criterion(logits, gts).item() * ims.size(0)
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(gts.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    if timed and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start if timed else None

    return {
        "loss": total_loss / max(1, len(all_labels)) if criterion is not None else None,
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(all_labels, all_preds),
        "elapsed": elapsed,
        "labels": all_labels,
        "preds": all_preds,
    }

# %% [markdown]
# ## 4. Train TIMM models

# %%
def train_timm_model(model_name: str) -> dict:
    print("\n" + "=" * 90)
    print(f"Training TIMM classifier: {model_name}")
    print("=" * 90)

    model, timm_name, pretrained = create_timm_classifier(model_name)
    print(f"Resolved TIMM model: {timm_name} | pretrained={pretrained}")
    criterion = nn.CrossEntropyLoss()
    safe_name = sanitize_name(model_name)
    best_path = CHECKPOINT_DIR / f"best_{safe_name}.pth"
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    epochs_no_improve = 0
    train_start = time.time()

    freeze_backbone_for_warmup(model)
    print(f"Warmup trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = make_warmup_optimizer(model)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA)

    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS:
            print("\n--- Switching to fine-tuning phase: unfreezing backbone with lower LR ---")
            trainable_count = unfreeze_final_backbone_portion(model, model_name)
            print(f"Fine-tune trainable parameters: {trainable_count:,}")
            optimizer = make_finetune_optimizer(model)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=STEP_GAMMA)
            epochs_no_improve = 0

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.zero_grad(set_to_none=True)

        for step, (ims, gts) in enumerate(tqdm(train_loader, desc=f"{model_name} epoch {epoch + 1}/{EPOCHS}", leave=False)):
            ims = ims.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)

            logits = model(ims)
            loss = criterion(logits, gts)
            (loss / ACCUMULATION_STEPS).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * ims.size(0)
            train_correct += (torch.argmax(logits, dim=1) == gts).sum().item()
            train_total += gts.size(0)

        scheduler.step()
        val_metrics = predict_pytorch(model, val_loader, criterion=criterion, timed=False)
        train_acc = train_correct / max(1, train_total)
        phase = "warmup" if epoch < WARMUP_EPOCHS else "finetune"
        lr_text = ",".join(f"{lr:.2e}" for lr in current_lrs)
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Phase: {phase} | LR: {lr_text} | "
            f"Train Loss: {train_loss / max(1, train_total):.4f} - Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} - Acc: {val_metrics['accuracy']:.4f} - Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["macro_f1"]
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "display_name": model_name,
                    "timm_name": timm_name,
                    "pretrained": pretrained,
                    "num_classes": NUM_CLASSES,
                    "img_size": IMG_SIZE,
                    "class_names": CLASS_NAMES,
                },
                best_path,
            )
            print(f"  --> Saved best checkpoint: val loss {best_val_loss:.4f}, macro F1 {best_val_f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  --> No improvement ({epochs_no_improve}/{PATIENCE})")

        if epochs_no_improve >= PATIENCE:
            print("  --> Early stopping triggered.")
            break

    train_time = time.time() - train_start
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = predict_pytorch(model, test_loader, criterion=None, timed=True)
    inf_time = test_metrics["elapsed"]
    params_m = count_params(model)

    result = {
        "Model": model_name,
        "Backend Name": timm_name,
        "Parameters (M)": round(params_m, 2),
        "Training Time (s)": round(train_time, 1),
        "Val F1-Score": round(best_val_f1, 4),
        "Best Val Loss": round(best_val_loss, 4),
        "Test Accuracy": round(test_metrics["accuracy"], 4),
        "Test F1-Score": round(test_metrics["macro_f1"], 4),
        "Cohen Kappa": round(test_metrics["cohen_kappa"], 4),
        "Inference Time (s)": round(inf_time, 2),
        "FPS": round(len(test_df) / inf_time, 1),
        "Latency (ms)": round((inf_time / len(test_df)) * 1000, 2),
        "Checkpoint Path": str(best_path),
        "Export Format": "",
        "Export Path": "",
        "Export Note": "",
    }

    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result

# %% [markdown]
# ## 5. YOLO dataset, training, and evaluation

# %%
def prepare_yolo_dataset():
    if YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for class_dir in CLASS_DIRS:
            (YOLO_DATA_DIR / split_name / class_dir).mkdir(parents=True, exist_ok=True)
        for _, row in split_df.iterrows():
            src = Path(row["path"])
            dst = YOLO_DATA_DIR / split_name / row["class_dir"] / src.name
            if dst.exists():
                dst = dst.with_name(f"{dst.stem}_{abs(hash(str(src))) % 10_000_000}{dst.suffix}")
            shutil.copy2(src, dst)
    print(f"Prepared YOLO classification dataset at {YOLO_DATA_DIR}")


prepare_yolo_dataset()


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


def train_yolo_model(model_name: str) -> dict:
    print("\n" + "=" * 90)
    print(f"Training YOLO classifier: {model_name}")
    print("=" * 90)

    weights_name = f"{model_name}.pt"
    train_start = time.time()
    yolo = YOLO(weights_name)
    yolo.train(
        data=str(YOLO_DATA_DIR),
        task="classify",
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=YOLO_BATCH_SIZE,
        patience=PATIENCE,
        seed=SEED,
        project=str(YOLO_RUNS_DIR),
        name=sanitize_name(model_name),
        exist_ok=True,
        device=yolo_device_arg(),
        verbose=True,
    )
    train_time = time.time() - train_start

    best_path = YOLO_RUNS_DIR / sanitize_name(model_name) / "weights" / "best.pt"
    if not best_path.exists():
        candidates = sorted((YOLO_RUNS_DIR / sanitize_name(model_name)).glob("**/best.pt"))
        if not candidates:
            raise FileNotFoundError(f"Could not locate YOLO best checkpoint for {model_name}")
        best_path = candidates[-1]

    best_yolo = YOLO(str(best_path))
    val_metrics = evaluate_yolo_model(best_yolo, val_df, timed=False)
    test_metrics = evaluate_yolo_model(best_yolo, test_df, timed=True)
    inf_time = test_metrics["elapsed"]
    params_m = count_params(best_yolo.model)

    result = {
        "Model": model_name,
        "Backend Name": weights_name,
        "Parameters (M)": round(params_m, 2),
        "Training Time (s)": round(train_time, 1),
        "Val F1-Score": round(val_metrics["macro_f1"], 4),
        "Test Accuracy": round(test_metrics["accuracy"], 4),
        "Test F1-Score": round(test_metrics["macro_f1"], 4),
        "Cohen Kappa": round(test_metrics["cohen_kappa"], 4),
        "Inference Time (s)": round(inf_time, 2),
        "FPS": round(len(test_df) / inf_time, 1),
        "Latency (ms)": round((inf_time / len(test_df)) * 1000, 2),
        "Checkpoint Path": str(best_path),
        "Export Format": "",
        "Export Path": "",
        "Export Note": "",
    }

    del yolo, best_yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return result

# %% [markdown]
# ## 6. Sequential comparison loop
#
# Partial CSV/JSON files are saved after every model.

# %%
comparison_results = []

for model_name in ALL_MODELS:
    try:
        if model_name in TIMM_MODELS:
            result = train_timm_model(model_name)
        else:
            result = train_yolo_model(model_name)
        comparison_results.append(result)
        print("Recorded result:")
        display(pd.DataFrame([result]))
    except Exception as exc:
        print(f"ERROR while running {model_name}: {type(exc).__name__}: {exc}")
        comparison_results.append(
            {
                "Model": model_name,
                "Backend Name": "",
                "Parameters (M)": np.nan,
                "Training Time (s)": np.nan,
                "Val F1-Score": np.nan,
                "Best Val Loss": np.nan,
                "Test Accuracy": np.nan,
                "Test F1-Score": np.nan,
                "Cohen Kappa": np.nan,
                "Inference Time (s)": np.nan,
                "FPS": np.nan,
                "Latency (ms)": np.nan,
                "Checkpoint Path": "",
                "Export Format": "not_run",
                "Export Path": "",
                "Export Note": f"Run failed: {type(exc).__name__}: {exc}",
            }
        )

    partial_df = pd.DataFrame(comparison_results)
    partial_df.to_csv(REPORT_DIR / "full_model_comparison_partial.csv", index=False)
    partial_df.to_json(REPORT_DIR / "full_model_comparison_partial.json", orient="records", indent=2)

# %%
df_summary = pd.DataFrame(comparison_results)
metric_columns = [
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
]

if not df_summary.empty:
    df_summary = df_summary.sort_values(by="Test F1-Score", ascending=False, na_position="last").reset_index(drop=True)
    print("\n" + "=" * 90)
    print("FULL LIGHTWEIGHT MODEL PERFORMANCE SUMMARY: SHRIMP DISEASE CLASSIFICATION")
    print("=" * 90)
    display(df_summary[metric_columns])

    summary_csv = REPORT_DIR / "full_model_comparison_summary.csv"
    summary_json = REPORT_DIR / "full_model_comparison_summary.json"
    metrics_csv = REPORT_DIR / "full_model_comparison_metrics_table.csv"
    df_summary.to_csv(summary_csv, index=False)
    df_summary.to_json(summary_json, orient="records", indent=2)
    df_summary[metric_columns].to_csv(metrics_csv, index=False)
    print(f"Saved full summary CSV: {summary_csv}")
    print(f"Saved full summary JSON: {summary_json}")
    print(f"Saved metrics-only CSV: {metrics_csv}")
else:
    print("No model results were produced.")

# %% [markdown]
# ## 7. Optional ONNX/YOLO export
#
# PyTorch/TIMM checkpoints export to ONNX. YOLO checkpoints try TFLite first and
# ONNX as a fallback.

# %%
def rebuild_timm_from_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TimmShrimpXNet(checkpoint["timm_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


def export_timm_checkpoint_to_onnx(row: pd.Series):
    model_name = row["Model"]
    model, _ = rebuild_timm_from_checkpoint(row["Checkpoint Path"])
    onnx_path = EXPORT_DIR / f"{sanitize_name(model_name)}.onnx"
    sample = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    try:
        torch.onnx.export(
            model,
            sample,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            sample,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "onnx", str(onnx_path), ""


def export_yolo_checkpoint(row: pd.Series):
    yolo_model = YOLO(row["Checkpoint Path"])
    try:
        exported = yolo_model.export(format="tflite", imgsz=IMG_SIZE)
        return "tflite", str(exported), ""
    except Exception as tflite_error:
        try:
            exported = yolo_model.export(format="onnx", imgsz=IMG_SIZE)
            return "onnx", str(exported), f"TFLite failed: {type(tflite_error).__name__}: {tflite_error}"
        except Exception as onnx_error:
            return "failed", "", f"TFLite failed: {tflite_error}; ONNX failed: {onnx_error}"


def export_trained_models(results_df: pd.DataFrame):
    export_results = []
    for _, row in results_df.iterrows():
        model_name = row["Model"]
        checkpoint_path = row.get("Checkpoint Path", "")
        if not checkpoint_path or not Path(checkpoint_path).exists():
            export_results.append({"Model": model_name, "Export Format": "skipped", "Export Path": "", "Export Note": "Missing checkpoint"})
            continue

        print("Exporting:", model_name)
        start = time.time()
        try:
            if model_name in TIMM_MODELS:
                export_format, export_path, export_note = export_timm_checkpoint_to_onnx(row)
            else:
                export_format, export_path, export_note = export_yolo_checkpoint(row)
        except Exception as exc:
            export_format, export_path, export_note = "failed", "", f"{type(exc).__name__}: {exc}"

        export_results.append(
            {
                "Model": model_name,
                "Export Format": export_format,
                "Export Path": export_path,
                "Export Note": export_note,
                "Export Time (s)": round(time.time() - start, 1),
            }
        )
        export_df = pd.DataFrame(export_results)
        display(export_df.tail(1))
        export_df.to_csv(REPORT_DIR / "export_results_partial.csv", index=False)

    export_df = pd.DataFrame(export_results)
    export_df.to_csv(REPORT_DIR / "export_results.csv", index=False)
    export_df.to_json(REPORT_DIR / "export_results.json", orient="records", indent=2)
    return export_df


if RUN_EXPORT_AFTER_TRAINING:
    export_df = export_trained_models(df_summary)
    display(export_df)
else:
    print("RUN_EXPORT_AFTER_TRAINING=False. Set it to True, or run: export_df = export_trained_models(df_summary)")

# %% [markdown]
# ## 8. Optional ONNX to LiteRT/TFLite conversion
#
# Mirrors `onnx_to_litert_conversion_ver2.ipynb`: ONNX validation,
# ONNX -> TensorFlow through `onnx2tf`, then float32 and dynamic-range TFLite.

# %%
conversion_packages = [
    "onnxruntime-gpu",
    "onnxslim",
    "sng4onnx>=1.0.1",
    "onnx_graphsurgeon>=0.3.26",
    "onnx2tf>=1.26.3,<1.29.0",
    "tensorflow>=2.15",
    "ai-edge-litert",
]


def conversion_import_name(package_spec: str) -> str:
    package = package_spec.split(">=")[0].split("==")[0].split("<")[0]
    return {
        "onnxruntime-gpu": "onnxruntime",
        "onnx_graphsurgeon": "onnx_graphsurgeon",
        "ai-edge-litert": "ai_edge_litert",
    }.get(package, package.replace("-", "_"))


def install_conversion_dependencies():
    missing_conversion = [
        pkg for pkg in conversion_packages if importlib.util.find_spec(conversion_import_name(pkg)) is None
    ]
    if missing_conversion:
        print("Installing missing conversion packages:", missing_conversion)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *missing_conversion])
    else:
        print("All conversion dependencies are already available.")


def convert_exported_onnx_to_litert(onnx_input_dir: Path = EXPORT_DIR):
    install_conversion_dependencies()

    import onnx
    import tensorflow as tf

    saved_model_root = CONVERSION_ROOT / "saved_models"
    tflite_dir = CONVERSION_ROOT / "tflite"
    conversion_report_dir = CONVERSION_ROOT / "reports"
    for directory in [saved_model_root, tflite_dir, conversion_report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(onnx_input_dir.glob("*.onnx"))
    if not onnx_files:
        raise RuntimeError(f"No ONNX files found in {onnx_input_dir}. Run export_trained_models(df_summary) first.")

    def validate_onnx(onnx_path: Path):
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def run_onnx2tf(onnx_path: Path, output_dir: Path):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        cmd = ["onnx2tf", "-i", str(onnx_path), "-o", str(output_dir), "-cotof"]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "onnx2tf failed\nSTDOUT:\n"
                + completed.stdout[-4000:]
                + "\nSTDERR:\n"
                + completed.stderr[-4000:]
            )

    def convert_saved_model_to_tflite(saved_model_dir: Path, tflite_path: Path, quantize=False):
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        return tflite_path

    def smoke_test_tflite(tflite_path: Path):
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        return {
            "input_shape": input_details["shape"].tolist(),
            "input_dtype": str(input_details["dtype"]),
            "output_shape": output_details["shape"].tolist(),
            "output_dtype": str(output_details["dtype"]),
        }

    conversion_results = []
    for onnx_path in onnx_files:
        name = sanitize_name(onnx_path.stem)
        model_saved_root = saved_model_root / name
        float_tflite = tflite_dir / f"{name}_float32.tflite"
        dynamic_tflite = tflite_dir / f"{name}_dynamic_range.tflite"
        result = {
            "model": name,
            "onnx_path": str(onnx_path),
            "onnx_size_mb": round(onnx_path.stat().st_size / (1024**2), 2),
            "saved_model_path": "",
            "float32_tflite_path": "",
            "dynamic_tflite_path": "",
            "float32_size_mb": np.nan,
            "dynamic_size_mb": np.nan,
            "smoke_test": "",
            "status": "failed",
            "error": "",
            "seconds": np.nan,
        }
        start = time.time()
        print("Converting:", onnx_path.name)
        try:
            validate_onnx(onnx_path)
            run_onnx2tf(onnx_path, model_saved_root)
            result["saved_model_path"] = str(model_saved_root)
            convert_saved_model_to_tflite(model_saved_root, float_tflite, quantize=False)
            result["float32_tflite_path"] = str(float_tflite)
            result["float32_size_mb"] = round(float_tflite.stat().st_size / (1024**2), 2)
            result["smoke_test"] = json.dumps(smoke_test_tflite(float_tflite))
            try:
                convert_saved_model_to_tflite(model_saved_root, dynamic_tflite, quantize=True)
                result["dynamic_tflite_path"] = str(dynamic_tflite)
                result["dynamic_size_mb"] = round(dynamic_tflite.stat().st_size / (1024**2), 2)
            except Exception as quant_error:
                result["dynamic_tflite_path"] = ""
                result["error"] = f"Dynamic quantization failed: {type(quant_error).__name__}: {quant_error}"
            result["status"] = "converted"
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["seconds"] = round(time.time() - start, 1)
        conversion_results.append(result)
        conversion_df = pd.DataFrame(conversion_results)
        display(conversion_df.tail(1))
        conversion_df.to_csv(conversion_report_dir / "onnx_to_litert_conversion_report_partial.csv", index=False)

    conversion_df = pd.DataFrame(conversion_results)
    conversion_df.to_csv(conversion_report_dir / "onnx_to_litert_conversion_report.csv", index=False)
    conversion_df.to_json(conversion_report_dir / "onnx_to_litert_conversion_report.json", orient="records", indent=2)
    return conversion_df


print("To convert ONNX exports, run: conversion_df = convert_exported_onnx_to_litert(EXPORT_DIR)")
