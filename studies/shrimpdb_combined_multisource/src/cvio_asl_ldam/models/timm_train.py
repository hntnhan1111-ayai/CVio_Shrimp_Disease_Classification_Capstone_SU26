"""Portable TIMM baseline training entry point."""

from __future__ import annotations

import argparse
import copy
import csv
import os
import time
from pathlib import Path

from cvio_asl_ldam.data.audit_dataset import resolve_dataset_root
from cvio_asl_ldam.data.make_split import create_split
from cvio_asl_ldam.evaluation.confusion_matrix import save_confusion_matrix
from cvio_asl_ldam.evaluation.metrics import classification_metrics
from cvio_asl_ldam.utils.io import load_yaml, write_json
from cvio_asl_ldam.utils.paths import ProjectPaths, resolve_device
from cvio_asl_ldam.utils.seed import seed_everything


def _rows(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="convnext_tiny_in22k")
    parser.add_argument("--config", default="configs/dataset/shrimpdiseasebd_seed42.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    import timm
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    paths = ProjectPaths.from_environment()
    config = load_yaml(args.config)
    dataset_root, _ = resolve_dataset_root(args.data_dir or paths.data_dir, config)
    output_root = Path(args.output_dir or paths.output_dir)
    manifest = output_root / "manifests" / "split_manifest_seed42_generated.csv"
    if not manifest.is_file():
        manifest = create_split(args.config, dataset_root, manifest.parent, args.seed)
    seed_everything(args.seed)
    resolved_device = resolve_device(os.environ.get("DEVICE", "auto"))
    primary_device = resolved_device.split(",")[0]
    device = torch.device(
        f"cuda:{primary_device}" if primary_device.isdigit() else "cpu"
    )

    class FrameDataset(Dataset):
        def __init__(self, rows, transform):
            self.rows = rows
            self.transform = transform

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            image = Image.open(dataset_root / row["rel_path"]).convert("RGB")
            return self.transform(image), int(row["label"])

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.82, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_loader = DataLoader(
        FrameDataset(_rows(manifest, "train"), train_transform),
        batch_size=32,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FrameDataset(_rows(manifest, "val"), eval_transform),
        batch_size=128,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        FrameDataset(_rows(manifest, "test"), eval_transform),
        batch_size=128,
        shuffle=False,
        num_workers=args.workers,
    )

    aliases = {
        "convnext_tiny_in22k": "convnext_tiny.fb_in22k",
        "mobilenet_v3_large": "mobilenetv3_large_100",
        "efficientnet_v2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    }
    model = timm.create_model(aliases.get(args.model, args.model), pretrained=True, num_classes=4)
    model.to(device)
    if resolved_device == "0,1":
        model = nn.DataParallel(model, device_ids=[0, 1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    def evaluate(loader):
        model.eval()
        true, predicted = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    logits = model(images)
                true.extend(labels.tolist())
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
        return classification_metrics(true, predicted)

    history = []
    best_state = None
    best_f1 = -1.0
    epochs = 1 if args.smoke_test else args.epochs
    started = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu()) * labels.shape[0]
        metrics = evaluate(val_loader)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": running_loss / len(train_loader.dataset),
                "val_macro_f1": metrics["macro_f1"],
            }
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(test_loader)
    run_dir = output_root / "training" / f"{args.model}__baseline_ce__seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    write_json(
        run_dir / "clean_test_metrics.json",
        {
            **test_metrics,
            "training_time_s": time.time() - started,
            "runtime_reference": "Kaggle T4x2 GPU, Python 3.12.3",
        },
    )
    save_confusion_matrix(test_metrics["confusion_matrix"], run_dir / "clean_test_confusion_matrix")
    state_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(state_model.state_dict(), run_dir / "best.pth")
    print(f"Completed TIMM run: {run_dir}")


if __name__ == "__main__":
    main()
