from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
EWU_ROOT = ROOT / "ewu_shrimp_disease-1"
HAND_ROOT = ROOT / "shrimpDisHandSegV2-1"
REPORT_DIR = ROOT / "reports" / "dataset_style_comparison"
FIG_DIR = REPORT_DIR / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EWU_WITH_IMAGE_NUM_PATTERN = re.compile(
    r"^Shrimp_(?P<shrimp_id>.+?)-(?P<img_num>\d+)-?$",
    re.IGNORECASE,
)
EWU_SINGLE_IMAGE_PATTERN = re.compile(r"^Shrimp_(?P<shrimp_id>.+)$", re.IGNORECASE)
EWU_PAREN_PATTERN = re.compile(
    r"^Shrimp_(?P<shrimp_id>[^\s()]+)\s*\((?P<img_num>\d+)\)$",
    re.IGNORECASE,
)
HAND_PATTERN = re.compile(
    r"^(?P<disease>.+)-(?P<shrimp_id>[^-]+)-img-(?P<img_num>\d+)$",
    re.IGNORECASE,
)

DATASETS = [
    {
        "dataset_key": "ewu",
        "dataset_name": "EWU Roboflow segmentation",
        "root": EWU_ROOT,
        "parser": "ewu",
    },
    {
        "dataset_key": "hand",
        "dataset_name": "Hand-labeled segmentation",
        "root": HAND_ROOT,
        "parser": "hand",
    },
]

COLORS = [
    (255, 70, 70),
    (70, 150, 255),
    (70, 210, 110),
    (250, 190, 70),
    (200, 90, 255),
]


@dataclass
class LabelInstance:
    class_id: int
    class_name: str
    points: list[tuple[float, float]]


def normalize_roboflow_stem(stem: str) -> str:
    stem = re.sub(r"_(jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    return stem


def parse_simple_yaml_names(yaml_path: Path) -> list[str]:
    if not yaml_path.exists():
        return []
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"names:\s*\[(.*?)\]", text, flags=re.DOTALL)
    if match:
        raw = match.group(1)
        return [part.strip().strip("'\"") for part in raw.split(",") if part.strip()]
    lines = text.splitlines()
    names: list[str] = []
    in_names = False
    for line in lines:
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not line.startswith(" ") and not line.startswith("-"):
                break
            item = re.sub(r"^\s*-\s*", "", line).strip().strip("'\"")
            if item:
                names.append(item)
    return names


def parse_image_name(path: Path, parser: str) -> dict:
    stem = normalize_roboflow_stem(path.stem)
    if parser == "ewu":
        match = EWU_WITH_IMAGE_NUM_PATTERN.match(stem)
        if match:
            shrimp_id = match.group("shrimp_id")
            img_num = int(match.group("img_num"))
        else:
            match = EWU_PAREN_PATTERN.match(stem)
            if match:
                shrimp_id = match.group("shrimp_id")
                img_num = int(match.group("img_num"))
            else:
                match = EWU_SINGLE_IMAGE_PATTERN.match(stem)
                if not match:
                    return {
                        "parse_ok": False,
                        "disease_from_name": "unparsed",
                        "shrimp_id": None,
                        "img_num": None,
                        "specimen_key": f"unparsed::{stem}",
                        "normalized_stem": stem,
                    }
                shrimp_id = match.group("shrimp_id")
                img_num = 1
        return {
            "parse_ok": True,
            "disease_from_name": "Shrimp",
            "shrimp_id": shrimp_id,
            "img_num": img_num,
            "specimen_key": f"shrimp::{shrimp_id}",
            "normalized_stem": stem,
        }

    match = HAND_PATTERN.match(stem)
    if match:
        disease = match.group("disease")
        shrimp_id = match.group("shrimp_id")
        img_num = int(match.group("img_num"))
        return {
            "parse_ok": True,
            "disease_from_name": disease,
            "shrimp_id": shrimp_id,
            "img_num": img_num,
            "specimen_key": f"{disease.lower()}::{shrimp_id}",
            "normalized_stem": stem,
        }
    return {
        "parse_ok": False,
        "disease_from_name": "unparsed",
        "shrimp_id": None,
        "img_num": None,
        "specimen_key": f"unparsed::{stem}",
        "normalized_stem": stem,
    }


def split_from_path(path: Path) -> str:
    lower_parts = [p.lower() for p in path.parts]
    for split in ["train", "valid", "val", "test"]:
        if split in lower_parts:
            return "valid" if split == "val" else split
    return "all"


def image_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_labels(label_path: Path, class_names: list[str]) -> list[LabelInstance]:
    if not label_path.exists():
        return []
    instances: list[LabelInstance] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        points = list(zip(coords[0::2], coords[1::2]))
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        instances.append(LabelInstance(class_id=class_id, class_name=class_name, points=points))
    return instances


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def average_hash(image_path: Path, hash_size: int = 8) -> str:
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS), dtype=np.float32)
    mean = float(arr.mean())
    bits = (arr >= mean).flatten()
    return "".join("1" if b else "0" for b in bits)


def dhash(image_path: Path, hash_size: int = 8) -> str:
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS), dtype=np.float32)
    bits = (arr[:, 1:] >= arr[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def color_signature(image_path: Path, size: int = 8) -> np.ndarray:
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    return arr.reshape(-1)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def draw_overlay(image_path: Path, instances: list[LabelInstance], size=(360, 360)) -> Image.Image:
    with Image.open(image_path) as img:
        base = img.convert("RGB")
    base.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    offset = ((size[0] - base.width) // 2, (size[1] - base.height) // 2)
    canvas.paste(base, offset)
    draw = ImageDraw.Draw(canvas, "RGBA")
    scale_x = base.width
    scale_y = base.height
    off_x, off_y = offset
    for inst in instances:
        color = COLORS[inst.class_id % len(COLORS)]
        pts = [(off_x + x * scale_x, off_y + y * scale_y) for x, y in inst.points]
        if len(pts) >= 3:
            draw.polygon(pts, fill=(*color, 70), outline=(*color, 230))
            draw.line(pts + [pts[0]], fill=(*color, 255), width=3)
    return canvas


def text_block(lines: list[str], width: int, height: int, fill=(255, 255, 255)) -> Image.Image:
    image = Image.new("RGB", (width, height), fill)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    y = 8
    for line in lines:
        draw.text((8, y), line[:90], fill=(20, 20, 20), font=font)
        y += 14
        if y > height - 14:
            break
    return image


def build_image_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows = []
    class_names_by_dataset = {}
    for cfg in DATASETS:
        root = cfg["root"]
        class_names = parse_simple_yaml_names(root / "data.yaml")
        class_names_by_dataset[cfg["dataset_key"]] = class_names
        for image_path in image_files(root):
            parsed = parse_image_name(image_path, cfg["parser"])
            label_path = label_path_for_image(image_path)
            instances = read_labels(label_path, class_names)
            class_counter = Counter(inst.class_name for inst in instances)
            class_id_counter = Counter(inst.class_id for inst in instances)
            rows.append(
                {
                    "dataset_key": cfg["dataset_key"],
                    "dataset_name": cfg["dataset_name"],
                    "split": split_from_path(image_path),
                    "image_path": str(image_path),
                    "image_name": image_path.name,
                    "label_path": str(label_path),
                    "label_exists": label_path.exists(),
                    "parse_ok": parsed["parse_ok"],
                    "specimen_key": parsed["specimen_key"],
                    "shrimp_id": parsed["shrimp_id"],
                    "img_num": parsed["img_num"],
                    "disease_from_name": parsed["disease_from_name"],
                    "normalized_stem": parsed["normalized_stem"],
                    "mask_instances": len(instances),
                    "mask_classes": ",".join(sorted(class_counter)),
                    "mask_class_ids": ",".join(str(k) for k in sorted(class_id_counter)),
                    "co_infection": len(class_id_counter) > 1,
                    "total_mask_area_norm": sum(polygon_area(inst.points) for inst in instances),
                    "mean_mask_area_norm": np.mean([polygon_area(inst.points) for inst in instances]) if instances else 0.0,
                    "ahash": average_hash(image_path),
                    "dhash": dhash(image_path),
                }
            )
    df = pd.DataFrame(rows)
    return df, class_names_by_dataset


def save_counts(df: pd.DataFrame) -> None:
    df.to_csv(REPORT_DIR / "image_level_summary.csv", index=False)

    dataset_summary = (
        df.groupby("dataset_key")
        .agg(
            images=("image_path", "count"),
            specimens=("specimen_key", "nunique"),
            parsed_images=("parse_ok", "sum"),
            labeled_images=("mask_instances", lambda s: int((s > 0).sum())),
            healthy_or_empty_images=("mask_instances", lambda s: int((s == 0).sum())),
            total_masks=("mask_instances", "sum"),
            co_infection_images=("co_infection", "sum"),
            mean_masks_per_labeled_image=("mask_instances", lambda s: float(s[s > 0].mean()) if (s > 0).any() else 0.0),
            max_images_per_specimen=("specimen_key", lambda s: int(s.value_counts().max())),
        )
        .reset_index()
    )
    dataset_summary.to_csv(REPORT_DIR / "dataset_summary.csv", index=False)

    class_rows = []
    for _, row in df.iterrows():
        if not row["mask_classes"]:
            class_rows.append(
                {
                    "dataset_key": row["dataset_key"],
                    "split": row["split"],
                    "class_name": "empty_label_healthy",
                    "instances": 0,
                    "image_mentions": 1,
                }
            )
            continue
        for class_name in row["mask_classes"].split(","):
            count = 0
            label_instances = read_labels(Path(row["label_path"]), [])
            for inst in label_instances:
                if str(inst.class_id) in row["mask_class_ids"].split(","):
                    pass
            class_rows.append(
                {
                    "dataset_key": row["dataset_key"],
                    "split": row["split"],
                    "class_name": class_name,
                    "instances": row["mask_classes"].split(",").count(class_name),
                    "image_mentions": 1,
                }
            )
    exploded = []
    for _, row in df.iterrows():
        label_instances = read_labels(Path(row["label_path"]), parse_simple_yaml_names(Path(row["image_path"]).parents[2] / "data.yaml"))
        if not label_instances:
            exploded.append((row["dataset_key"], row["split"], "empty_label_healthy", 0, 1))
        for inst in label_instances:
            exploded.append((row["dataset_key"], row["split"], inst.class_name, 1, 1))
    class_df = pd.DataFrame(exploded, columns=["dataset_key", "split", "class_name", "instances", "image_mentions"])
    class_summary = class_df.groupby(["dataset_key", "split", "class_name"], as_index=False).agg(
        instances=("instances", "sum"),
        image_mentions=("image_mentions", "sum"),
    )
    class_summary.to_csv(REPORT_DIR / "class_and_coinfection_counts.csv", index=False)

    group_summary = (
        df.groupby(["dataset_key", "specimen_key"])
        .agg(
            images=("image_path", "count"),
            splits=("split", lambda s: ",".join(sorted(set(s)))),
            shrimp_id=("shrimp_id", "first"),
            disease_from_name=("disease_from_name", "first"),
            mask_instances=("mask_instances", "sum"),
            co_infection_images=("co_infection", "sum"),
            example_images=("image_name", lambda s: "; ".join(list(s)[:5])),
        )
        .reset_index()
        .sort_values(["dataset_key", "images"], ascending=[True, False])
    )
    group_summary.to_csv(REPORT_DIR / "specimen_group_summary.csv", index=False)

    print("Dataset summary")
    print(dataset_summary.to_string(index=False))
    print("\nClass summary")
    print(class_summary.to_string(index=False))


def create_group_panel(df: pd.DataFrame, dataset_key: str, group_keys: list[str], output_path: Path, title: str) -> None:
    tile_w, tile_h = 360, 430
    max_cols = 4
    rows = []
    for group_key in group_keys:
        group = df[(df.dataset_key == dataset_key) & (df.specimen_key == group_key)].sort_values("img_num")
        tiles = []
        for _, row in group.head(max_cols).iterrows():
            instances = read_labels(Path(row.label_path), parse_simple_yaml_names(Path(row.image_path).parents[2] / "data.yaml"))
            overlay = draw_overlay(Path(row.image_path), instances, size=(360, 330))
            caption = text_block(
                [
                    row.image_name,
                    f"group: {row.specimen_key}",
                    f"masks: {row.mask_instances} | classes: {row.mask_classes or 'empty'}",
                ],
                360,
                100,
            )
            tile = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
            tile.paste(overlay, (0, 0))
            tile.paste(caption, (0, 330))
            tiles.append(tile)
        while len(tiles) < max_cols:
            tiles.append(Image.new("RGB", (tile_w, tile_h), (250, 250, 250)))
        row_img = Image.new("RGB", (tile_w * max_cols, tile_h), (255, 255, 255))
        for idx, tile in enumerate(tiles):
            row_img.paste(tile, (idx * tile_w, 0))
        rows.append(row_img)

    header = text_block([title], tile_w * max_cols, 40, fill=(235, 240, 250))
    panel = Image.new("RGB", (tile_w * max_cols, 40 + tile_h * len(rows)), (255, 255, 255))
    panel.paste(header, (0, 0))
    y = 40
    for row_img in rows:
        panel.paste(row_img, (0, y))
        y += tile_h
    panel.save(output_path)


def representative_rows(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    rows = []
    subset = df[df.dataset_key == dataset_key].copy()
    subset = subset.sort_values(["specimen_key", "mask_instances", "img_num"], ascending=[True, False, True])
    for _, group in subset.groupby("specimen_key"):
        rows.append(group.iloc[0])
    return pd.DataFrame(rows)


def find_similar_pairs(df: pd.DataFrame, top_n: int = 36) -> pd.DataFrame:
    # Compare specimen representatives first. This is faster and better aligned with
    # the paper figure: "these groups look visually similar, but label style differs."
    ewu = representative_rows(df, "ewu")
    hand = representative_rows(df, "hand")
    hand_rows = list(hand.itertuples(index=False))
    hand_bits = np.asarray(
        [[1 if ch == "1" else 0 for ch in (row.ahash + row.dhash)] for row in hand_rows],
        dtype=np.uint8,
    )
    hand_colors = np.asarray([color_signature(Path(row.image_path)) for row in hand_rows], dtype=np.float32)
    pairs = []
    print(f"Visual hash matching {len(ewu)} EWU specimen representatives against {len(hand)} hand-labeled representatives...", flush=True)
    for erow in ewu.itertuples(index=False):
        ebits = np.asarray([1 if ch == "1" else 0 for ch in (erow.ahash + erow.dhash)], dtype=np.uint8)
        esig = color_signature(Path(erow.image_path)).astype(np.float32)
        ham = np.count_nonzero(hand_bits != ebits, axis=1)
        color_dist = np.mean(np.abs(hand_colors - esig[None, :]), axis=1)
        score = ham.astype(np.float32) + 40.0 * color_dist
        best_indices = np.argsort(score)[:2]
        for best_idx in best_indices:
            hrow = hand_rows[int(best_idx)]
            pairs.append(
                {
                    "score": float(score[best_idx]),
                    "hash_hamming": int(ham[best_idx]),
                    "color_l1": float(color_dist[best_idx]),
                    "ewu_image": erow.image_name,
                    "ewu_path": erow.image_path,
                    "ewu_label_path": erow.label_path,
                    "ewu_group": erow.specimen_key,
                    "ewu_masks": erow.mask_instances,
                    "ewu_classes": erow.mask_classes,
                    "hand_image": hrow.image_name,
                    "hand_path": hrow.image_path,
                    "hand_label_path": hrow.label_path,
                    "hand_group": hrow.specimen_key,
                    "hand_masks": hrow.mask_instances,
                    "hand_classes": hrow.mask_classes,
                }
            )
    pair_df = pd.DataFrame(pairs).sort_values(["score", "hash_hamming", "color_l1"]).head(top_n)
    pair_df.to_csv(REPORT_DIR / "ewu_hand_visual_hash_similar_pairs.csv", index=False)
    return pair_df


def create_pair_panel(pair_df: pd.DataFrame, output_path: Path, max_pairs: int = 12) -> None:
    tile_w, tile_h = 360, 430
    row_w = tile_w * 2
    header = text_block(["EWU vs hand-labeled visually similar images with mask overlays"], row_w, 42, fill=(235, 240, 250))
    rows = []
    for _, pair in pair_df.head(max_pairs).iterrows():
        ewu_instances = read_labels(Path(pair.ewu_label_path), parse_simple_yaml_names(EWU_ROOT / "data.yaml"))
        hand_instances = read_labels(Path(pair.hand_label_path), parse_simple_yaml_names(HAND_ROOT / "data.yaml"))
        ewu_overlay = draw_overlay(Path(pair.ewu_path), ewu_instances, size=(360, 320))
        hand_overlay = draw_overlay(Path(pair.hand_path), hand_instances, size=(360, 320))
        ewu_caption = text_block(
            [
                "EWU",
                pair.ewu_image,
                f"group: {pair.ewu_group}",
                f"masks/classes: {pair.ewu_masks} / {pair.ewu_classes or 'empty'}",
            ],
            360,
            110,
        )
        hand_caption = text_block(
            [
                "Hand-labeled",
                pair.hand_image,
                f"group: {pair.hand_group}",
                f"masks/classes: {pair.hand_masks} / {pair.hand_classes or 'empty'}",
                f"match score: {pair.score:.2f}",
            ],
            360,
            110,
        )
        left = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
        right = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
        left.paste(ewu_overlay, (0, 0))
        left.paste(ewu_caption, (0, 320))
        right.paste(hand_overlay, (0, 0))
        right.paste(hand_caption, (0, 320))
        row_img = Image.new("RGB", (row_w, tile_h), (255, 255, 255))
        row_img.paste(left, (0, 0))
        row_img.paste(right, (tile_w, 0))
        rows.append(row_img)
    panel = Image.new("RGB", (row_w, 42 + tile_h * len(rows)), (255, 255, 255))
    panel.paste(header, (0, 0))
    y = 42
    for row_img in rows:
        panel.paste(row_img, (0, y))
        y += tile_h
    panel.save(output_path)


def main() -> None:
    df, class_names_by_dataset = build_image_table()
    print(f"Built image table with {len(df)} rows.", flush=True)
    save_counts(df)

    for dataset_key in ["ewu", "hand"]:
        group_counts = df[df.dataset_key == dataset_key].groupby("specimen_key").size().sort_values(ascending=False)
        selected_groups = group_counts.head(4).index.tolist()
        create_group_panel(
            df,
            dataset_key,
            selected_groups,
            FIG_DIR / f"{dataset_key}_within_specimen_label_style_examples.png",
            f"{dataset_key.upper()} within-specimen images and mask style",
        )

    pair_df = find_similar_pairs(df)
    create_pair_panel(pair_df, FIG_DIR / "ewu_vs_hand_visual_hash_similar_pairs.png", max_pairs=12)

    report_md = REPORT_DIR / "README_dataset_style_comparison.md"
    report_md.write_text(
        "\n".join(
            [
                "# Dataset Style Comparison Report",
                "",
                "Generated artifacts:",
                "",
                "- `dataset_summary.csv`",
                "- `class_and_coinfection_counts.csv`",
                "- `specimen_group_summary.csv`",
                "- `image_level_summary.csv`",
                "- `ewu_hand_visual_hash_similar_pairs.csv`",
                "- `figures/ewu_within_specimen_label_style_examples.png`",
                "- `figures/hand_within_specimen_label_style_examples.png`",
                "- `figures/ewu_vs_hand_visual_hash_similar_pairs.png`",
                "",
                "Use these figures to discuss labeling consistency, mask style differences, same-specimen grouping, and visually similar EWU/hand-labeled images.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nSaved reports to: {REPORT_DIR}")
    for path in sorted(REPORT_DIR.rglob("*")):
        if path.is_file():
            print(path.relative_to(REPORT_DIR))


if __name__ == "__main__":
    main()
