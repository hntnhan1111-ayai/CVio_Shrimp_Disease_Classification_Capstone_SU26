"""Central configuration for the shrimp disease paper experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_TITLE = "Improving Lightweight Shrimp Disease Classification with Co-Infection-Aware Losses and RandAugment"
DATASET_ID = "uynnhy/processed-images"
DEFAULT_OUTPUT_DIR = Path("/kaggle/working/shrimp_outputs")

SEED = 42
SPLIT_SEED = 42
REPEAT = 1
PYTHON_TARGET = "3.12"
RUNTIME_TARGET = "Kaggle T4x2 GPU"

IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_DIRS = ["1. Healthy", "2. BG", "3. WSSV", "4. WSSV_BG"]
CLASS_NAMES = ["Healthy", "BG", "WSSV", "WSSV_BG"]
CLASS_TO_LABEL = dict(zip(CLASS_DIRS, range(NUM_CLASSES)))
DIR_TO_NAME = dict(zip(CLASS_DIRS, CLASS_NAMES))
EXPECTED_CLASS_COUNTS = {
    "1. Healthy": 403,
    "2. BG": 198,
    "3. WSSV": 328,
    "4. WSSV_BG": 220,
}
EXPECTED_NAME_COUNTS = dict(zip(CLASS_NAMES, EXPECTED_CLASS_COUNTS.values()))
EXPECTED_TOTAL_IMAGES = 1149
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RUNS_SUBDIR = "runs"
FIGURES_SUBDIR = "figures"
REPORTS_SUBDIR = "reports"
YOLO_DATASET_SUBDIR = "yolo_fixed_dataset"

EPOCHS = 30
SMOKE_TEST_EPOCHS = 1
USE_AMP = True

RAND_AUGMENT_NUM_OPS = 2
RAND_AUGMENT_MAGNITUDE = 9

TORCH_EFFECTIVE_BATCH = 128
TORCH_MICRO_BATCH_FALLBACKS = [32, 16, 8]
TORCH_EVAL_BATCH = 128
TORCH_WORKERS = 2
TORCH_WEIGHT_DECAY = 0.0
CONVNEXT_WARMUP_EPOCHS = 5
CONVNEXT_PATIENCE = 5
CONVNEXT_WARMUP_LR = 1e-3
CONVNEXT_BACKBONE_LR = 2e-5
CONVNEXT_HEAD_LR = 1e-4
CONVNEXT_STEP_SIZE = 3
CONVNEXT_STEP_GAMMA = 0.9
CONVNEXT_UNFREEZE_FROM_FEATURE_INDEX = 5

LIGHTWEIGHT_TORCH_PATIENCE = 15
LIGHTWEIGHT_LR = 1e-4

YOLO_BATCH_FALLBACKS = [128, 64, 32, 16]
YOLO_WORKERS = 8
YOLO_PATIENCE = 15
YOLO_CACHE = True
YOLO_OPTIMIZER = "AdamW"
YOLO_LR0 = 1.25e-3
YOLO_LRF = 0.01
YOLO_COS_LR = True

CORE_CONDITIONS = [
    {"condition_key": "ce_no_randaugment", "loss_key": "baseline_ce", "randaugment": False},
    {"condition_key": "asl_no_randaugment", "loss_key": "asl_single_label", "randaugment": False},
    {"condition_key": "pairwise_no_randaugment", "loss_key": "pairwise_coinfection_ranking_asl", "randaugment": False},
    {"condition_key": "ce_randaugment", "loss_key": "baseline_ce", "randaugment": True},
    {"condition_key": "asl_randaugment", "loss_key": "asl_single_label", "randaugment": True},
    {"condition_key": "pairwise_randaugment", "loss_key": "pairwise_coinfection_ranking_asl", "randaugment": True},
]
FAMILY_CONDITIONS = [
    CORE_CONDITIONS[0],
    CORE_CONDITIONS[1],
    CORE_CONDITIONS[5],
]

CONVNEXT_CORE_MODEL_KEY = "convnext_tiny_shrimpxnet"
CONVNEXT_CORE_MODEL_NAME = "ConvNeXt-Tiny ShrimpXNet-style"
YOLO_CORE_MODEL = "yolo26m-cls"
YOLO_FAMILY_MODELS = ["yolov8m-cls", "yolov9m-cls", "yolov10m-cls", "yolo11m-cls", "yolo26m-cls"]
LIGHTWEIGHT_MODELS = [
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

TIMM_MODEL_ALIASES = {
    "convnext_tiny_in22k": ["convnext_tiny.fb_in22k", "convnext_tiny.fb_in1k", "convnext_tiny"],
    "efficientnet_b0": ["efficientnet_b0.ra_in1k", "tf_efficientnet_b0", "efficientnet_b0"],
    "repvgg_a0": ["repvgg_a0.rvgg_in1k", "repvgg_a0"],
    "efficientnet_v2_s": ["tf_efficientnetv2_s.in21k_ft_in1k", "tf_efficientnetv2_s", "efficientnetv2_rw_s"],
    "fastvit_t8": ["fastvit_t8.apple_in1k", "fastvit_t8"],
    "edgenext_xx_small": ["edgenext_xx_small.in1k", "edgenext_xx_small"],
    "mobileone_s0": ["mobileone_s0.apple_in1k", "mobileone_s0"],
    "mobilevit_s": ["mobilevit_s.cvnets_in1k", "mobilevit_s"],
    "mobilenetv4_conv_small": ["mobilenetv4_conv_small.e2400_r224_in1k", "mobilenetv4_conv_small"],
    "mnasnet_100": ["mnasnet_100.rmsp_in1k", "mnasnet_100"],
    "ghostnetv2_100": ["ghostnetv2_100.in1k", "ghostnetv2_100"],
    "rexnet_100": ["rexnet_100.nav_in1k", "rexnet_100"],
    "mobilenetv4_hybrid_medium": ["mobilenetv4_hybrid_medium.e500_r224_in1k", "mobilenetv4_hybrid_medium"],
    "efficientvit_m1": ["efficientvit_m1.r224_in1k", "efficientvit_m1"],
}


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    return {
        "output": root,
        "runs": root / RUNS_SUBDIR,
        "figures": root / FIGURES_SUBDIR,
        "reports": root / REPORTS_SUBDIR,
        "yolo_dataset": root / YOLO_DATASET_SUBDIR,
    }


def epochs_for(smoke_test: bool) -> int:
    return SMOKE_TEST_EPOCHS if smoke_test else EPOCHS

