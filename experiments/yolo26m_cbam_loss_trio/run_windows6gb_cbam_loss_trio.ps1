param(
  [ValidateSet("smoke", "full")]
  [string]$Mode = "smoke"
)

$ErrorActionPreference = "Stop"

$REPO = "C:\Users\ADMIN\Downloads\CVio_Shrimp_Disease_Classification_Capstone_SU26"
$OUT_ROOT = "C:\Users\ADMIN\Downloads\cvio_windows6gb_outputs"
$DATASET_ROOT = "C:\Users\ADMIN\Downloads\processed-images\processed_images"

Set-Location $REPO

$env:PYTHONUNBUFFERED = "1"
$env:CUDA_VISIBLE_DEVICES = "0"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
  throw "Missing uv venv. Run: uv venv .venv --python 3.12.2"
}
if (!(Test-Path "$REPO\shrimp_scripts")) {
  throw "Missing shrimp_scripts in repo root. Pull/copy the full CVio project code first."
}
if (!(Test-Path "$REPO\experiments\asl_custom_loss_screening")) {
  throw "Missing experiments\asl_custom_loss_screening in repo root. Pull/copy the full CVio project code first."
}
if (!(Test-Path $DATASET_ROOT)) {
  throw "Dataset root not found: $DATASET_ROOT"
}

New-Item -ItemType Directory -Force $OUT_ROOT | Out-Null

$EPOCHS = 1
$START = 1
$LIMIT = 1
$OUT = "$OUT_ROOT\yolo26m_cbam_loss_trio_SMOKE_outputs"
$WORK = "$OUT_ROOT\YOLO26M_CBAM_LOSS_TRIO_SMOKE_project"
$LOG = "$OUT_ROOT\yolo26m_cbam_loss_trio_SMOKE_live.log"

if ($Mode -eq "full") {
  $EPOCHS = 30
  $START = 1
  $LIMIT = 0
  $OUT = "$OUT_ROOT\yolo26m_cbam_loss_trio_outputs"
  $WORK = "$OUT_ROOT\YOLO26M_CBAM_LOSS_TRIO_FULL_project"
  $LOG = "$OUT_ROOT\yolo26m_cbam_loss_trio_live.log"
}

& .\.venv\Scripts\python.exe -u .\experiments\yolo26m_cbam_loss_trio\yolo26m_cbam_loss_trio_runner.py `
  --project_dir "$REPO" `
  --work_project_dir "$WORK" `
  --output_dir "$OUT" `
  --local_dataset_root "$DATASET_ROOT" `
  --epochs $EPOCHS `
  --batch 4 `
  --workers 0 `
  --imgsz 224 `
  --device 0 `
  --start $START `
  --limit $LIMIT `
  --force_project_copy `
  --stop_on_failure `
  2>&1 | Tee-Object $LOG
