$ErrorActionPreference = "Stop"
Write-Host "=== Windows / CPU / RAM ==="
systeminfo | Select-String "Host Name|OS Name|OS Version|Total Physical Memory|Available Physical Memory"
Write-Host "=== GPU ==="
nvidia-smi
Write-Host "=== Python ==="
python --version
python -c "import sys; print(sys.executable)"
Write-Host "=== Torch CUDA ==="
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); print('vram GB:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,2) if torch.cuda.is_available() else 'NO CUDA')"
