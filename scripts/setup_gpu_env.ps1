# One-time setup of Person A's environment on Windows (GPU laptop or backup laptop).
#   powershell -ExecutionPolicy Bypass -File scripts\setup_gpu_env.ps1
# Needs: Python 3.12 (py launcher), git, an NVIDIA driver supporting CUDA 12.8.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

function Invoke-Checked { param([scriptblock]$Cmd, [string]$What)
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) { Invoke-Checked { py -3.12 -m venv .venv } "creating .venv" }
$Py = Join-Path $Repo ".venv\Scripts\python.exe"
Invoke-Checked { & $Py -m pip install -U pip } "pip upgrade"
Invoke-Checked { & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 } "installing CUDA PyTorch"
Invoke-Checked { & $Py -m pip install -r requirements-ml.txt } "installing requirements-ml.txt"
& (Join-Path $PSScriptRoot "setup_triposr.ps1")
Invoke-Checked { & $Py -c "import torch; print('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')" } "torch check"
