param(
    [ValidateSet("auto", "gpu", "cpu")]
    [string]$Mode = "auto",

    [ValidateSet("cu118", "cu126", "cu128")]
    [string]$CudaVariant = "cu118"
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

function Get-TorchIndexUrl([string]$TargetMode, [string]$TargetCudaVariant) {
    if ($TargetMode -eq "cpu") {
        return "https://download.pytorch.org/whl/cpu"
    }

    switch ($TargetCudaVariant) {
        "cu118" { return "https://download.pytorch.org/whl/cu118" }
        "cu126" { return "https://download.pytorch.org/whl/cu126" }
        "cu128" { return "https://download.pytorch.org/whl/cu128" }
        default { throw "Unsupported CUDA variant: $TargetCudaVariant" }
    }
}

function Test-NvidiaGpuAvailable {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return $false
    }

    try {
        $gpuList = & $nvidiaSmi.Source -L 2>$null
        return ($LASTEXITCODE -eq 0 -and $gpuList)
    } catch {
        return $false
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not installed. Install uv first, then run this script again."
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} else {
    Write-Output "Using existing virtual environment: .venv"
}

$gpuAvailable = Test-NvidiaGpuAvailable
$resolvedMode = $Mode
if ($Mode -eq "auto") {
    if ($gpuAvailable) {
        $resolvedMode = "gpu"
        Write-Output "Detected NVIDIA GPU via nvidia-smi. Installing CUDA PyTorch ($CudaVariant)."
    } else {
        $resolvedMode = "cpu"
        Write-Output "No NVIDIA GPU detected. Falling back to CPU PyTorch."
    }
} elseif ($Mode -eq "gpu" -and -not $gpuAvailable) {
    throw "GPU mode was requested, but no NVIDIA GPU was detected by nvidia-smi."
}

$torchIndexUrl = Get-TorchIndexUrl -TargetMode $resolvedMode -TargetCudaVariant $CudaVariant
Write-Output "Installing torch/torchvision from: $torchIndexUrl"

uv pip install --python $python torch torchvision --index-url $torchIndexUrl
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

uv pip install --python $python -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$verifyScript = @"
import json
import torch, torchvision, ultralytics, cv2, pandas

payload = {
    "python_environment_ready": True,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "ultralytics": ultralytics.__version__,
    "cv2": cv2.__version__,
    "pandas": pandas.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
}
if payload["cuda_available"] and payload["cuda_device_count"] > 0:
    payload["cuda_device_name"] = torch.cuda.get_device_name(0)
print(json.dumps(payload, ensure_ascii=True))
"@

$verifyOutput = & $python -c $verifyScript
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$verify = $verifyOutput | ConvertFrom-Json

Write-Output "Python environment ready"
Write-Output "torch $($verify.torch) cuda $($verify.cuda_available)"
Write-Output "torchvision $($verify.torchvision)"
Write-Output "ultralytics $($verify.ultralytics)"
Write-Output "cv2 $($verify.cv2)"
Write-Output "pandas $($verify.pandas)"
if ($verify.cuda_available) {
    Write-Output "CUDA device count: $($verify.cuda_device_count)"
    Write-Output "CUDA device 0: $($verify.cuda_device_name)"
}

if ($resolvedMode -eq "gpu" -and -not $verify.cuda_available) {
    throw "CUDA PyTorch was installed, but torch.cuda.is_available() is False."
}
