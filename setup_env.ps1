$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

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

uv pip install --python $python torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

uv pip install --python $python -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -c "import torch, torchvision, ultralytics, cv2, pandas; print('Python environment ready'); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchvision', torchvision.__version__); print('ultralytics', ultralytics.__version__); print('cv2', cv2.__version__); print('pandas', pandas.__version__)"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
