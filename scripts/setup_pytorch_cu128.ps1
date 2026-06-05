# Install PyTorch nightly with CUDA 12.8 (required for RTX 50xx / sm_120).
# Run from repo root in PowerShell:
#   .\scripts\setup_pytorch_cu128.ps1
#
# Needs ~8 GB free on the drive where .venv lives (wheel is ~2.8 GB).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$MinFreeGb = 8

function Get-FreeGb($Path) {
    $drive = (Get-Item $Path).PSDrive.Name
    return [math]::Round((Get-PSDrive $drive).Free / 1GB, 2)
}

$free = Get-FreeGb $Root
if ($free -lt $MinFreeGb) {
    Write-Host "ERROR: Only ${free} GB free on $(Split-Path $Root -Qualifier). Need at least $MinFreeGb GB." -ForegroundColor Red
    Write-Host "Free space (pip cache, old zips, Temp), then re-run this script."
    exit 1
}

Set-Location $Root

if (-not (Test-Path $Venv)) {
    Write-Host "Creating venv at $Venv ..."
    py -3 -m venv $Venv
}

$Py = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"

& $Pip install -U pip
& $Pip uninstall torch torchvision torchaudio -y 2>$null
& $Pip install --pre torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/nightly/cu128 `
    --no-cache-dir

Write-Host "`nVerifying GPU ..."
& $Py scripts\verify_pytorch_cuda.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`nUse this Python for attack eval:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python scripts\compare_hidden_vs_moe_attacks.py --device cuda ..."
