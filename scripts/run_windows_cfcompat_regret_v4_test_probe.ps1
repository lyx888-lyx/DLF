[CmdletBinding()]
param(
    [switch]$Preflight,
    [switch]$RunOnce
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if (($Preflight -and $RunOnce) -or (-not $Preflight -and -not $RunOnce)) {
    throw "Choose exactly one: -Preflight or -RunOnce"
}

$Branch = (git branch --show-current).Trim()
$Commit = (git rev-parse HEAD).Trim()
Write-Host "Project: $ProjectRoot"
Write-Host "Branch:  $Branch"
Write-Host "Commit:  $Commit"
Write-Host "Probe:   MOSI Test viability: frozen CFCompatKD vs Regret-Preserve v4"
Write-Host "Seed:    1113 only"

if ($Branch -ne "feature/cfcompat-regret-v4-test-viability-probe-v1") {
    throw "Run this probe only from feature/cfcompat-regret-v4-test-viability-probe-v1; current branch: $Branch"
}

& python -m py_compile ".\test_probe_cfcompat_regret_v4_seed1113.py"
if ($LASTEXITCODE -ne 0) {
    throw "Test-probe Python syntax check failed."
}

if ($Preflight) {
    & python ".\test_probe_cfcompat_regret_v4_seed1113.py" `
        --preflight `
        --dataset mosi `
        --gpu-ids 0 `
        --num-workers 1 `
        --result-root result `
        --model-save-dir pt
    if ($LASTEXITCODE -ne 0) {
        throw "Test-probe preflight failed with exit code $LASTEXITCODE"
    }
    Write-Host "Preflight only: MOSI Test was not constructed."
    return
}

Write-Host ""
Write-Host "============================================================"
Write-Host "ONE-TIME TEST PROBE AUTHORIZED"
Write-Host "This command will now construct MOSI Test and consume the"
Write-Host "single viability probe for Seed1113 CFCompatKD vs v4."
Write-Host "No Train/Valid loader, optimizer, or checkpoint selection is used."
Write-Host "============================================================"
Write-Host ""

& python ".\test_probe_cfcompat_regret_v4_seed1113.py" `
    --run-once `
    --confirm-token "RUN_MOSI_TEST_PROBE_1113_CFCompat_vs_V4_ONCE" `
    --dataset mosi `
    --gpu-ids 0 `
    --num-workers 1 `
    --result-root result `
    --model-save-dir pt
if ($LASTEXITCODE -ne 0) {
    throw "One-time MOSI Test viability probe failed with exit code $LASTEXITCODE"
}
