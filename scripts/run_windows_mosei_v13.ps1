param(
    [int]$GpuId = 0,

    [ValidateSet("highest", "high", "medium")]
    [string]$MatmulPrecision = "high",

    [int]$NumWorkers = 0,

    [switch]$Preflight,
    [switch]$Smoke,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

if ($NumWorkers -ne 0) {
    throw "MOSEI v13 Windows runner fixes NumWorkers=0."
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    $pythonArgs = @(
        ".\train_mosei_v13.py",
        "--seed", "1113",
        "--gpu-id", [string]$GpuId,
        "--num-workers", [string]$NumWorkers,
        "--matmul-precision", $MatmulPrecision
    )

    if ($Preflight) {
        $pythonArgs += "--preflight-only"
    }
    if ($Smoke) {
        $pythonArgs += "--smoke-test"
    }
    if ($Overwrite) {
        $pythonArgs += "--overwrite"
    }

    Write-Host "================ MOSEI frozen v13 Windows runner =================="
    Write-Host "Seed:              1113 (frozen from MOSI v13)"
    Write-Host ("GPU:               {0}" -f $GpuId)
    Write-Host ("NumWorkers:        {0}" -f $NumWorkers)
    Write-Host ("MatmulPrecision:   {0}" -f $MatmulPrecision)
    Write-Host ("Preflight:         {0}" -f [bool]$Preflight)
    Write-Host ("Smoke:             {0}" -f [bool]$Smoke)
    Write-Host ("Overwrite:         {0}" -f [bool]$Overwrite)
    Write-Host "Official Valid:    fold training/selection NEVER uses it"
    Write-Host "Official Test:     WILL NOT BE CONSTRUCTED"
    Write-Host "Downstream blend:  Raw5 0.5 + v13 0.5 (NO MOSEI weight search)"
    Write-Host "==================================================================="

    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "MOSEI frozen v13 failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
