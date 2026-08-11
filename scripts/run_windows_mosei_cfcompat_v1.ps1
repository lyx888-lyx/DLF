param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Cache", "Train")]
    [string]$Action,

    [Parameter(Mandatory=$true)]
    [ValidateSet(1111,1112,1113,1114,1115)]
    [int]$Seed,

    [int]$GpuId = 0,

    [ValidateSet("highest", "high", "medium")]
    [string]$MatmulPrecision = "high",

    [int]$NumWorkers = 0,

    [switch]$Smoke,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

if ($NumWorkers -ne 0) {
    throw "MOSEI CFCompat Windows runner fixes NumWorkers=0."
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    $pythonArgs = @(
        ".\train_mosei_cfcompat_v1_hotfix.py",
        "--action", $Action.ToLowerInvariant(),
        "--seed", [string]$Seed,
        "--gpu-id", [string]$GpuId,
        "--num-workers", [string]$NumWorkers,
        "--matmul-precision", $MatmulPrecision
    )

    if ($Smoke) {
        $pythonArgs += "--smoke-test"
    }
    if ($Overwrite) {
        $pythonArgs += "--overwrite"
    }

    Write-Host "================ MOSEI CFCompat-v1 Windows runner ================="
    Write-Host ("Action:            {0}" -f $Action)
    Write-Host ("Seed:              {0}" -f $Seed)
    Write-Host ("GPU:               {0}" -f $GpuId)
    Write-Host ("NumWorkers:        {0}" -f $NumWorkers)
    Write-Host ("MatmulPrecision:   {0}" -f $MatmulPrecision)
    Write-Host ("Smoke:             {0}" -f [bool]$Smoke)
    Write-Host ("Overwrite:         {0}" -f [bool]$Overwrite)
    Write-Host "Official Test:     WILL NOT BE CONSTRUCTED"
    Write-Host "==================================================================="

    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "MOSEI CFCompat-v1 $Action seed $Seed failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
