param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(1111,1112,1113,1114,1115)]
    [int]$Seed,

    [int]$GpuId = 0,

    [ValidateSet("highest","high","medium")]
    [string]$MatmulPrecision = "high",

    [switch]$Smoke,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    Write-Host "================ MOSEI ModDrop Stage-1 ============================"
    Write-Host "Seed:             $Seed"
    Write-Host "GPU:              $GpuId"
    Write-Host "Matmul precision: $MatmulPrecision"
    Write-Host "Num workers:      0 (Windows-safe frozen setting)"
    Write-Host "Smoke:            $Smoke"
    Write-Host "Overwrite:        $Overwrite"
    Write-Host "Official Test:    NOT CONSTRUCTED"
    Write-Host "=================================================================="

    $cli = @(
        "train_mosei_moddrop_stage1_v1.py",
        "--seed", "$Seed",
        "--gpu-id", "$GpuId",
        "--num-workers", "0",
        "--matmul-precision", $MatmulPrecision
    )

    if ($Smoke) {
        $cli += "--smoke-test"
    }
    if ($Overwrite) {
        $cli += "--overwrite"
    }

    & python @cli
    if ($LASTEXITCODE -ne 0) {
        throw "MOSEI ModDrop Stage-1 seed $Seed failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
