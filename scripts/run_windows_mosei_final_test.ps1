param(
    [int]$GpuId = 0,

    [ValidateSet("highest", "high", "medium")]
    [string]$MatmulPrecision = "high",

    [int]$NumWorkers = 0,

    [switch]$Preflight,
    [switch]$ExecuteFinalTest
)

$ErrorActionPreference = "Stop"

if ($NumWorkers -ne 0) {
    throw "MOSEI final Test Windows runner fixes NumWorkers=0."
}
if ([bool]$Preflight -eq [bool]$ExecuteFinalTest) {
    throw "Choose exactly one mode: -Preflight OR -ExecuteFinalTest."
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    $pythonArgs = @(
        ".\evaluate_mosei_fixedblend_dp57_final_test_hotfix.py",
        "--gpu-id", [string]$GpuId,
        "--num-workers", [string]$NumWorkers,
        "--matmul-precision", $MatmulPrecision
    )

    if ($Preflight) {
        $pythonArgs += "--preflight"
    }
    if ($ExecuteFinalTest) {
        $pythonArgs += "--execute-final-test"
    }

    Write-Host "================ MOSEI FixedBlend-DP57 FINAL TEST ================="
    Write-Host ("Mode:              {0}" -f $(if ($Preflight) { "PREFLIGHT — NO TEST" } else { "EXECUTE ONE-SHOT FINAL TEST" }))
    Write-Host ("GPU:               {0}" -f $GpuId)
    Write-Host ("NumWorkers:        {0}" -f $NumWorkers)
    Write-Host ("MatmulPrecision:   {0}" -f $MatmulPrecision)
    Write-Host "Raw5:              seeds 1111-1115, equal 0.2"
    Write-Host "v13:               formal frozen seed1113 consensus"
    Write-Host "Blend:             Raw5 0.5 + v13 0.5 (frozen; NO search)"
    Write-Host "DP57 anchor:       seed1114 (Valid-selected; frozen)"
    Write-Host "DP57 preserves:    Acc7 + Acc5"
    Write-Host "Test outputs:      aggregate metrics/projection summary ONLY"
    Write-Host "Sample predictions:WILL NOT BE WRITTEN"
    Write-Host "Rerun/overwrite:   DISABLED BY CODE"
    if ($ExecuteFinalTest) {
        Write-Host "Official Test:     WILL BE CONSTRUCTED ONCE AFTER ALL PREFLIGHT CHECKS"
    }
    else {
        Write-Host "Official Test:     WILL NOT BE CONSTRUCTED"
    }
    Write-Host "===================================================================="

    & python @pythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "MOSEI final Test runner failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
