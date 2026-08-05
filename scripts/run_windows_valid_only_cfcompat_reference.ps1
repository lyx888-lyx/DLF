[CmdletBinding()]
param(
    [ValidateSet(1112, 1113, 1115)]
    [int]$Seed = 1112,

    [switch]$Formal,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("Python command failed with exit code {0}: python {1}" -f $LASTEXITCODE, ($PythonArgs -join ' '))
    }
}

$PreviousPythonWarnings = $env:PYTHONWARNINGS
$env:PYTHONWARNINGS = "ignore::FutureWarning"

try {
    Write-Host "Project: $ProjectRoot"
    Write-Host "Branch:  $(git branch --show-current)"
    Write-Host "Commit:  $(git rev-parse HEAD)"
    Write-Host "Stage:   Original CFCompatKD Valid-only reference"
    Write-Host "Seed:    $Seed"

    $RequiredPaths = @(
        ".\pt\DLF_mosi_seed${Seed}_best.pth",
        ".\pt\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\DLF_mosi_seed${Seed}_best_valid.pth",
        ".\result\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\mosi_per_seed.csv",
        ".\result\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\manifest.json",
        ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\train_counterfactual_compatibility.csv",
        ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\cf_compat_config.json"
    )
    foreach ($RequiredPath in $RequiredPaths) {
        if (-not (Test-Path $RequiredPath)) {
            throw "Required CFCompat reference asset is absent: $RequiredPath"
        }
    }

    $compileArgs = @(
        "-m", "py_compile",
        ".\rebuild_windows_valid_only_cfcompat_reference.py"
    )
    Invoke-CheckedPython -PythonArgs $compileArgs

    $preflightArgs = @(
        ".\rebuild_windows_valid_only_prerequisites.py",
        "--action", "preflight",
        "--seed", "$Seed",
        "--gpu-ids", "0",
        "--num-workers", "1"
    )
    Invoke-CheckedPython -PythonArgs $preflightArgs

    $runArgs = @(
        ".\rebuild_windows_valid_only_cfcompat_reference.py",
        "--seed", "$Seed",
        "--gpu-ids", "0",
        "--num-workers", "1"
    )
    if (-not $Formal) {
        $runArgs += "--smoke-test"
    }
    if ($Overwrite) {
        $runArgs += "--overwrite"
    }
    Invoke-CheckedPython -PythonArgs $runArgs
}
finally {
    if ($null -eq $PreviousPythonWarnings) {
        Remove-Item Env:PYTHONWARNINGS -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONWARNINGS = $PreviousPythonWarnings
    }
}
