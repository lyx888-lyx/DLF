[CmdletBinding()]
param(
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
$PreviousCublasWorkspace = $env:CUBLAS_WORKSPACE_CONFIG
$env:PYTHONWARNINGS = "ignore::FutureWarning"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

try {
    Write-Host "Project: $ProjectRoot"
    Write-Host "Branch:  $(git branch --show-current)"
    Write-Host "Commit:  $(git rev-parse HEAD)"
    Write-Host "Stage:   Safe-CFCompatKD Valid-only screen"
    Write-Host "Mode:    $(if ($Formal) { 'formal 3-seed x 3-run grid' } else { 'seed1112 real-chain smoke' })"

    $RequiredPaths = @()
    foreach ($Seed in @(1112, 1113, 1115)) {
        $RequiredPaths += @(
            ".\pt\DLF_mosi_seed${Seed}_best.pth",
            ".\result\windows_valid_only_prereq_v1\clean_dlf\seed${Seed}\manifest.json",
            ".\pt\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\DLF_mosi_seed${Seed}_best_valid.pth",
            ".\result\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\mosi_per_seed.csv",
            ".\result\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\manifest.json",
            ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\train_counterfactual_compatibility.csv",
            ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\cf_compat_config.json",
            ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\manifest.json",
            ".\pt\missing_baseline\cf_compat_kd_v1\benchmark_multiseed\seed${Seed}\DLF_mosi_seed${Seed}_best_valid.pth",
            ".\result\missing_baseline\cf_compat_kd_v1\benchmark_multiseed\seed${Seed}\mosi_per_seed.csv",
            ".\result\missing_baseline\cf_compat_kd_v1\benchmark_multiseed\seed${Seed}\manifest.json"
        )
    }
    foreach ($RequiredPath in $RequiredPaths) {
        if (-not (Test-Path $RequiredPath)) {
            throw "Required Safe-CFCompat asset is absent: $RequiredPath"
        }
    }

    $compileArgs = @(
        "-m", "py_compile",
        ".\trains\singleTask\cfcompat_safe_projection_utils.py",
        ".\train_cfcompat_safe_projection_valid_screen.py",
        ".\train_cfcompat_safe_projection_valid_screen_v2.py",
        ".\train_cfcompat_safe_projection_valid_screen_v3.py",
        ".\audit_cfcompat_safe_projection_valid_screen.py",
        ".\smoke_test_cfcompat_safe_projection.py",
        ".\smoke_windows_safe_cfcompat_valid_screen.py"
    )
    Invoke-CheckedPython -PythonArgs $compileArgs

    $preflightArgs = @(
        ".\rebuild_windows_valid_only_prerequisites.py",
        "--action", "preflight",
        "--seed", "1112",
        "--gpu-ids", "0",
        "--num-workers", "1"
    )
    Invoke-CheckedPython -PythonArgs $preflightArgs

    if (-not $Formal) {
        $smokeArgs = @(
            ".\smoke_windows_safe_cfcompat_valid_screen.py",
            "--gpu-ids", "0",
            "--num-workers", "1"
        )
        if ($Overwrite) {
            $smokeArgs += "--overwrite"
        }
        Invoke-CheckedPython -PythonArgs $smokeArgs
        return
    }

    $OutputDir = ".\result\missing_baseline\cfcompat_safe_projection_v1\mosi\valid_screen"
    $ModelDir = ".\pt\missing_baseline\cfcompat_safe_projection_v1\mosi\valid_screen"
    if ((Test-Path $OutputDir) -or (Test-Path $ModelDir)) {
        if (-not $Overwrite) {
            throw "Formal Safe-CFCompat output already exists. Inspect it or rerun with -Overwrite."
        }
        Remove-Item $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item $ModelDir -Recurse -Force -ErrorAction SilentlyContinue
    }

    Invoke-CheckedPython -PythonArgs @(".\smoke_test_cfcompat_safe_projection.py")

    $formalArgs = @(
        ".\train_cfcompat_safe_projection_valid_screen_v3.py",
        "--dataset", "mosi",
        "--seeds", "1112", "1113", "1115",
        "--gpu-ids", "0",
        "--num-workers", "1",
        "--result-root", "result",
        "--model-save-dir", "pt",
        "--log-dir", "log\windows_valid_only_prereq_v1"
    )
    Invoke-CheckedPython -PythonArgs $formalArgs

    Invoke-CheckedPython -PythonArgs @(
        ".\audit_cfcompat_safe_projection_valid_screen.py",
        "--result-dir", $OutputDir
    )

    $SummaryPath = Join-Path $OutputDir "safe_projection_valid_screen_summary.json"
    $GridPath = Join-Path $OutputDir "safe_projection_valid_grid_summary.csv"
    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    $Grid = Import-Csv $GridPath | Select-Object Seed, Run, BestValidEpoch, J_valid, valid_LAV_MAE, valid_LA_MAE, valid_LV_MAE, valid_L_MAE

    Write-Host "`n================ Safe-CFCompat grid ================"
    $Grid | Format-Table -AutoSize
    Write-Host "`n================ Candidate gates ================"
    Write-Host ("safe_uniform passed:  {0}; mean gain: {1}" -f $Summary.candidate_gates.safe_uniform.passed, $Summary.candidate_gates.safe_uniform.mean_gain_valid_J_vs_CFCompatKD)
    Write-Host ("safe_cfcompat passed: {0}; mean gain: {1}" -f $Summary.candidate_gates.safe_cfcompat.passed, $Summary.candidate_gates.safe_cfcompat.mean_gain_valid_J_vs_CFCompatKD)
    Write-Host ("verdict: {0}" -f $Summary.verdict)
    Write-Host ("report:  {0}" -f (Join-Path $OutputDir "safe_projection_valid_screen_report.md"))
    Write-Host "official Test was not constructed"
}
finally {
    if ($null -eq $PreviousPythonWarnings) {
        Remove-Item Env:PYTHONWARNINGS -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONWARNINGS = $PreviousPythonWarnings
    }
    if ($null -eq $PreviousCublasWorkspace) {
        Remove-Item Env:CUBLAS_WORKSPACE_CONFIG -ErrorAction SilentlyContinue
    }
    else {
        $env:CUBLAS_WORKSPACE_CONFIG = $PreviousCublasWorkspace
    }
}
