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
    Write-Host "Stage:   Student-safe dynamic utility + residual CFCompat v3"
    Write-Host "Mode:    $(if ($Formal) { 'formal 3-seed x 4-run grid' } else { 'seed1112 real-chain smoke' })"
    Write-Host "Test:    forbidden"

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
            throw "Required student-safe utility v3 asset is absent: $RequiredPath"
        }
    }

    Invoke-CheckedPython -PythonArgs @(
        "-m", "py_compile",
        ".\trains\singleTask\cfcompat_student_safe_utility_utils.py",
        ".\train_cfcompat_student_safe_utility_valid_screen.py",
        ".\audit_cfcompat_student_safe_utility_valid_screen.py",
        ".\smoke_test_cfcompat_student_safe_utility.py"
    )

    Invoke-CheckedPython -PythonArgs @(
        ".\rebuild_windows_valid_only_prerequisites.py",
        "--action", "preflight",
        "--seed", "1112",
        "--gpu-ids", "0",
        "--num-workers", "1"
    )

    Invoke-CheckedPython -PythonArgs @(
        ".\smoke_test_cfcompat_student_safe_utility.py"
    )

    $runArgs = @(
        ".\train_cfcompat_student_safe_utility_valid_screen.py",
        "--dataset", "mosi",
        "--gpu-ids", "0",
        "--num-workers", "1",
        "--result-root", "result",
        "--model-save-dir", "pt",
        "--log-dir", "log\windows_valid_only_prereq_v1"
    )
    if (-not $Formal) {
        $runArgs += "--smoke-test"
    }
    if ($Overwrite) {
        $runArgs += "--overwrite"
    }
    Invoke-CheckedPython -PythonArgs $runArgs

    if (-not $Formal) {
        $SmokeGridPath = ".\result\missing_baseline\cfcompat_student_safe_utility_v3\mosi\valid_screen\smoke\student_safe_utility_v3_valid_grid_summary.csv"
        if (Test-Path $SmokeGridPath) {
            Write-Host "`n================ v3 smoke grid ================"
            Import-Csv $SmokeGridPath |
                Select-Object Seed, Run, BestValidEpoch, J_valid, `
                    projection_active_fraction, projection_abstain_fraction, `
                    projection_mean_utility, projection_mean_difficulty, `
                    projection_mean_final_gate |
                Format-Table -AutoSize
        }
        return
    }

    $OutputDir = ".\result\missing_baseline\cfcompat_student_safe_utility_v3\mosi\valid_screen"
    Invoke-CheckedPython -PythonArgs @(
        ".\audit_cfcompat_student_safe_utility_valid_screen.py",
        "--result-dir", $OutputDir
    )

    $SummaryPath = Join-Path $OutputDir "student_safe_utility_v3_valid_screen_summary.json"
    $GridPath = Join-Path $OutputDir "student_safe_utility_v3_valid_grid_summary.csv"
    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    $Grid = Import-Csv $GridPath | Select-Object `
        Seed, Run, BestValidEpoch, J_valid, `
        valid_LAV_MAE, valid_LA_MAE, valid_LV_MAE, valid_L_MAE, `
        projection_active_fraction, projection_abstain_fraction, `
        projection_mean_utility, projection_mean_difficulty, `
        projection_mean_final_gate, `
        projection_active_compatibility_utility_spearman

    Write-Host "`n================ Student-safe utility v3 grid ================"n    $Grid | Format-Table -AutoSize
    Write-Host "`n================ Candidate gates ================"n    foreach ($Run in @(
        "student_safe_uniform",
        "student_safe_utility",
        "student_safe_utility_residual_cfcompat"
    )) {
        $Gate = $Summary.candidate_gates.$Run
        Write-Host ("{0} passed: {1}; mean J degradation: {2}; mean harmful reduction: {3}" -f `
            $Run, $Gate.passed, $Gate.mean_J_degradation_vs_CFCompatKD, `
            $Gate.mean_harmful_imitation_reduction)
    }
    Write-Host ("verdict: {0}" -f $Summary.verdict)
    Write-Host ("report:  {0}" -f (Join-Path $OutputDir "student_safe_utility_v3_valid_screen_report.md"))
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
