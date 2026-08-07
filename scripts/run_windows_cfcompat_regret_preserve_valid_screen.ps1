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
    $Branch = (git branch --show-current).Trim()
    $Commit = (git rev-parse HEAD).Trim()
    Write-Host "Project: $ProjectRoot"
    Write-Host "Branch:  $Branch"
    Write-Host "Commit:  $Commit"
    Write-Host "Stage:   Regret-Aware Preserve-or-Distill CFCompatKD v4"
    Write-Host "Seed:    1113 only"
    Write-Host "Mode:    $(if ($Formal) { 'single-seed full development trajectory' } else { 'single-seed 2-epoch real-chain smoke' })"
    Write-Host "Test:    forbidden"

    if ($Branch -ne "feature/cfcompat-regret-preserve-distill-valid-screen-v4") {
        throw "Run v4 only from feature/cfcompat-regret-preserve-distill-valid-screen-v4; current branch: $Branch"
    }

    $Seed = 1113
    $RequiredPaths = @(
        ".\pt\DLF_mosi_seed${Seed}_best.pth",
        ".\result\windows_valid_only_prereq_v1\clean_dlf\seed${Seed}\manifest.json",
        ".\pt\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\DLF_mosi_seed${Seed}_best_valid.pth",
        ".\result\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\mosi_per_seed.csv",
        ".\result\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\manifest.json",
        ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\train_counterfactual_compatibility.csv",
        ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\cf_compat_config.json",
        ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\manifest.json",
        ".\result\missing_baseline\cf_compat_kd_v1\benchmark_multiseed\seed${Seed}\mosi_per_seed.csv",
        ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_valid_grid_summary.csv",
        ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_raw_valid_events.csv",
        ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_valid_screen_summary.json",
        ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_source_manifest.json",
        ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_audit_check.json"
    )
    foreach ($RequiredPath in $RequiredPaths) {
        if (-not (Test-Path $RequiredPath)) {
            throw "Required v4 asset is absent: $RequiredPath"
        }
    }

    Invoke-CheckedPython -PythonArgs @(
        "-m", "py_compile",
        ".\trains\singleTask\cfcompat_regret_preserve_utils.py",
        ".\train_cfcompat_regret_preserve_valid_screen.py",
        ".\audit_cfcompat_regret_preserve_valid_screen.py",
        ".\smoke_test_cfcompat_regret_preserve.py"
    )

    Invoke-CheckedPython -PythonArgs @(
        ".\rebuild_windows_valid_only_prerequisites.py",
        "--action", "preflight",
        "--seed", "1113",
        "--gpu-ids", "0",
        "--num-workers", "1"
    )

    Invoke-CheckedPython -PythonArgs @(
        ".\smoke_test_cfcompat_regret_preserve.py"
    )

    $RunArgs = @(
        ".\train_cfcompat_regret_preserve_valid_screen.py",
        "--dataset", "mosi",
        "--gpu-ids", "0",
        "--num-workers", "1",
        "--result-root", "result",
        "--model-save-dir", "pt",
        "--log-dir", "log\windows_valid_only_prereq_v1"
    )
    if (-not $Formal) {
        $RunArgs += "--smoke-test"
    }
    if ($Overwrite) {
        $RunArgs += "--overwrite"
    }
    Invoke-CheckedPython -PythonArgs $RunArgs

    $OutputDir = ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev"
    if (-not $Formal) {
        $OutputDir = Join-Path $OutputDir "smoke"
        $GridPath = Join-Path $OutputDir "regret_preserve_v4_candidate_grid.csv"
        if (Test-Path $GridPath) {
            Write-Host "`n================ v4 smoke candidate ================"
            Import-Csv $GridPath |
                Select-Object Seed, Run, BestValidEpoch, J_valid, `
                    projection_distill_fraction, projection_preserve_fraction, `
                    projection_decision_abstain_fraction |
                Format-Table -AutoSize
        }
        return
    }

    Invoke-CheckedPython -PythonArgs @(
        ".\audit_cfcompat_regret_preserve_valid_screen.py",
        "--result-dir", $OutputDir
    )

    $SummaryPath = Join-Path $OutputDir "regret_preserve_v4_valid_screen_summary.json"
    $GridPath = Join-Path $OutputDir "regret_preserve_v4_candidate_grid.csv"
    $NegativePath = Join-Path $OutputDir "regret_preserve_v4_negative_transfer_metrics.csv"
    $Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
    $Grid = Import-Csv $GridPath
    $Negative = Import-Csv $NegativePath | Where-Object { $_.Mode -eq "MISSING_ALL" }

    Write-Host "`n================ v4 candidate ================"
    $Grid |
        Select-Object Seed, Run, BestValidEpoch, J_valid, `
            valid_LAV_MAE, valid_LA_MAE, valid_LV_MAE, valid_L_MAE, `
            projection_distill_fraction, projection_preserve_fraction, `
            projection_decision_abstain_fraction |
        Format-Table -AutoSize

    Write-Host "`n================ Missing-modality negative transfer ================"
    $Negative |
        Select-Object Seed, Run, negative_transfer_rate, `
            severe_negative_transfer_rate, positive_transfer_rate, `
            mean_regret_vs_baseline |
        Format-Table -AutoSize

    Write-Host "`n================ Frozen development checks ================"
    $Summary.candidate_gate.checks.PSObject.Properties |
        Select-Object Name, Value |
        Format-Table -AutoSize
    Write-Host ("verdict: {0}" -f $Summary.verdict)
    Write-Host ("report:  {0}" -f (Join-Path $OutputDir "regret_preserve_v4_valid_screen_report.md"))
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
