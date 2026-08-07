[CmdletBinding()]
param(
    [switch]$GateOnly,
    [switch]$Formal,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if ($GateOnly -and $Formal) {
    throw "Choose at most one of -GateOnly / -Formal."
}

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("Python command failed with exit code {0}: python {1}" -f $LASTEXITCODE, ($PythonArgs -join ' '))
    }
}

$Branch = (git branch --show-current).Trim()
$Commit = (git rev-parse HEAD).Trim()
Write-Host "Project: $ProjectRoot"
Write-Host "Branch:  $Branch"
Write-Host "Commit:  $Commit"
Write-Host "Stage:   Cross-Fitted Transfer-Risk CFCompatKD v5"
Write-Host "Seed:    1113 only"
Write-Host "Test:    forbidden"
Write-Host "Mode:    $(if ($GateOnly) { 'cheap gate-only pre-screen' } elseif ($Formal) { 'single full Seed1113 trajectory' } else { '2-epoch real-chain smoke' })"

if ($Branch -ne "feature/cfcompat-crossfit-transfer-risk-gate-v5") {
    throw "Run v5 only from feature/cfcompat-crossfit-transfer-risk-gate-v5; current branch: $Branch"
}

$Seed = 1113
$RequiredPaths = @(
    ".\pt\DLF_mosi_seed${Seed}_best.pth",
    ".\result\windows_valid_only_prereq_v1\clean_dlf\seed${Seed}\manifest.json",
    ".\pt\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\DLF_mosi_seed${Seed}_best_valid.pth",
    ".\result\missing_baseline\moddrop_benchmark_multiseed_v1\seed${Seed}\mosi_per_seed.csv",
    ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\train_counterfactual_compatibility.csv",
    ".\result\counterfactual_compatibility\cf_compat_v1_multiseed\mosi\seed${Seed}\cf_compat_config.json",
    ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_valid_grid_summary.csv",
    ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_audit_check.json",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_audit_check.json"
)
foreach ($RequiredPath in $RequiredPaths) {
    if (-not (Test-Path $RequiredPath)) {
        throw "Required v5 asset is absent: $RequiredPath"
    }
}

Invoke-CheckedPython -PythonArgs @(
    "-m", "py_compile",
    ".\trains\singleTask\cfcompat_crossfit_transfer_risk_utils.py",
    ".\train_cfcompat_crossfit_transfer_risk_valid_screen.py",
    ".\audit_cfcompat_crossfit_transfer_risk_valid_screen.py",
    ".\smoke_test_cfcompat_crossfit_transfer_risk.py"
)

Invoke-CheckedPython -PythonArgs @(
    ".\rebuild_windows_valid_only_prerequisites.py",
    "--action", "preflight",
    "--seed", "1113",
    "--gpu-ids", "0",
    "--num-workers", "1"
)

Invoke-CheckedPython -PythonArgs @(".\smoke_test_cfcompat_crossfit_transfer_risk.py")

$RunArgs = @(
    ".\train_cfcompat_crossfit_transfer_risk_valid_screen.py",
    "--dataset", "mosi",
    "--gpu-ids", "0",
    "--num-workers", "1",
    "--result-root", "result",
    "--model-save-dir", "pt",
    "--log-dir", "log\windows_valid_only_prereq_v1"
)
if ($GateOnly) { $RunArgs += "--gate-only" }
elseif (-not $Formal) { $RunArgs += "--smoke-test" }
if ($Overwrite) { $RunArgs += "--overwrite" }
Invoke-CheckedPython -PythonArgs $RunArgs

$OutputDir = ".\result\missing_baseline\cfcompat_crossfit_transfer_risk_v5\mosi\valid_screen\seed1113_dev"
if ($GateOnly) { $OutputDir = Join-Path $OutputDir "gate_only" }
elseif (-not $Formal) { $OutputDir = Join-Path $OutputDir "smoke" }

$GateSummaryPath = Join-Path $OutputDir "crossfit_transfer_risk_v5_gate_summary.json"
if (Test-Path $GateSummaryPath) {
    $Gate = Get-Content $GateSummaryPath -Raw | ConvertFrom-Json
    Write-Host "`n================ v5 risk-gate pre-screen ================"
    Write-Host ("OOF Train AUC: {0:N6}" -f [double]$Gate.train_oof.roc_auc)
    Write-Host ("Valid AUC:     {0:N6}" -f [double]$Gate.valid_full_train_gate.roc_auc)
    Write-Host ("Valid Brier:   {0:N6}" -f [double]$Gate.valid_full_train_gate.brier)
    Write-Host ("Constant:      {0:N6}" -f [double]$Gate.valid_full_train_gate.constant_brier)
    Write-Host ("Prescreen:     {0}" -f $Gate.prescreen_passed)
}

if ($GateOnly) {
    Write-Host "Gate-only run complete; no Student trajectory and no Test were constructed."
    return
}

$GridPath = Join-Path $OutputDir "crossfit_transfer_risk_v5_candidate_grid.csv"
if (-not (Test-Path $GridPath)) {
    Write-Host "No candidate trajectory artifact exists. The formal gate pre-screen stopped before training."
    return
}

if (-not $Formal) {
    Write-Host "`n================ v5 smoke candidate ================"
    Import-Csv $GridPath |
        Select-Object Seed, Run, BestValidEpoch, J_valid, `
            projection_mean_oof_benefit_probability, projection_positive_risk_weight_fraction, `
            projection_safe_teacher_active_fraction, projection_effective_distill_fraction |
        Format-Table -AutoSize
    return
}

Invoke-CheckedPython -PythonArgs @(
    ".\audit_cfcompat_crossfit_transfer_risk_valid_screen.py",
    "--result-dir", $OutputDir
)

$Summary = Get-Content (Join-Path $OutputDir "crossfit_transfer_risk_v5_valid_screen_summary.json") -Raw | ConvertFrom-Json
$Transfer = Import-Csv (Join-Path $OutputDir "crossfit_transfer_risk_v5_transfer_metrics.csv") |
    Where-Object { $_.Mode -eq "MISSING_ALL" }

Write-Host "`n================ v5 candidate ================"
Import-Csv $GridPath |
    Select-Object Seed, Run, BestValidEpoch, J_valid, valid_LAV_MAE, valid_LA_MAE, valid_LV_MAE, valid_L_MAE, `
        projection_mean_oof_benefit_probability, projection_positive_risk_weight_fraction, `
        projection_safe_teacher_active_fraction, projection_effective_distill_fraction |
    Format-Table -AutoSize

Write-Host "`n================ Missing-modality transfer ================"
$Transfer |
    Select-Object Seed, Run, negative_transfer_rate, severe_negative_transfer_rate, positive_transfer_rate, mean_regret_vs_baseline |
    Format-Table -AutoSize

Write-Host "`n================ Frozen v5 development checks ================"
$Summary.candidate_gate.checks.PSObject.Properties |
    Select-Object Name, Value |
    Format-Table -AutoSize
Write-Host ("verdict: {0}" -f $Summary.verdict)
Write-Host "official Test was not constructed"
