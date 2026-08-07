[CmdletBinding()]
param(
    [string]$TrainGateCsv = ".\result\missing_baseline\cfcompat_crossfit_transfer_risk_v5\mosi\valid_screen\seed1113_dev\gate_only\crossfit_transfer_risk_v5_train_oof_gate.csv",
    [string]$ValidGateCsv = ".\result\missing_baseline\cfcompat_crossfit_transfer_risk_v5\mosi\valid_screen\seed1113_dev\gate_only\crossfit_transfer_risk_v5_valid_gate_diagnostic.csv",
    [string]$OutputDir = ".\result\missing_baseline\cfcompat_split_shift_failure_diagnostic_v5p1\mosi\seed1113_train_valid",
    [int]$MinRuleSupport = 20,
    [int]$TopKFailures = 100,
    [double]$NearZeroLabelThreshold = 0.5,
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

$Branch = (git branch --show-current).Trim()
$Commit = (git rev-parse HEAD).Trim()
Write-Host "Project: $ProjectRoot"
Write-Host "Branch:  $Branch"
Write-Host "Commit:  $Commit"
Write-Host "Stage:   CFCompatKD v5.1 Train->Valid split-shift failure diagnostic"
Write-Host "Seed:    1113 frozen v5 gate artifacts"
Write-Host "Test:    forbidden / not constructed"
Write-Host "Train:   $TrainGateCsv"
Write-Host "Valid:   $ValidGateCsv"
Write-Host "Output:  $OutputDir"

if ($Branch -ne "agent/cfcompat-split-shift-failure-diagnostic-v5p1") {
    throw "Run this diagnostic only from agent/cfcompat-split-shift-failure-diagnostic-v5p1; current branch: $Branch"
}
if (-not (Test-Path $TrainGateCsv)) {
    throw "Frozen v5 Train OOF gate CSV is absent: $TrainGateCsv"
}
if (-not (Test-Path $ValidGateCsv)) {
    throw "Frozen v5 Valid gate diagnostic CSV is absent: $ValidGateCsv"
}

Invoke-CheckedPython -PythonArgs @(
    "-m", "py_compile",
    ".\analyze_cfcompat_split_shift_failure_v5p1.py",
    ".\smoke_test_cfcompat_split_shift_failure.py"
)
Invoke-CheckedPython -PythonArgs @(".\smoke_test_cfcompat_split_shift_failure.py")

$RunArgs = @(
    ".\analyze_cfcompat_split_shift_failure_v5p1.py",
    "--train-gate-csv", $TrainGateCsv,
    "--valid-gate-csv", $ValidGateCsv,
    "--output-dir", $OutputDir,
    "--near-zero-label-threshold", ([string]$NearZeroLabelThreshold),
    "--min-rule-support", ([string]$MinRuleSupport),
    "--top-k-failures", ([string]$TopKFailures)
)
if ($Overwrite) { $RunArgs += "--overwrite" }
Invoke-CheckedPython -PythonArgs $RunArgs

$SummaryPath = Join-Path $OutputDir "split_shift_failure_summary.json"
$ModePath = Join-Path $OutputDir "mode_failure_summary.csv"
$ShiftPath = Join-Path $OutputDir "feature_shift_summary.csv"
$RelationshipPath = Join-Path $OutputDir "benefit_relationship_shift.csv"
$RulePath = Join-Path $OutputDir "failure_interaction_rules.csv"

$Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
Write-Host "`n================ v5.1 split-shift diagnostic ================"
Write-Host ("Train OOF gate error: {0:N4}" -f [double]$Summary.gate_error_rates.train_oof)
Write-Host ("Valid gate error:     {0:N4}" -f [double]$Summary.gate_error_rates.valid_full_train)
Write-Host ("Delta Valid-Train:   {0:N4}" -f [double]$Summary.gate_error_rates.delta_valid_minus_train)
Write-Host "Test accessed:        $($Summary.test_accessed)"

Write-Host "`n================ Mode-level gate failures ================"
Import-Csv $ModePath |
    Format-Table split, mode, N, beneficial_prevalence, gate_misclassification_rate, `
        false_positive_rate_among_nonbeneficial, false_negative_rate_among_beneficial -AutoSize

Write-Host "`n================ Top feature distribution shifts ================"
Import-Csv $ShiftPath |
    Select-Object -First 12 mode, feature, standardized_mean_difference, ks_statistic, train_mean, valid_mean |
    Format-Table -AutoSize

Write-Host "`n================ Top relationship instabilities ================"
Import-Csv $RelationshipPath |
    Select-Object -First 12 mode, feature, train_spearman_vs_teacher_advantage, `
        valid_spearman_vs_teacher_advantage, spearman_sign_flip, train_univariate_auc, `
        valid_univariate_auc, auc_orientation_flip |
    Format-Table -AutoSize

Write-Host "`n================ Top Valid-enriched failure rules ================"
Import-Csv $RulePath |
    Where-Object { $_.valid_support -ne "" } |
    Sort-Object {[double]$_.valid_misclassification_lift_vs_split} -Descending |
    Select-Object -First 12 rule, valid_support, valid_gate_misclassification_rate, `
        valid_misclassification_lift_vs_split, train_gate_misclassification_rate, `
        valid_minus_train_misclassification_rate |
    Format-Table -AutoSize

Write-Host "Diagnostic complete. No Student training and no Test access occurred."
