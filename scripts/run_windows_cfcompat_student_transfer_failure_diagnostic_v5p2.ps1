[CmdletBinding()]
param([switch]$Overwrite)

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
Write-Host "Project:  $ProjectRoot"
Write-Host "Branch:   $Branch"
Write-Host "Commit:   $Commit"
Write-Host "Stage:    CFCompatKD v5.2 Student-transfer failure diagnostic"
Write-Host "Seed:     1113 frozen Valid artifacts"
Write-Host "Training: forbidden / not performed"
Write-Host "Test:     forbidden / not constructed"

if ($Branch -ne "agent/cfcompat-student-transfer-failure-v5p2") {
    throw "Run v5.2 only from agent/cfcompat-student-transfer-failure-v5p2; current branch: $Branch"
}

$Candidate = ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_raw_valid_events.csv"
$Reference = ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_reference_raw_valid_events.csv"
$Gate = ".\result\missing_baseline\cfcompat_crossfit_transfer_risk_v5\mosi\valid_screen\seed1113_dev\gate_only\crossfit_transfer_risk_v5_valid_gate_diagnostic.csv"
$Output = ".\result\missing_baseline\cfcompat_student_transfer_failure_diagnostic_v5p2\mosi\seed1113_valid"

foreach ($Path in @($Candidate, $Reference, $Gate)) {
    if (-not (Test-Path $Path)) { throw "Required frozen diagnostic input is absent: $Path" }
}

Invoke-CheckedPython -PythonArgs @(
    "-m", "py_compile",
    ".\analyze_cfcompat_student_transfer_failure_v5p2.py",
    ".\smoke_test_cfcompat_student_transfer_failure.py"
)
Invoke-CheckedPython -PythonArgs @(".\smoke_test_cfcompat_student_transfer_failure.py")

$Args = @(
    ".\analyze_cfcompat_student_transfer_failure_v5p2.py",
    "--v4-candidate-csv", $Candidate,
    "--v4-reference-csv", $Reference,
    "--v5-valid-gate-csv", $Gate,
    "--output-dir", $Output
)
if ($Overwrite) { $Args += "--overwrite" }
Invoke-CheckedPython -PythonArgs $Args

$Summary = Get-Content (Join-Path $Output "student_transfer_failure_summary.json") -Raw | ConvertFrom-Json
Write-Host "`n================ v5.2 core counts ================"
Write-Host ("Teacher beneficial events:                    {0}" -f $Summary.counts.teacher_beneficial_events)
Write-Host ("Beneficial Teacher but Student not improved:  {0}" -f $Summary.counts.beneficial_teacher_not_improved_events)
Write-Host ("Beneficial Teacher + negative transfer:        {0}" -f $Summary.counts.beneficial_teacher_negative_transfer_events)
Write-Host ("Beneficial Teacher + severe negative:          {0}" -f $Summary.counts.beneficial_teacher_severe_negative_events)
Write-Host ("v5 TP but Student not improved:                 {0}" -f $Summary.counts.v5_gate_true_positive_but_student_not_improved_events)

Write-Host "`n================ Run-level transfer ================"
Import-Csv (Join-Path $Output "student_transfer_summary.csv") |
    Where-Object { $_.mode -eq "ALL" } |
    Select-Object Run, N, teacher_beneficial_prevalence, positive_transfer_rate, negative_transfer_rate, severe_negative_transfer_rate, beneficial_teacher_negative_transfer_rate |
    Format-Table -AutoSize

Write-Host "`n================ Top beneficial-Teacher failure features ================"
Import-Csv (Join-Path $Output "beneficial_teacher_failure_feature_summary.csv") |
    Select-Object -First 12 mode, feature, success_N, failure_N, failure_minus_success_smd, ks_statistic |
    Format-Table -AutoSize

Write-Host "`n================ Top beneficial-Teacher failure rules ================"
Import-Csv (Join-Path $Output "beneficial_teacher_failure_rules.csv") |
    Select-Object -First 12 rule, support, beneficial_teacher_failure_rate, lift_vs_teacher_helpful, negative_transfer_rate |
    Format-Table -AutoSize

Write-Host "`nDiagnostic complete. No Student training and no Test access occurred."
Write-Host ("Output: {0}" -f $Output)
