param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-objective-audit-v11"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$required = @(
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_fold_manifest.csv",
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_fold_assignment.csv",
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_valid_screen_summary.json"
)
for ($fold = 0; $fold -lt 5; $fold++) {
    $required += ".\pt\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\seed1113\fold$fold\conservative_train_holdout_bank.pth"
}
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen v10 audit input missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_objective_audit_utils.py `
    .\audit_cfcompat_objective_train_oof_v11.py `
    .\smoke_test_cfcompat_objective_audit.py

python .\smoke_test_cfcompat_objective_audit.py

$argsList = @(
    ".\audit_cfcompat_objective_train_oof_v11.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList

$root = ".\result\missing_baseline\cfcompat_objective_audit_v11\mosi\train_oof\seed1113_dev"
$summary = Join-Path $root "objective_audit_v11_summary.json"
$headline = Join-Path $root "objective_audit_v11_headline_gradient_matrix.csv"
$eventSummary = Join-Path $root "objective_audit_v11_event_summary.csv"

Write-Host ""
Write-Host "================ v11 objective audit summary ================"
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v11 headline gradient matrix ==============="
Import-Csv $headline | Sort-Object OOFGroup, TrainComponent | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v11 OOF branch summary ====================="
Import-Csv $eventSummary | Where-Object { $_.group_family -in @("BRANCH", "BRANCH_X_TEACHER", "BRANCH_X_S0") } | Format-Table -AutoSize

Write-Host ""
Write-Host "Complete. This was a pure diagnostic audit: no new model training or checkpoint selection."
Write-Host "Official Test was never constructed or accessed."
Write-Host "Result root: $root"
