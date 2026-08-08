param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-gradient-surgery-v12"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$required = @(
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_valid_screen_summary.json",
    ".\result\missing_baseline\cfcompat_objective_audit_v11\mosi\train_oof\seed1113_dev\objective_audit_v11_summary.json"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen mechanism/reference artifact missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_gradient_surgery_utils.py `
    .\train_cfcompat_gradient_surgery_valid_screen_v12.py `
    .\smoke_test_cfcompat_gradient_surgery.py

python .\smoke_test_cfcompat_gradient_surgery.py

$argsList = @(
    ".\train_cfcompat_gradient_surgery_valid_screen_v12.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList

$root = ".\result\missing_baseline\cfcompat_gradient_surgery_v12\mosi\valid_screen\seed1113_dev"
$summary = Join-Path $root "gradient_surgery_v12_valid_screen_summary.json"
$transfer = Join-Path $root "gradient_surgery_v12_transfer_summary.csv"
$manifest = Join-Path $root "gradient_surgery_v12_fold_manifest.csv"
$windows = Join-Path $root "gradient_surgery_v12_update_windows.csv"
$consensus = Join-Path $root "gradient_surgery_v12_valid_consensus_events.csv"

Write-Host ""
Write-Host "================ v12 mechanism summary =================="
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v12 transfer summary ==================="
Import-Csv $transfer | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12 fold manifest ======================"
Import-Csv $manifest | Select-Object `
    Fold, `
    TrainEpochCount, `
    AbsoluteBestTrainHoldoutEpoch, `
    ConservativeSelectedEpoch, `
    AbsoluteBestTrainHoldoutJ, `
    ConservativeSelectedTrainHoldoutJ, `
    surgery_conflict_window_fraction, `
    surgery_mean_conflict_cosine, `
    surgery_mean_removed_fraction_on_conflict | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12 gradient-surgery diagnostics ======="
$rows = Import-Csv $windows
$conflicts = $rows | Where-Object { $_.conflict -eq "True" }
$conflictRate = $conflicts.Count / [double]$rows.Count
$meanCos = (($conflicts | ForEach-Object { [double]$_.gradient_cosine_before }) | Measure-Object -Average).Average
$meanRemoved = (($conflicts | ForEach-Object { [double]$_.supervised_l2_removed_fraction }) | Measure-Object -Average).Average
$minPostDot = (($rows | ForEach-Object { [double]$_.post_projection_dot }) | Measure-Object -Minimum).Minimum
Write-Host ("conflict window rate:                {0:P2}" -f $conflictRate)
Write-Host ("mean cosine on conflicts:            {0:N6}" -f $meanCos)
Write-Host ("mean supervised L2 removed/conflict: {0:P2}" -f $meanRemoved)
Write-Host ("minimum post-projection dot:          {0:N9}" -f $minPostDot)

Write-Host ""
Write-Host "================ v12 consensus diagnostics =============="
$consensusRows = Import-Csv $consensus
$missing = $consensusRows | Where-Object { $_.Mode -in @("LA", "LV", "L") }
$applyRate = (($missing | Where-Object { $_.consensus_applied -eq "True" }).Count) / [double]$missing.Count
$meanAgree = (($missing | ForEach-Object { [double]$_.consensus_sign_agreement }) | Measure-Object -Average).Average
$meanStd = (($missing | ForEach-Object { [double]$_.consensus_fold_std }) | Measure-Object -Average).Average
$meanAbsDelta = (($missing | ForEach-Object { [double]$_.abs_consensus_delta }) | Measure-Object -Average).Average
Write-Host ("consensus applied rate: {0:P2}" -f $applyRate)
Write-Host ("mean sign agreement:   {0:N6}" -f $meanAgree)
Write-Host ("mean fold std:         {0:N6}" -f $meanStd)
Write-Host ("mean abs delta:        {0:N6}" -f $meanAbsDelta)

Write-Host ""
Write-Host "Complete. v12 changed only residual optimizer gradient geometry relative to v10."
Write-Host "Fold selection used only Train video holdouts. Official Test was never constructed or accessed."
Write-Host "Result root: $root"
