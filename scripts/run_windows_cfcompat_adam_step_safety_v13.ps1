param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-adam-step-safety-v13"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$v12Result = ".\result\missing_baseline\cfcompat_gradient_surgery_v12\mosi\valid_screen\seed1113_dev"
$required = @(
    (Join-Path $v12Result "gradient_surgery_v12_candidate_grid.csv"),
    (Join-Path $v12Result "gradient_surgery_v12_candidate_raw_valid_events.csv"),
    (Join-Path $v12Result "gradient_surgery_v12_valid_screen_summary.json"),
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev\conservative_crossfit_v10_valid_screen_summary.json",
    ".\result\missing_baseline\cfcompat_window_actual_step_audit_v12p3\mosi\train_oof\seed1113_dev\window_v12p3_summary.json"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen v4/v8/v10/v12/v12.3 artifact missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_adam_step_safety_utils.py `
    .\train_cfcompat_adam_step_safety_valid_screen_v13.py `
    .\smoke_test_cfcompat_adam_step_safety.py
if ($LASTEXITCODE -ne 0) {
    throw "v13 py_compile failed with exit code $LASTEXITCODE"
}

# Synthetic only: no dataset loader and therefore no extra Valid observation.
python .\smoke_test_cfcompat_adam_step_safety.py
if ($LASTEXITCODE -ne 0) {
    throw "v13 synthetic safety smoke failed with exit code $LASTEXITCODE"
}

$argsList = @(
    ".\train_cfcompat_adam_step_safety_valid_screen_v13.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "v13 formal run failed with exit code $LASTEXITCODE; result printing aborted"
}

$root = ".\result\missing_baseline\cfcompat_adam_step_safety_v13\mosi\valid_screen\seed1113_dev"
$summary = Join-Path $root "adam_step_safety_v13_valid_screen_summary.json"
$transfer = Join-Path $root "adam_step_safety_v13_transfer_summary.csv"
$manifest = Join-Path $root "adam_step_safety_v13_fold_manifest.csv"
$safety = Join-Path $root "adam_step_safety_v13_actual_step_safety_windows.csv"
$sentinel = Join-Path $root "adam_step_safety_v13_sentinel_manifest.csv"

foreach ($path in @($summary, $transfer, $manifest, $safety, $sentinel)) {
    if (-not (Test-Path $path)) {
        throw "v13 completed without required result artifact: $path"
    }
}

Write-Host ""
Write-Host "================ v13 metric summary ============================"
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v13 transfer summary =========================="
Import-Csv $transfer | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v13 fold manifest ============================="
Import-Csv $manifest |
    Select-Object Fold, TrainEpochCount, AbsoluteBestTrainHoldoutEpoch, ConservativeSelectedEpoch, ConservativeSelectedTrainHoldoutJ, step_safety_projected_window_fraction, step_safety_mean_removed_delta_l2_fraction, step_safety_mean_raw_surgery_to_adam_effective_cosine |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v13 sentinel manifest ========================="
Import-Csv $sentinel |
    Select-Object Fold, Group, Mode, N, TrainModeEventN, Prevalence, SelectionMargin |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v13 actual-step safety ========================"
$rows = Import-Csv $safety
$windows = $rows | Group-Object Fold, Epoch, UpdateWindow
$projectedWindows = 0
$removed = @()
$rawAdam = @()
foreach ($window in $windows) {
    $first = $window.Group[0]
    if ($first.projected_any -eq "True") { $projectedWindows++ }
    $removed += [double]$first.removed_delta_l2_fraction_global
    $rawAdam += [double]$first.raw_surgery_to_adam_effective_cosine
}
Write-Host ("optimizer windows:                         {0}" -f $windows.Count)
Write-Host ("actual-step projected window rate:        {0:P2}" -f ($projectedWindows / [double]$windows.Count))
Write-Host ("mean removed Adam displacement fraction:  {0:P2}" -f (($removed | Measure-Object -Average).Average))
Write-Host ("mean raw-surgery vs Adam-effective cosine: {0:N6}" -f (($rawAdam | Measure-Object -Average).Average))
foreach ($mode in @("LA", "LV", "L")) {
    $local = $rows | Where-Object { $_.Mode -eq $mode }
    $rate = (($local | Where-Object { $_.projected -eq "True" }).Count) / [double]$local.Count
    Write-Host ("projection rate {0}: {1:P2}" -f $mode, $rate)
}

Write-Host ""
Write-Host "================ v13 decision ================================"
$data = Get-Content $summary -Raw | ConvertFrom-Json
Write-Host ("candidate J:                         {0}" -f $data.candidate_J)
Write-Host ("frozen v12 J:                       {0}" -f $data.frozen_v12_J)
Write-Host ("direct metric gate vs v12 passed:   {0}" -f $data.direct_v12_metric_improvement_gate.passed)
Write-Host ("legacy v8/v4 target gate passed:    {0}" -f $data.legacy_v8_v4_target_gate.passed)
Write-Host ("verdict:                             {0}" -f $data.verdict)
Write-Host ""
Write-Host "Direct metric gate was frozen before this Valid run: J, Teacher-beneficial NTR, and overall NTR must strictly improve over v12; Teacher-nonbeneficial NTR may degrade by at most 3pp."
Write-Host "Legacy target gate is unchanged from v12. Official Valid is first used after all five folds are frozen."
Write-Host "Synthetic smoke does not load Valid. Official Test was never constructed or accessed."
Write-Host "Result root: $root"
