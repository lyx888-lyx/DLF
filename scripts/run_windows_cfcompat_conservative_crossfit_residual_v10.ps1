param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-conservative-crossfit-residual-v10"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$required = @(
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_crossfit_residual_consensus_v9\mosi\valid_screen\seed1113_dev\crossfit_residual_v9_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_crossfit_residual_consensus_v9\mosi\valid_screen\seed1113_dev\crossfit_residual_v9_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_crossfit_residual_consensus_v9\mosi\valid_screen\seed1113_dev\crossfit_residual_v9_valid_screen_summary.json"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen reference missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_conservative_crossfit_residual_utils.py `
    .\train_cfcompat_conservative_crossfit_residual_valid_screen_v10.py `
    .\smoke_test_cfcompat_conservative_crossfit_residual.py

# Reuse the v9 consensus smoke because v10 deliberately leaves consensus unchanged.
python .\smoke_test_cfcompat_crossfit_residual_consensus.py
python .\smoke_test_cfcompat_conservative_crossfit_residual.py

$argsList = @(
    ".\train_cfcompat_conservative_crossfit_residual_valid_screen_v10.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList

$root = ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev"
$summary = Join-Path $root "conservative_crossfit_v10_valid_screen_summary.json"
$transfer = Join-Path $root "conservative_crossfit_v10_transfer_summary.csv"
$manifest = Join-Path $root "conservative_crossfit_v10_fold_manifest.csv"
$consensus = Join-Path $root "conservative_crossfit_v10_valid_consensus_events.csv"

Write-Host ""
Write-Host "================ v10 mechanism summary ==============="
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v10 transfer summary ================="
Import-Csv $transfer | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v10 fold selector manifest ==========="
Import-Csv $manifest | Select-Object `
    Fold, `
    TrainEpochCount, `
    AbsoluteBestTrainHoldoutEpoch, `
    ConservativeSelectedEpoch, `
    AbsoluteBestTrainHoldoutJ, `
    ConservativeSelectedTrainHoldoutJ, `
    SelectedRelativeJDegradation, `
    EpochReductionVsAbsoluteBest, `
    AbsoluteBestHoldoutMeanAbsResidual, `
    ConservativeHoldoutMeanAbsResidual, `
    ConservativeToBestResidualMagnitudeRatio | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v10 consensus diagnostics ============"
$rows = Import-Csv $consensus
$missing = $rows | Where-Object { $_.Mode -in @("LA", "LV", "L") }
$applyRate = (($missing | Where-Object { $_.consensus_applied -eq "True" }).Count) / [double]$missing.Count
$meanAgree = (($missing | ForEach-Object { [double]$_.consensus_sign_agreement }) | Measure-Object -Average).Average
$meanStd = (($missing | ForEach-Object { [double]$_.consensus_fold_std }) | Measure-Object -Average).Average
$meanAbsDelta = (($missing | ForEach-Object { [double]$_.abs_consensus_delta }) | Measure-Object -Average).Average
Write-Host ("consensus applied rate: {0:P2}" -f $applyRate)
Write-Host ("mean sign agreement:   {0:N6}" -f $meanAgree)
Write-Host ("mean fold std:         {0:N6}" -f $meanStd)
Write-Host ("mean abs delta:        {0:N6}" -f $meanAbsDelta)

Write-Host ""
Write-Host "Complete. Fold selectors used only Train video holdouts."
Write-Host "Official Valid was not used for fold selection. Official Test was never constructed or accessed."
Write-Host "Result root: $root"
