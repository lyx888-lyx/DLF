param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-crossfit-residual-consensus-v9"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$required = @(
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_raw_valid_events.csv"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen reference missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_crossfit_residual_consensus_utils.py `
    .\train_cfcompat_crossfit_residual_consensus_valid_screen_v9.py `
    .\smoke_test_cfcompat_crossfit_residual_consensus.py

python .\smoke_test_cfcompat_crossfit_residual_consensus.py

$argsList = @(
    ".\train_cfcompat_crossfit_residual_consensus_valid_screen_v9.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList

$root = ".\result\missing_baseline\cfcompat_crossfit_residual_consensus_v9\mosi\valid_screen\seed1113_dev"
$summary = Join-Path $root "crossfit_residual_v9_valid_screen_summary.json"
$transfer = Join-Path $root "crossfit_residual_v9_transfer_summary.csv"
$manifest = Join-Path $root "crossfit_residual_v9_fold_manifest.csv"
$consensus = Join-Path $root "crossfit_residual_v9_valid_consensus_events.csv"

Write-Host ""
Write-Host "================ v9 mechanism summary ================"
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v9 transfer summary ================="
Import-Csv $transfer | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v9 fold manifest ===================="
Import-Csv $manifest | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v9 consensus diagnostics ============"
$rows = Import-Csv $consensus
$apply = ($rows | Where-Object { $_.consensus_applied -eq "True" }).Count / [double]$rows.Count
$agreement = ($rows | ForEach-Object { [double]$_.consensus_sign_agreement } | Measure-Object -Average).Average
$std = ($rows | ForEach-Object { [double]$_.consensus_fold_std } | Measure-Object -Average).Average
$delta = ($rows | ForEach-Object { [math]::Abs([double]$_.consensus_delta) } | Measure-Object -Average).Average
Write-Host ("consensus applied rate: {0:P2}" -f $apply)
Write-Host ("mean sign agreement:   {0:N6}" -f $agreement)
Write-Host ("mean fold std:         {0:N6}" -f $std)
Write-Host ("mean abs delta:        {0:N6}" -f $delta)

Write-Host ""
Write-Host "Complete. Fold checkpoints were selected only by Train video holdouts."
Write-Host "Official Valid was not used for fold selection. Official Test was never constructed or accessed."
Write-Host "Result root: $root"
