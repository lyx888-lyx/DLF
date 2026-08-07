param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-sample-conditioned-residual-v8"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$required = @(
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_raw_valid_events.csv",
    ".\result\missing_baseline\cfcompat_frozen_backbone_adapter_isolation_v7\mosi\valid_screen\seed1113_dev\frozen_backbone_v7_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_frozen_backbone_adapter_isolation_v7\mosi\valid_screen\seed1113_dev\frozen_backbone_v7_candidate_raw_valid_events.csv"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen reference missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_sample_residual_utils.py `
    .\train_cfcompat_sample_conditioned_residual_valid_screen_v8.py `
    .\smoke_test_cfcompat_sample_conditioned_residual.py

python .\smoke_test_cfcompat_sample_conditioned_residual.py

$argsList = @(
    ".\train_cfcompat_sample_conditioned_residual_valid_screen_v8.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList

$root = ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev"
$summary = Join-Path $root "sample_residual_v8_valid_screen_summary.json"
$transfer = Join-Path $root "sample_residual_v8_transfer_summary.csv"
$epochs = Join-Path $root "sample_residual_v8_valid_epoch_transfer_summary.csv"
$residual = Join-Path $root "sample_residual_v8_residual_valid_events.csv"

Write-Host ""
Write-Host "================ v8 mechanism summary ================"
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v8 transfer summary ================="
Import-Csv $transfer | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v8 epoch trajectory summary ========="
Import-Csv $epochs | Format-Table -AutoSize

Write-Host ""
Write-Host "================ v8 residual magnitude ==============="
$rows = Import-Csv $residual
$abs = $rows | ForEach-Object { [double]$_.abs_residual_delta }
$mean = ($abs | Measure-Object -Average).Average
$max = ($abs | Measure-Object -Maximum).Maximum
Write-Host ("mean abs residual: {0:N6}" -f $mean)
Write-Host ("max abs residual:  {0:N6}" -f $max)

Write-Host ""
Write-Host "Complete. Official Test was never constructed or accessed."
Write-Host "Result root: $root"
