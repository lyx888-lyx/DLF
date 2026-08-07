param(
    [switch]$Overwrite,
    [int[]]$GpuIds = @(0)
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-frozen-backbone-adapter-isolation-v7"
$CurrentBranch = (git branch --show-current).Trim()
if ($CurrentBranch -ne $ExpectedBranch) {
    throw "Wrong branch: '$CurrentBranch'. Expected '$ExpectedBranch'."
}

Write-Host "================ CFCompatKD v7 ================"
Write-Host "Branch: $CurrentBranch"
Write-Host "HEAD:   $((git rev-parse HEAD).Trim())"
Write-Host "Stage:  hard frozen-backbone / missing-adapter isolation"
Write-Host "Scope:  Seed1113 official Valid only; Test forbidden"
Write-Host "Trainables: missing_audio_token, missing_vision_token, mask_adapter.weight"
Write-Host "Per-epoch Valid sample predictions: enabled"

$Files = @(
    "trains/singleTask/cfcompat_adapter_isolation_utils.py",
    "train_cfcompat_frozen_backbone_adapter_isolation_valid_screen_v7.py",
    "smoke_test_cfcompat_frozen_backbone_adapter_isolation.py"
)
python -m py_compile @Files
if ($LASTEXITCODE -ne 0) { throw "py_compile failed" }

python smoke_test_cfcompat_frozen_backbone_adapter_isolation.py
if ($LASTEXITCODE -ne 0) { throw "v7 utility smoke test failed" }

$Args = @(
    "train_cfcompat_frozen_backbone_adapter_isolation_valid_screen_v7.py",
    "--dataset", "mosi",
    "--num-workers", "1",
    "--gpu-ids"
) + ($GpuIds | ForEach-Object { "$_" })
if ($Overwrite) { $Args += "--overwrite" }

python @Args
if ($LASTEXITCODE -ne 0) { throw "v7 formal run failed" }

$Root = "result/missing_baseline/cfcompat_frozen_backbone_adapter_isolation_v7/mosi/valid_screen/seed1113_dev"
$SummaryPath = Join-Path $Root "frozen_backbone_v7_valid_screen_summary.json"
$TransferPath = Join-Path $Root "frozen_backbone_v7_transfer_summary.csv"
$EpochPath = Join-Path $Root "frozen_backbone_v7_valid_epoch_transfer_summary.csv"
$ClipPath = Join-Path $Root "frozen_backbone_v7_valid_clip_failure_epoch_summary.csv"
$OnsetPath = Join-Path $Root "frozen_backbone_v7_valid_failure_onset.csv"

foreach ($Path in @($SummaryPath,$TransferPath,$EpochPath,$ClipPath,$OnsetPath)) {
    if (-not (Test-Path $Path)) { throw "Missing expected artifact: $Path" }
}

Write-Host "`n================ v7 mechanism summary ================"
Get-Content $SummaryPath -Raw
Write-Host "`n================ v7 transfer summary ================="
Import-Csv $TransferPath | Format-Table -AutoSize
Write-Host "`n================ v7 epoch trajectory summary ========="
Import-Csv $EpochPath | Format-Table -AutoSize
Write-Host "`n================ v7 clip failure onset summary ========"
Import-Csv $ClipPath | Format-Table -AutoSize
Write-Host "`nComplete. Official Test was never constructed or accessed."
Write-Host "Failure onset table: $OnsetPath"
