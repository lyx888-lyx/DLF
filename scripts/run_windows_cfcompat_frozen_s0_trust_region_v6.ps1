param(
    [switch]$SmokeOnly,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-frozen-s0-trust-region-v6"
$Branch = (git branch --show-current).Trim()
if ($Branch -ne $ExpectedBranch) {
    throw "Wrong branch. Expected '$ExpectedBranch' but found '$Branch'."
}

$V4Root = ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev"
$Required = @(
    "$V4Root\regret_preserve_v4_candidate_grid.csv",
    "$V4Root\regret_preserve_v4_candidate_raw_valid_events.csv",
    "$V4Root\regret_preserve_v4_valid_screen_summary.json"
)
foreach ($Path in $Required) {
    if (-not (Test-Path $Path)) { throw "Missing frozen v4 prerequisite: $Path" }
}

Write-Host "Project: $(Get-Location)"
Write-Host "Branch:  $Branch"
Write-Host "Commit:  $(git rev-parse HEAD)"
Write-Host "Stage:   CFCompatKD v6 frozen-S0 function-level trust-region mechanism test"
Write-Host "Seed:    1113 only"
Write-Host "Test:    forbidden / not constructed"
Write-Host "Base:    frozen v4 DISTILL/PRESERVE/ABSTAIN"
Write-Host "Change:  one-sided safe trust to cached pre-training Student S0"

python -m py_compile `
    .\trains\singleTask\cfcompat_s0_trust_region_utils.py `
    .\train_cfcompat_frozen_s0_trust_region_valid_screen_v6.py `
    .\smoke_test_cfcompat_frozen_s0_trust_region.py
if ($LASTEXITCODE -ne 0) { throw "py_compile failed" }

python .\smoke_test_cfcompat_frozen_s0_trust_region.py
if ($LASTEXITCODE -ne 0) { throw "v6 utility smoke test failed" }

$RunArgs = @()
if ($Overwrite) { $RunArgs += "--overwrite" }
if ($SmokeOnly) {
    python .\train_cfcompat_frozen_s0_trust_region_valid_screen_v6.py --smoke-test @RunArgs
    if ($LASTEXITCODE -ne 0) { throw "v6 training smoke run failed" }
    Write-Host "Smoke-only complete. No formal mechanism decision was made."
    exit 0
}

python .\train_cfcompat_frozen_s0_trust_region_valid_screen_v6.py @RunArgs
if ($LASTEXITCODE -ne 0) { throw "v6 formal Seed1113 Valid-only run failed" }

$Out = ".\result\missing_baseline\cfcompat_frozen_s0_trust_region_v6\mosi\valid_screen\seed1113_dev"
Write-Host ""
Write-Host "================ v6 mechanism summary ================"
Get-Content (Join-Path $Out "frozen_s0_v6_valid_screen_summary.json") -Raw
Write-Host ""
Write-Host "================ v6 transfer summary ================="
Import-Csv (Join-Path $Out "frozen_s0_v6_transfer_summary.csv") | Format-Table -AutoSize
Write-Host ""
Write-Host "================ v6 S0 drift summary ================="
Import-Csv (Join-Path $Out "frozen_s0_v6_valid_s0_drift_summary.csv") | Format-Table -AutoSize
Write-Host ""
Write-Host "Complete. One Seed1113 Student trajectory was trained; official Test was never constructed or accessed."
