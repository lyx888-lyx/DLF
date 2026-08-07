[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$Branch = (git branch --show-current).Trim()
if ($Branch -ne "feature/cfcompat-crossfit-transfer-risk-gate-v5") {
    throw "Run v5 audit only from feature/cfcompat-crossfit-transfer-risk-gate-v5; current branch: $Branch"
}

$OutputDir = ".\result\missing_baseline\cfcompat_crossfit_transfer_risk_v5\mosi\valid_screen\seed1113_dev"
& python ".\audit_cfcompat_crossfit_transfer_risk_valid_screen.py" --result-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "v5 independent audit failed with exit code $LASTEXITCODE"
}
