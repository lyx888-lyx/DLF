[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$Branch = (git branch --show-current).Trim()
if ($Branch -ne "feature/cfcompat-regret-preserve-beneficial-guard-v4p1") {
    throw "Audit v4.1 only from feature/cfcompat-regret-preserve-beneficial-guard-v4p1; current branch: $Branch"
}

$OutputDir = ".\result\missing_baseline\cfcompat_regret_preserve_guard_v4p1\mosi\valid_screen\seed1113_dev"
& python ".\audit_cfcompat_regret_preserve_guard_valid_screen.py" --result-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "v4.1 independent audit failed with exit code $LASTEXITCODE"
}
