[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$Branch = (git branch --show-current).Trim()
if ($Branch -ne "feature/cfcompat-regret-best-memory-v4p2") {
    throw "Run v4.2 audit only from feature/cfcompat-regret-best-memory-v4p2; current branch: $Branch"
}

$OutputDir = ".\result\missing_baseline\cfcompat_regret_best_memory_v4p2\mosi\valid_screen\seed1113_dev"
& python ".\audit_cfcompat_regret_best_memory_valid_screen.py" --result-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "v4.2 independent audit failed with exit code $LASTEXITCODE"
}
