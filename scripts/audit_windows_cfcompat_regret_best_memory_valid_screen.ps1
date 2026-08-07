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

# Preserve the original audit as an immutable diagnostic. It writes its JSON
# payload before raising. Capture its output so an expected legacy precision
# mismatch does not flood the console with a traceback.
$OriginalOutput = & python ".\audit_cfcompat_regret_best_memory_valid_screen.py" --result-dir $OutputDir 2>&1
$OriginalExit = $LASTEXITCODE
if (-not (Test-Path (Join-Path $OutputDir "regret_best_memory_v4p2_audit_check.json"))) {
    $OriginalOutput | ForEach-Object { Write-Host $_ }
    throw "Original v4.2 audit did not produce its diagnostic JSON (exit=$OriginalExit)."
}
if ($OriginalExit -eq 0) {
    $OriginalOutput | ForEach-Object { Write-Host $_ }
}
else {
    Write-Host "Original v4.2 audit reported the known float64-vs-float32 replay mismatch; running precision-corrected replay."
}

& python ".\audit_cfcompat_regret_best_memory_valid_screen_precision_fixed.py" --result-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "v4.2 precision-corrected independent audit failed with exit code $LASTEXITCODE"
}
