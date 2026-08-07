[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$OutputDir = ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev"
if (-not (Test-Path $OutputDir)) {
    throw "Formal v4 output is absent: $OutputDir"
}

& python ".\audit_cfcompat_regret_preserve_valid_screen.py" --result-dir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Independent v4 audit failed with exit code $LASTEXITCODE"
}

$SummaryPath = Join-Path $OutputDir "regret_preserve_v4_valid_screen_summary.json"
$Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
Write-Host ("verdict: {0}" -f $Summary.verdict)
Write-Host "official Test was not constructed"
