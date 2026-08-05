[CmdletBinding()]
param(
    [ValidateSet(1112, 1113, 1115)]
    [int]$Seed = 1112,

    [switch]$Formal,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE: python $($Arguments -join ' ')"
    }
}

Write-Host "Project: $ProjectRoot"
Write-Host "Branch:  $(git branch --show-current)"
Write-Host "Commit:  $(git rev-parse HEAD)"
Write-Host "Seed:    $Seed"

Invoke-CheckedPython -m py_compile `
    .\rebuild_windows_valid_only_prerequisites.py `
    .\trains\singleTask\windows_valid_only_clean_dlf.py

Invoke-CheckedPython .\rebuild_windows_valid_only_prerequisites.py `
    --action preflight `
    --seed $Seed `
    --gpu-ids 0 `
    --num-workers 1

$arguments = @(
    ".\rebuild_windows_valid_only_prerequisites.py",
    "--action", "clean-dlf",
    "--seed", "$Seed",
    "--gpu-ids", "0",
    "--num-workers", "1"
)

if (-not $Formal) {
    $arguments += "--smoke-test"
}
if ($Overwrite) {
    $arguments += "--overwrite"
}

Invoke-CheckedPython @arguments
