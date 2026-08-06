[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("Python command failed with exit code {0}: python {1}" -f $LASTEXITCODE, ($PythonArgs -join ' '))
    }
}

$OutputDir = ".\result\missing_baseline\cfcompat_student_safe_utility_v3\mosi\valid_screen"
if (-not (Test-Path $OutputDir)) {
    throw "Formal v3 result directory is absent: $OutputDir"
}

Write-Host "Project: $ProjectRoot"
Write-Host "Branch:  $(git branch --show-current)"
Write-Host "Commit:  $(git rev-parse HEAD)"
Write-Host "Stage:   Student-safe dynamic-utility v3 audit only (no training)"

Invoke-CheckedPython -PythonArgs @(
    "-m", "py_compile",
    ".\trains\singleTask\cfcompat_student_safe_utility_utils.py",
    ".\audit_cfcompat_student_safe_utility_valid_screen.py"
)

Invoke-CheckedPython -PythonArgs @(
    ".\audit_cfcompat_student_safe_utility_valid_screen.py",
    "--result-dir", $OutputDir
)
