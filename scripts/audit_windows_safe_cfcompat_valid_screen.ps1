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

$PreviousPythonWarnings = $env:PYTHONWARNINGS
$env:PYTHONWARNINGS = "ignore::FutureWarning"

try {
    $ResultDir = ".\result\missing_baseline\cfcompat_safe_projection_v1\mosi\valid_screen"
    if (-not (Test-Path $ResultDir)) {
        throw "Formal Safe-CFCompat result directory is absent: $ResultDir"
    }

    Write-Host "Project: $ProjectRoot"
    Write-Host "Branch:  $(git branch --show-current)"
    Write-Host "Commit:  $(git rev-parse HEAD)"
    Write-Host "Stage:   Safe-CFCompat audit only (no training)"

    Invoke-CheckedPython -PythonArgs @(
        "-m", "py_compile",
        ".\audit_windows_safe_cfcompat_valid_screen.py"
    )

    Invoke-CheckedPython -PythonArgs @(
        ".\audit_windows_safe_cfcompat_valid_screen.py",
        "--result-dir", $ResultDir
    )
}
finally {
    if ($null -eq $PreviousPythonWarnings) {
        Remove-Item Env:PYTHONWARNINGS -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONWARNINGS = $PreviousPythonWarnings
    }
}
