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
    param([Parameter(Mandatory = $true)][string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw ("Python command failed with exit code {0}: python {1}" -f $LASTEXITCODE, ($PythonArgs -join ' '))
    }
}

$PreviousPythonWarnings = $env:PYTHONWARNINGS
$env:PYTHONWARNINGS = "ignore::FutureWarning"

try {
    Write-Host "Project: $ProjectRoot"
    Write-Host "Branch:  $(git branch --show-current)"
    Write-Host "Commit:  $(git rev-parse HEAD)"
    Write-Host "Seed:    $Seed"

    $compileArgs = @(
        "-m", "py_compile",
        ".\rebuild_windows_valid_only_prerequisites.py",
        ".\trains\singleTask\windows_valid_only_clean_dlf.py"
    )
    Invoke-CheckedPython -PythonArgs $compileArgs

    $preflightArgs = @(
        ".\rebuild_windows_valid_only_prerequisites.py",
        "--action", "preflight",
        "--seed", "$Seed",
        "--gpu-ids", "0",
        "--num-workers", "1"
    )
    Invoke-CheckedPython -PythonArgs $preflightArgs

    $runArgs = @(
        ".\rebuild_windows_valid_only_prerequisites.py",
        "--action", "clean-dlf",
        "--seed", "$Seed",
        "--gpu-ids", "0",
        "--num-workers", "1"
    )

    if (-not $Formal) {
        $runArgs += "--smoke-test"
    }
    if ($Overwrite) {
        $runArgs += "--overwrite"
    }

    Invoke-CheckedPython -PythonArgs $runArgs
}
finally {
    if ($null -eq $PreviousPythonWarnings) {
        Remove-Item Env:PYTHONWARNINGS -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONWARNINGS = $PreviousPythonWarnings
    }
}
