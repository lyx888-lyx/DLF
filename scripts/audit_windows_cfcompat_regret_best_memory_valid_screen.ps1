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
$OriginalAuditJson = Join-Path $OutputDir "regret_best_memory_v4p2_audit_check.json"
$PrecisionAuditJson = Join-Path $OutputDir "regret_best_memory_v4p2_audit_check_precision_fixed.json"

# The legacy auditor is intentionally retained as an immutable diagnostic. It
# can exit non-zero on the known float64-vs-float32 best-memory replay issue.
# Run it as a real child process with stdout/stderr redirected to files so
# PowerShell 5.1 cannot turn native stderr into a terminating NativeCommandError
# under $ErrorActionPreference='Stop'.
$TempRoot = Join-Path $env:TEMP ("dlf-v4p2-audit-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempRoot | Out-Null
$LegacyStdout = Join-Path $TempRoot "legacy.stdout.txt"
$LegacyStderr = Join-Path $TempRoot "legacy.stderr.txt"

try {
    $PythonExe = (Get-Command python).Source
    $LegacyArgs = @(
        ".\audit_cfcompat_regret_best_memory_valid_screen.py",
        "--result-dir", $OutputDir
    )
    $Legacy = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $LegacyArgs `
        -WorkingDirectory $ProjectRoot `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $LegacyStdout `
        -RedirectStandardError $LegacyStderr

    $OriginalExit = $Legacy.ExitCode
    if (-not (Test-Path $OriginalAuditJson)) {
        if (Test-Path $LegacyStdout) { Get-Content $LegacyStdout | ForEach-Object { Write-Host $_ } }
        if (Test-Path $LegacyStderr) { Get-Content $LegacyStderr | ForEach-Object { Write-Host $_ } }
        throw "Original v4.2 audit did not produce its diagnostic JSON (exit=$OriginalExit)."
    }

    if ($OriginalExit -eq 0) {
        if (Test-Path $LegacyStdout) { Get-Content $LegacyStdout | ForEach-Object { Write-Host $_ } }
    }
    else {
        Write-Host "Original v4.2 audit reported the known legacy memory replay mismatch (exit=$OriginalExit)."
        Write-Host "Running precision-corrected float32/state-independent audit now."
    }

    & python ".\audit_cfcompat_regret_best_memory_valid_screen_precision_fixed.py" --result-dir $OutputDir
    if ($LASTEXITCODE -ne 0) {
        throw "v4.2 precision-corrected independent audit failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path $PrecisionAuditJson)) {
        throw "Precision-corrected v4.2 audit did not produce: $PrecisionAuditJson"
    }
}
finally {
    if (Test-Path $TempRoot) {
        Remove-Item $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
