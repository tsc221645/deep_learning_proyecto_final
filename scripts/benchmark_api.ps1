$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    $env:PYTHONPATH = "$repoRoot\src;$repoRoot"
    python -m scripts.benchmark_api
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
