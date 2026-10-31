$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "dm1") {
    throw "Activa el entorno Conda dm1 antes de ejecutar este script."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python -m predictguard.data.download
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
