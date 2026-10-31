$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "dm1") {
    throw "Activa el entorno Conda dm1 antes de ejecutar este script."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python -m pip install --no-user --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple "torch==2.14.0"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    python -m pip install --no-user -e ".[dev]"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
