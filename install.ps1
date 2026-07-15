# Delta V node - one-line installer for Windows.
#   irm <raw-url>/install.ps1 | iex
# Installs Python deps and launches the friendly setup wizard.
$ErrorActionPreference = "Stop"

Write-Host "DV  Delta V - установка ноды`n"

# 1. Python 3.11+
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Host "Нужен Python 3.11+. Скачайте с https://www.python.org/downloads/"
    Write-Host "(при установке отметьте галочку 'Add Python to PATH'), затем запустите снова."
    exit 1
}

# 2. deltav package
Write-Host "Ставлю Delta V..."
if ((Test-Path "pyproject.toml") -and (Select-String -Path "pyproject.toml" -Pattern "deltav-network" -Quiet)) {
    python -m pip install -q --user -e ".[hub]"
} else {
    try { python -m pip install -q --user "deltav-network[hub]" }
    catch {
        Write-Host "Пакет ещё не в PyPI. Запустите install.ps1 из папки с исходниками Delta V."
        exit 1
    }
}

# 3. wizard
Write-Host ""
python -m deltav.cli setup @args
