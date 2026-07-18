# Delta V node - one-line installer for Windows.
#   irm https://raw.githubusercontent.com/alexkolumbo/deltav/main/install.ps1 | iex
# Finds a real Python, installs Delta V, and launches the friendly wizard.
$ErrorActionPreference = "Stop"
Write-Host "DV  Delta V - установка ноды`n"

# 1. Find a REAL Python 3.11+.
#    `python` on a fresh Windows is often the Microsoft Store alias stub — it
#    "exists" as a command but only prints 'Python was not found...'. So we
#    don't trust Get-Command; we actually run each candidate and require it to
#    print its version. `py -3` (the official launcher) is tried too.
#    Result goes into two script-scope variables rather than a returned array:
#    PowerShell unrolls single-element arrays on return, and `$a[1..($a.Count-1)]`
#    on a 1-element array is the DESCENDING range 1..0 — it yields ($null, $a[0]),
#    which silently passed "python" as the script name (`python.exe python -m pip`).
$script:PyExe = ""
$script:PyPre = @()

function Resolve-Python {
    foreach ($cand in @("python", "python3", "py")) {
        $pre = if ($cand -eq "py") { @("-3") } else { @() }
        try {
            $out = (& $cand @pre "-c" "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null | Out-String).Trim()
        } catch { continue }
        if ($out -match "^(\d+)\.(\d+)$") {
            $maj = [int]$Matches[1]; $min = [int]$Matches[2]
            if ($maj -gt 3 -or ($maj -eq 3 -and $min -ge 11)) {
                $script:PyExe = $cand
                $script:PyPre = $pre
                return $true
            }
        }
    }
    return $false
}

if (-not (Resolve-Python)) {
    Write-Host "Не найден реальный Python 3.11+ (в PATH, скорее всего, заглушка из Microsoft Store)."
    Write-Host ""
    Write-Host "Поставьте Python и запустите снова:"
    Write-Host "  winget install -e --id Python.Python.3.12"
    Write-Host "  # затем закройте и откройте PowerShell заново"
    Write-Host "или скачайте с https://www.python.org/downloads/ (отметьте 'Add python.exe to PATH')."
    exit 1
}
Write-Host ("Python: " + ((@($script:PyExe) + $script:PyPre) -join " "))

function Invoke-Py { & $script:PyExe @($script:PyPre + $args) }

# 2. Install Delta V. From a source checkout -> editable; otherwise from the
#    GitHub tarball (no git required — the package isn't on PyPI yet).
Write-Host "Ставлю Delta V..."
Invoke-Py -m pip install -q --upgrade pip
if ((Test-Path "pyproject.toml") -and (Select-String -Path "pyproject.toml" -Pattern "deltav-network" -Quiet)) {
    Invoke-Py -m pip install -q --user -e ".[hub]"
} else {
    $url = "https://github.com/alexkolumbo/deltav/archive/refs/heads/main.tar.gz"
    Invoke-Py -m pip install -q --user "deltav-network[hub] @ $url"
}

# 3. Wizard.
Write-Host ""
Invoke-Py -m deltav.cli setup @args
