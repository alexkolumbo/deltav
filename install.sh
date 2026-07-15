#!/bin/sh
# Delta V node — one-line installer for Linux / macOS.
#   curl -fsSL <raw-url>/install.sh | sh
# Installs Python deps and launches the friendly setup wizard.
set -e

echo "ΔV  Delta V — установка ноды"
echo

# 1. Python 3.11+
if ! command -v python3 >/dev/null 2>&1; then
  echo "Нужен Python 3.11+. Установите его и запустите снова:"
  echo "  Ubuntu/Debian:  sudo apt install -y python3 python3-pip"
  echo "  macOS:          brew install python"
  exit 1
fi

# 2. deltav package (from PyPI once published, or local checkout)
echo "Ставлю Delta V…"
if [ -f "pyproject.toml" ] && grep -q "deltav-network" pyproject.toml 2>/dev/null; then
  python3 -m pip install -q --user -e ".[hub]"
else
  python3 -m pip install -q --user "deltav-network[hub]" 2>/dev/null || {
    echo "Пакет ещё не в PyPI. Запустите install.sh из папки с исходниками Delta V."
    exit 1
  }
fi

# 3. wizard
echo
exec python3 -m deltav.cli setup "$@"
