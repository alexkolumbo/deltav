#!/bin/sh
# Delta V node — one-line installer for Linux / macOS.
#   curl -fsSL https://raw.githubusercontent.com/alexkolumbo/deltav/main/install.sh | sh
# Installs Python deps and launches the friendly setup wizard.
set -e

echo "ΔV  Delta V — установка ноды"
echo

# 1. Python 3.11+ (verify the version actually runs, not just that it exists)
PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && \
     "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info>=(3,11) else 1)' 2>/dev/null; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Нужен Python 3.11+. Установите его и запустите снова:"
  echo "  Ubuntu/Debian:  sudo apt install -y python3 python3-pip"
  echo "  macOS:          brew install python"
  exit 1
fi
echo "Python: $PY"

# 2. Delta V — from a source checkout (editable) or the GitHub tarball
#    (no git required; the package isn't on PyPI yet).
echo "Ставлю Delta V…"
"$PY" -m pip install -q --upgrade pip 2>/dev/null || true
if [ -f "pyproject.toml" ] && grep -q "deltav-network" pyproject.toml 2>/dev/null; then
  "$PY" -m pip install -q --user -e ".[hub]"
else
  url="https://github.com/alexkolumbo/deltav/archive/refs/heads/main.tar.gz"
  "$PY" -m pip install -q --user "deltav-network[hub] @ $url"
fi

# 3. wizard
echo
exec "$PY" -m deltav.cli setup "$@"
