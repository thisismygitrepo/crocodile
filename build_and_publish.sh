#!/usr/bin/env bash
set -euo pipefail

echo "==> Cleaning previous build artifacts"
rm -rf dist build *.egg-info || true

echo "==> Building (uv build)"
if ! command -v uv >/dev/null 2>&1; then
  echo "'uv' command not found. Install uv first: https://github.com/astral-sh/uv" >&2
  exit 127
fi
uv build

if [[ "${SKIP_PUBLISH:-}" == "1" ]]; then
  echo "==> SKIP_PUBLISH=1 set; skipping upload. Artifacts left in dist/."
  exit 0
fi

echo "==> Extracting PyPI token"
PYPI_TOKEN=""
if [ -f "$HOME/dotfiles/creds/msc/.pypirc" ]; then
  PYPI_TOKEN=$(awk -F ' *= *' '/\[pypi\]/{f=1} f&&$1=="password"{print $2; exit}' "$HOME/dotfiles/creds/msc/.pypirc")
elif [ -f "$HOME/.pypirc" ]; then
  PYPI_TOKEN=$(awk -F ' *= *' '/\[pypi\]/{f=1} f&&$1=="password"{print $2; exit}' "$HOME/.pypirc")
fi

if [ -z "$PYPI_TOKEN" ]; then
  echo "PyPI token not found in ~/.pypirc or dotfiles path" >&2
  exit 1
fi

echo "==> Publishing (uv publish)"
uv publish --token "$PYPI_TOKEN"

echo "==> Done"
