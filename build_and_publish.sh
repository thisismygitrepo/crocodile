#!/usr/bin/env bash

# Build and publish
uv build

# Extract PyPI token from .pypirc file
if [ -f "$HOME/dotfiles/creds/msc/.pypirc" ]; then
  PYPI_TOKEN=$(awk -F ' *= *' '/\[pypi\]/{f=1} f&&$1=="password"{print $2; exit}' "$HOME/dotfiles/creds/msc/.pypirc")
elif [ -f "$HOME/.pypirc" ]; then
  PYPI_TOKEN=$(awk -F ' *= *' '/\[pypi\]/{f=1} f&&$1=="password"{print $2; exit}' "$HOME/.pypirc")
fi

if [ -z "$PYPI_TOKEN" ]; then
  echo "PYPI token not found in ~/.pypirc or dotfiles path" >&2
  exit 1
fi

uv publish --token "$PYPI_TOKEN"
