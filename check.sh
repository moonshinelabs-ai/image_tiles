#!/bin/bash
set -e

echo "=== Running Ruff formatter (with autofix) ==="
uv run ruff format .

echo ""
echo "=== Running Ruff linter with import sorting (with autofix) ==="
uv run ruff check --fix .

echo ""
echo "=== Running ty type checker ==="
uv run ty check

echo ""
echo "=== All checks passed! ==="
