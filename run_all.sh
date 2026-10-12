#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
python scripts/run_synthetic.py
cd supplement
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
rm -f supplement.aux supplement.log supplement.out supplement.spl
cd "$ROOT"
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
