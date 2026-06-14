#!/usr/bin/env bash
# Chạy ứng dụng mô phỏng SO-101 (macOS / Linux)
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "[setup] Tạo virtual environment..."
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/python app.py "$@"
