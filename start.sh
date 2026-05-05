#!/usr/bin/env bash
set -e

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
fi

exec uv run --env-file=.env fastapi dev src/main.py --host 0.0.0.0 --port 7860
