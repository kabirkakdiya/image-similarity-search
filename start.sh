#!/usr/bin/env bash
set -e

exec uv run fastapi run src/main.py --host 0.0.0.0 --port 7860
