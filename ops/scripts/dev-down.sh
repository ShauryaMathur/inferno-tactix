#!/usr/bin/env bash
set -euo pipefail

pkill -f "python -m inferno_api" || true
pkill -f "python -m inferno_sim" || true
pkill -f "webpack serve --no-https" || true
