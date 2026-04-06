#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Start these in separate terminals:"
echo "1) cd $ROOT_DIR/apps/api && PYTHONPATH=src python -m inferno_api"
echo "2) cd $ROOT_DIR/apps/simulator && PYTHONPATH=src python -m inferno_sim"
echo "3) cd $ROOT_DIR/apps/web && npm start"
