#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR/apps/web"
npm install

cd "$ROOT_DIR/apps/api"
pip install -r requirements.txt

cd "$ROOT_DIR/apps/simulator"
pip install -r requirements.txt

cd "$ROOT_DIR/apps/headless-client"
pip install -r requirements.txt
