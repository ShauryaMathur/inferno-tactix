#!/usr/bin/env bash
# Deploy FireCastBot API and web frontend to GCP Cloud Run.
#
# Usage:
#   ./ops/scripts/deploy-gcp.sh              # build + deploy both services
#   ./ops/scripts/deploy-gcp.sh --api-only   # API only
#   ./ops/scripts/deploy-gcp.sh --web-only   # web only
#
# Run from the project root or from any subdirectory — the script resolves
# the root automatically.
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
GCP_PROJECT="ee-shauryamathur2001"
GCP_REGION="us-central1"
REGISTRY="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/inferno-tactix"
API_IMAGE="${REGISTRY}/api:latest"
WEB_IMAGE="${REGISTRY}/web:latest"
API_URL="https://inferno-api-604513266448.${GCP_REGION}.run.app"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ── Argument parsing ──────────────────────────────────────────────────────────
DEPLOY_API=true
DEPLOY_WEB=true

for arg in "$@"; do
  case "$arg" in
    --api-only) DEPLOY_WEB=false ;;
    --web-only) DEPLOY_API=false ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: $0 [--api-only | --web-only]"
      exit 1
      ;;
  esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
step() { echo; echo "▶ $*"; }

# ── Auth ──────────────────────────────────────────────────────────────────────
step "Configuring Docker credentials for Artifact Registry"
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

# ── API ───────────────────────────────────────────────────────────────────────
if $DEPLOY_API; then
  step "Building API image (Dockerfile.firecastbot)"
  docker build --platform linux/amd64 \
    -f apps/api/Dockerfile.firecastbot \
    -t "$API_IMAGE" \
    "$ROOT_DIR"

  step "Pushing API image"
  docker push "$API_IMAGE"

  step "Deploying inferno-api to Cloud Run"
  gcloud run deploy inferno-api \
    --image "$API_IMAGE" \
    --region "$GCP_REGION" \
    --project "$GCP_PROJECT" \
    --quiet

  echo "✓ API deployed: $API_URL"
fi

# ── Web ───────────────────────────────────────────────────────────────────────
if $DEPLOY_WEB; then
  step "Building web image (Dockerfile.prod, API_BASE_URL baked in)"
  docker build --platform linux/amd64 --no-cache \
    -f apps/web/Dockerfile.prod \
    --build-arg API_BASE_URL="$API_URL" \
    -t "$WEB_IMAGE" \
    "$ROOT_DIR"

  step "Pushing web image"
  docker push "$WEB_IMAGE"

  step "Deploying inferno-web to Cloud Run"
  gcloud run deploy inferno-web \
    --image "$WEB_IMAGE" \
    --region "$GCP_REGION" \
    --project "$GCP_PROJECT" \
    --quiet

  WEB_URL="https://inferno-web-604513266448.${GCP_REGION}.run.app"
  echo "✓ Web deployed: $WEB_URL"
fi

echo
echo "Done."
