#!/usr/bin/env bash
# Deploy to Google Cloud Run.
#
# Prerequisite (run once, in your own terminal - it opens a browser):
#   gcloud auth login
#
# Then:  ./deploy/cloudrun.sh
#
# Region is asia-south1 (Mumbai): the judges are in Goa, and every 100ms of
# transatlantic round trip would be latency this project spent real effort
# removing.
set -euo pipefail

GCLOUD="${GCLOUD:-$(command -v gcloud || echo /opt/homebrew/share/google-cloud-sdk/bin/gcloud)}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-voice-rag}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
die()  { printf '\n\033[1;31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

step "Checking authentication"
ACCOUNT="$("$GCLOUD" auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)"
[ -n "$ACCOUNT" ] || die "not authenticated. Run:  gcloud auth login"
echo "authenticated as $ACCOUNT"

PROJECT="${PROJECT:-$("$GCLOUD" config get-value project 2>/dev/null || true)}"
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
  PROJECT="$("$GCLOUD" projects list --format='value(projectId)' --limit=1 2>/dev/null || true)"
  [ -n "$PROJECT" ] || die "no GCP project found. Create one at https://console.cloud.google.com/projectcreate then re-run."
fi
"$GCLOUD" config set project "$PROJECT" >/dev/null 2>&1
echo "project: $PROJECT"

step "Checking billing"
# Cloud Run and Cloud Build both refuse to run without a billing account linked.
# The free tier still applies - linking billing is not the same as being charged.
BILLED="$("$GCLOUD" beta billing projects describe "$PROJECT" --format='value(billingEnabled)' 2>/dev/null || echo "unknown")"
if [ "$BILLED" != "True" ]; then
  echo "billing does not appear to be enabled (reported: $BILLED)"
  echo "Enable it here, then re-run:"
  echo "  https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT"
  [ "$BILLED" = "unknown" ] || die "billing must be linked before deploying"
  echo "(could not verify - continuing, the API calls below will fail clearly if it is off)"
fi

step "Enabling required APIs (idempotent, can take a minute)"
"$GCLOUD" services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

step "Building image with Cloud Build"
IMAGE="gcr.io/$PROJECT/$SERVICE"
# The image embeds ~18k chunks at build time. On the default single-core builder
# that step alone exceeded a 45 minute deadline; e2-highcpu-8 finishes it in a
# couple of minutes and costs cents because the build is short.
"$GCLOUD" builds submit --tag "$IMAGE" --timeout=45m \
  --machine-type=e2-highcpu-8 --project "$PROJECT"

step "Deploying to Cloud Run"
ENV_ARGS="EMBED_THREADS=2"
if [ -f .env ]; then
  # Read the keys from .env rather than taking them on the command line, so they
  # never land in shell history.
  GROQ="$(grep -E '^GROQ_API_KEY=' .env | cut -d= -f2- || true)"
  SARVAM="$(grep -E '^SARVAM_API_KEY=' .env | cut -d= -f2- || true)"
  [ -n "${GROQ:-}" ]   && ENV_ARGS="$ENV_ARGS,GROQ_API_KEY=$GROQ"
  [ -n "${SARVAM:-}" ] && ENV_ARGS="$ENV_ARGS,SARVAM_API_KEY=$SARVAM"
fi

"$GCLOUD" run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --allow-unauthenticated \
  --port 7860 \
  --memory 2Gi \
  --cpu 2 \
  --concurrency 8 \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 4 \
  --set-env-vars "$ENV_ARGS" \
  --project "$PROJECT"

URL="$("$GCLOUD" run services describe "$SERVICE" --region "$REGION" \
        --format='value(status.url)' --project "$PROJECT")"

step "Verifying"
curl -fsS "$URL/health" | head -c 400 || die "health check failed - check logs: gcloud run services logs read $SERVICE --region $REGION"

cat <<EOF


  Live:  $URL

  min-instances is 0, so the service scales to zero and the first request after
  idle pays a cold start (model + index load). That is free. Before a demo or
  judging, either open the link a minute early, or pin one warm instance:

    $GCLOUD run services update $SERVICE --region $REGION --min-instances 1

  and set it back to 0 afterwards - a pinned instance bills continuously and is
  outside the always-free tier.
EOF
