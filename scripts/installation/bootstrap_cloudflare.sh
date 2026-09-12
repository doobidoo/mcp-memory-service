#!/usr/bin/env bash
# Bootstrap Cloudflare backend for MCP Memory Service.
# Prerequisites: Cloudflare account, API token, and account ID in .env
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "❌ Project venv not found. Run: uv pip install --python .venv/bin/python -e ."
  exit 1
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "✅ Created .env from .env.example"
fi

if grep -q 'CLOUDFLARE_API_TOKEN=your-cloudflare-api-token-here' .env \
   || grep -q 'CLOUDFLARE_ACCOUNT_ID=your-account-id-here' .env; then
  echo ""
  echo "============================================================"
  echo "  Cloudflare credentials needed in .env"
  echo "============================================================"
  echo ""
  echo "1. Create account:  https://dash.cloudflare.com/sign-up"
  echo "2. Copy Account ID from the dashboard sidebar"
  echo "3. Create API token: https://dash.cloudflare.com/profile/api-tokens"
  echo "   Use Custom Token with:"
  echo "     - Vectorize: Edit"
  echo "     - D1: Edit"
  echo "     - Workers AI: Read"
  echo "     - R2: Edit (optional)"
  echo ""
  echo "4. Edit ${ROOT}/.env and set:"
  echo "     CLOUDFLARE_API_TOKEN=..."
  echo "     CLOUDFLARE_ACCOUNT_ID=..."
  echo ""
  echo "5. Re-run: bash scripts/installation/bootstrap_cloudflare.sh"
  echo "============================================================"
  exit 1
fi

echo "🚀 Creating Cloudflare resources..."
"$PYTHON" scripts/installation/setup_cloudflare_resources.py --no-r2 --write-env --yes

echo ""
echo "🧪 Running integration test..."
"$PYTHON" scripts/testing/test_cloudflare_backend.py
