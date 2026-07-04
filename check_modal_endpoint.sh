#!/bin/bash

# Modal Endpoint Diagnostic Script
# Checks if Modal is deployed and endpoint is working

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Modal Endpoint Diagnostic                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
MODAL_USERNAME="isuryaprakashs"
MODAL_APP="plumonet-inference"
MODAL_ENDPOINT="https://suryaprakash--plumonet-inference-run-inference.modal.run"

echo "📍 Modal Configuration:"
echo "  Username: $MODAL_USERNAME"
echo "  App: $MODAL_APP"
echo "  Endpoint: $MODAL_ENDPOINT"
echo ""

# Check 1: Can we reach the endpoint?
echo "🔍 Check 1: Testing endpoint connectivity..."
echo ""

response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$MODAL_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"blob_name": "test.zip"}')

echo "  HTTP Status: $response"
echo ""

if [ "$response" = "200" ]; then
    echo "  ✅ Status 200 OK - Endpoint is working!"
    echo ""
elif [ "$response" = "404" ]; then
    echo "  ❌ Status 404 Not Found - App not deployed or URL wrong"
    echo ""
    echo "  SOLUTIONS:"
    echo "    1. Check Modal dashboard: https://modal.com/apps/$MODAL_USERNAME/main"
    echo "    2. Redeploy: modal deploy modal_inference_FIXED.py"
    echo "    3. Verify endpoint URL in .env"
    echo ""
elif [ "$response" = "000" ]; then
    echo "  ❌ Connection failed - Can't reach endpoint"
    echo ""
    echo "  POSSIBLE CAUSES:"
    echo "    1. Modal endpoint is down"
    echo "    2. Network/firewall blocking"
    echo "    3. Wrong URL in .env"
    echo ""
else
    echo "  ⚠️  Status $response - Unexpected response"
    echo ""
fi

# Check 2: Get full response
echo "🔍 Check 2: Getting full response..."
echo ""

full_response=$(curl -s -X POST "$MODAL_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d '{"blob_name": "test.zip"}')

echo "  Response:"
echo "  $full_response"
echo ""

# Check 3: Check .env file
echo "🔍 Check 3: Checking .env file..."
echo ""

ENV_FILE="/Users/suryaprakash/Desktop/worknow/PlumoNet/.env"

if [ -f "$ENV_FILE" ]; then
    echo "  ✅ .env file exists"
    echo ""
    echo "  MODAL_ENDPOINT_URL setting:"
    grep "MODAL_ENDPOINT_URL" "$ENV_FILE" || echo "    ❌ MODAL_ENDPOINT_URL not found in .env"
    echo ""
else
    echo "  ❌ .env file not found at: $ENV_FILE"
    echo ""
fi

# Check 4: Recommendations
echo "════════════════════════════════════════════════════════════"
echo "📋 NEXT STEPS:"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ "$response" = "404" ]; then
    echo "1️⃣  Check Modal Dashboard:"
    echo "    https://modal.com/apps/$MODAL_USERNAME/main"
    echo ""
    echo "2️⃣  If app not deployed, redeploy:"
    echo "    modal deploy modal_inference_FIXED.py"
    echo ""
    echo "3️⃣  Copy exact endpoint URL from dashboard"
    echo ""
    echo "4️⃣  Update .env:"
    echo "    MODAL_ENDPOINT_URL=<paste-endpoint-here>"
    echo ""
    echo "5️⃣  Restart Docker:"
    echo "    docker run -p 3000:3000 --env-file .env ..."
    echo ""
elif [ "$response" = "200" ]; then
    echo "✅ Modal endpoint is working!"
    echo ""
    echo "If you're still getting 404 errors in Docker:"
    echo "  1. Verify .env has correct URL"
    echo "  2. Restart Docker container"
    echo "  3. Try upload again"
    echo ""
else
    echo "❓ Unclear status. Check:"
    echo "  1. Internet connectivity"
    echo "  2. Modal app deployment status"
    echo "  3. Endpoint URL format"
    echo ""
fi

echo "════════════════════════════════════════════════════════════"
echo ""
