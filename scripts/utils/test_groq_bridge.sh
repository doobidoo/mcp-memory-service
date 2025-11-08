#!/bin/bash
# Test script for Groq bridge integration
# Demonstrates usage without requiring API key

set -e

echo "=== Groq Bridge Integration Test ==="
echo ""

# Check if groq package is installed
echo "1. Checking Python groq package..."
if python3 -c "import groq" 2>/dev/null; then
    echo "   ✓ groq package installed"
else
    echo "   ✗ groq package NOT installed"
    echo ""
    echo "To install: pip install groq"
    echo "Or: uv pip install groq"
    exit 1
fi

# Check if API key is set
echo ""
echo "2. Checking GROQ_API_KEY environment variable..."
if [ -z "$GROQ_API_KEY" ]; then
    echo "   ✗ GROQ_API_KEY not set"
    echo ""
    echo "To set: export GROQ_API_KEY='your-api-key-here'"
    echo "Get your API key from: https://console.groq.com/keys"
    echo ""
    echo "Skipping API test (would require valid key)"
else
    echo "   ✓ GROQ_API_KEY configured"

    # Test the bridge with a simple query
    echo ""
    echo "3. Testing Groq bridge with sample query..."
    echo ""

    python3 scripts/utils/groq_agent_bridge.py \
        "Rate the complexity of this Python function on a scale of 1-10: def add(a, b): return a + b" \
        --json
fi

echo ""
echo "=== Integration Test Complete ==="
echo ""
echo "Usage examples:"
echo ""
echo "# Complexity analysis"
echo "python scripts/utils/groq_agent_bridge.py \"Analyze complexity 1-10: \$(cat file.py)\""
echo ""
echo "# Security scan"
echo "python scripts/utils/groq_agent_bridge.py \"Check for security issues: \$(cat file.py)\" --json"
echo ""
echo "# With custom model and temperature"
echo "python scripts/utils/groq_agent_bridge.py \"Your prompt\" --model llama2-70b-4096 --temperature 0.3"
