# Groq Agent Bridge - Requirements

Install the required package:

```bash
pip install groq
# or
uv pip install groq
```

Set up your environment:

```bash
export GROQ_API_KEY="your-api-key-here"
```

## Available Models

The Groq bridge supports multiple high-performance models:

| Model | Context | Best For | Speed |
|-------|---------|----------|-------|
| **llama-3.3-70b-versatile** | 128K | General purpose (default) | ~300ms |
| **moonshotai/kimi-k2-instruct** | 256K | Agentic coding, tool calling | ~200ms |
| **llama-3.1-8b-instant** | 128K | Fast, simple tasks | ~100ms |

**Kimi K2 Features:**
- 256K context window (largest on GroqCloud)
- 1 trillion parameters (32B activated)
- Excellent for front-end development and complex coding
- Superior agentic intelligence and tool calling
- 185 tokens/second throughput

## Usage Examples

### As a library from another AI agent:

```python
from groq_agent_bridge import GroqAgentBridge

# Initialize the bridge
bridge = GroqAgentBridge()

# Simple call
response = bridge.call_model_raw("Explain quantum computing in simple terms")
print(response)

# Advanced call with options
result = bridge.call_model(
    prompt="Generate Python code for a binary search tree",
    model="llama-3.3-70b-versatile",
    max_tokens=500,
    temperature=0.3,
    system_message="You are an expert Python programmer"
)
print(result)
```

### Command-line usage:

```bash
# Simple usage (uses default llama-3.3-70b-versatile)
./scripts/utils/groq "What is machine learning?"

# Use Kimi K2 for complex coding tasks
./scripts/utils/groq "Generate a React component with hooks" \
  --model "moonshotai/kimi-k2-instruct"

# Fast simple queries with llama-3.1-8b-instant
./scripts/utils/groq "Rate complexity 1-10: def add(a,b): return a+b" \
  --model "llama-3.1-8b-instant"

# Full options with default model
./scripts/utils/groq "Generate a SQL query" \
  --model "llama-3.3-70b-versatile" \
  --max-tokens 200 \
  --temperature 0.5 \
  --system "You are a database expert" \
  --json
```

### Integration with bash scripts:

```bash
#!/bin/bash
export GROQ_API_KEY="your-key"

# Get response and save to file
python groq_agent_bridge.py "Write a haiku about code" --temperature 0.9 > response.txt

# JSON output for parsing
json_response=$(python groq_agent_bridge.py "Explain REST APIs" --json)
# Parse with jq or other tools
```

This provides a completely non-interactive way for other AI agents to call Groq's models!
