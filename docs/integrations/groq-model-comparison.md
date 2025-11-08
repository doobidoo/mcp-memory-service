# Groq Model Comparison for Code Quality Analysis

## Available Models

### 1. llama-3.3-70b-versatile (Default)
**Best for:** General-purpose code analysis with detailed explanations

**Characteristics:**
- ✅ Comprehensive, detailed responses
- ✅ Thorough breakdown of complexity factors
- ✅ Balanced speed and quality
- ⚠️ Can be verbose for simple tasks

**Performance:**
- Response time: ~1.2-1.6s
- Detail level: High
- Accuracy: Excellent

**Example Output (Complexity 6/10):**
```
**Complexity Rating: 6/10**

Here's a breakdown of the complexity factors:
1. **Functionality**: The function performs data processing...
2. **Conditional Statements**: There are two conditional statements...
3. **Loops**: There is one loop...
[... detailed analysis continues ...]
```

### 2. moonshotai/kimi-k2-instruct (Recommended for Code Analysis)
**Best for:** Fast, accurate code analysis with agentic intelligence

**Characteristics:**
- ✅ **Fastest response time** (~0.9s)
- ✅ Concise, accurate assessments
- ✅ 256K context window (largest on GroqCloud)
- ✅ Excellent for complex coding tasks
- ✅ Superior agentic intelligence

**Performance:**
- Response time: ~0.9s (fastest tested)
- Detail level: Concise but accurate
- Accuracy: Excellent

**Example Output (Complexity 2/10):**
```
Complexity: 2/10

The function is short, uses only basic control flow and dict/list
operations, and has no recursion, nested loops, or advanced algorithms.
```

**Kimi K2 Features:**
- 1 trillion parameters (32B activated MoE)
- 256K context window
- 185 tokens/second throughput
- Optimized for front-end development
- Superior tool calling capabilities

### 3. llama-3.1-8b-instant
**Best for:** Simple queries requiring minimal analysis

**Characteristics:**
- ⚠️ Despite name "instant", actually slower than Kimi K2
- ⚠️ Very verbose, includes unnecessary details
- ✅ Lower cost than larger models

**Performance:**
- Response time: ~1.6s (slowest tested)
- Detail level: Very high (sometimes excessive)
- Accuracy: Good but over-explains

**Example Output (Complexity 4/10):**
```
I would rate the complexity of this function a 4 out of 10.

Here's a breakdown of the factors I considered:
- **Readability**: 6/10
- **Locality**: 7/10
- **Abstraction**: 8/10
- **Efficiency**: 9/10
[... continues with edge cases, type hints, etc ...]
```

## Recommendations by Use Case

### Pre-commit Hooks (Speed Critical)
**Use: moonshotai/kimi-k2-instruct**
```bash
./scripts/utils/groq "Complexity 1-10: $(cat file.py)" --model moonshotai/kimi-k2-instruct
```
- Fastest response (~0.9s)
- Accurate enough for quality gates
- Minimizes developer wait time

### PR Review (Quality Critical)
**Use: llama-3.3-70b-versatile**
```bash
./scripts/utils/groq "Detailed analysis: $(cat file.py)"
```
- Comprehensive feedback
- Detailed explanations help reviewers
- Balanced speed/quality

### Security Analysis (Accuracy Critical)
**Use: moonshotai/kimi-k2-instruct**
```bash
./scripts/utils/groq "Security scan: $(cat file.py)" --model moonshotai/kimi-k2-instruct
```
- Excellent at identifying vulnerabilities
- Fast enough for CI/CD
- Superior agentic intelligence for complex patterns

### Simple Queries
**Use: llama-3.1-8b-instant** (if cost is priority)
```bash
./scripts/utils/groq "Is this function pure?" --model llama-3.1-8b-instant
```
- Lowest cost
- Good for yes/no questions
- Avoid for complex analysis (slower than Kimi K2)

## Performance Summary

| Model | Response Time | Detail Level | Best For | Context |
|-------|--------------|--------------|----------|---------|
| **Kimi K2** | 0.9s ⚡ | Concise ✓ | Speed + Accuracy | 256K |
| **llama-3.3-70b** | 1.2-1.6s | Detailed ✓ | Comprehensive | 128K |
| **llama-3.1-8b** | 1.6s | Very Detailed | Cost savings | 128K |

## Cost Comparison (Groq Pricing)

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| Kimi K2 | $1.00/M | $3.00/M | Premium speed + quality |
| llama-3.3-70b | ~$0.50/M | ~$0.80/M | Balanced |
| llama-3.1-8b | ~$0.05/M | ~$0.10/M | High volume |

## Switching Models

All models use the same interface:
```bash
# Default (llama-3.3-70b-versatile)
./scripts/utils/groq "Your prompt"

# Kimi K2 (recommended for code analysis)
./scripts/utils/groq "Your prompt" --model moonshotai/kimi-k2-instruct

# Fast/cheap
./scripts/utils/groq "Your prompt" --model llama-3.1-8b-instant
```

## Conclusion

**For MCP Memory Service code quality workflows:**
- ✅ **Kimi K2**: Best overall - fastest, accurate, excellent for code
- ✅ **llama-3.3-70b**: Good for detailed explanations in PR reviews
- ⚠️ **llama-3.1-8b**: Avoid for code analysis despite "instant" name
