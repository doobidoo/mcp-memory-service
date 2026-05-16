# Benchmark Adapters

Adapter scripts to run LongMemEval and LoCoMo benchmarks against alternative memory tools for comparison.

## Available Adapters

| Adapter | Tool | Requirements |
|---------|------|-------------|
| `adapter_mem0.py` | [mem0](https://github.com/mem0ai/mem0) | `pip install mem0ai` + API key |
| `adapter_supermemory.py` | [supermemory](https://github.com/supermemoryai/supermemory) | Self-hosted instance or API key |

## Usage

```bash
# Run LongMemEval with mem0
export MEM0_API_KEY=your_key
python benchmark_longmemeval.py --adapter mem0

# Run with supermemory
export SUPERMEMORY_API_URL=http://localhost:3000
python benchmark_longmemeval.py --adapter supermemory

# Run with native mcp-memory-service (default)
python benchmark_longmemeval.py --adapter native
```

## Adding a New Adapter

1. Create `adapter_yourservice.py`
2. Implement `MemoryAdapter` interface (setup, store, search, teardown)
3. Add env var configuration at the top
4. Test against LongMemEval dataset

## Interface

```python
class MemoryAdapter(ABC):
    @property
    def name(self) -> str: ...
    async def setup(self) -> None: ...
    async def store(self, content: str, metadata: dict) -> str: ...
    async def search(self, query: str, limit: int = 5) -> list[dict]: ...
    async def teardown(self) -> None: ...
```
