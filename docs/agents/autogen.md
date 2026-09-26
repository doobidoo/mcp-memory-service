# AutoGen Integration Guide

Use mcp-memory-service as the persistent memory backend for AutoGen 0.4+ multi-agent conversations.

## Setup

```bash
pip install mcp-memory-service autogen-agentchat httpx
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
```

## Context Injection Before Each Turn

The most effective pattern for AutoGen: retrieve relevant memory and inject it into the agent's system message before each conversation turn.

```python
import httpx
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models import OpenAIChatCompletionClient

MEMORY_URL = "http://localhost:8000"


async def retrieve_context(query: str, tags: list[str] | None = None, limit: int = 5) -> str:
    """Retrieve relevant memory context for injection into system message.

    The two search endpoints each do half the job: /api/search ranks by the query but
    cannot filter tags, /api/search/by-tag filters by tag but ignores the query and
    caps nothing. When tags are given, scope wins: by-tag sees every matching memory,
    where ranking first and filtering afterwards would only ever see the top of the
    list and silently miss a match that ranks below it in a shared store. The cost is
    that a scoped result is not query-ranked — cap it and let the model read it.
    """
    async with httpx.AsyncClient() as client:
        if tags:
            response = await client.post(
                f"{MEMORY_URL}/api/search/by-tag",
                json={"tags": tags, "match_all": True},  # default is ANY, not ALL
            )
        else:
            response = await client.post(
                f"{MEMORY_URL}/api/search",
                json={"query": query, "n_results": max(1, min(limit, 100))},
            )
        # Without this, a 422 (n_results out of the 1-100 range, no tags supplied)
        # reads as an empty result list and the agent is told there is no memory.
        response.raise_for_status()
        memories = [h["memory"] for h in response.json()["results"]][:limit]

    if not memories:
        return ""
    return "Relevant context from memory:\n" + "\n".join(f"- {m['content']}" for m in memories)


async def store_finding(content: str, agent_name: str, tags: list[str] | None = None):
    """Store an important finding from an agent."""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{MEMORY_URL}/api/memories",
            json={
                "content": content,
                "tags": tags or [],
                "memory_type": "observation",
            },
            headers={"X-Agent-ID": agent_name},
        )


# Build memory-aware agents
async def build_memory_agent(name: str, task_description: str) -> AssistantAgent:
    # Pre-load relevant context before conversation starts
    context = await retrieve_context(task_description)

    system_message = f"You are {name}, a helpful AI assistant."
    if context:
        system_message += f"\n\n{context}"

    return AssistantAgent(
        name=name,
        system_message=system_message,
        model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
    )


async def main():
    task = "Analyze the performance characteristics of our REST API"

    researcher = await build_memory_agent("researcher", task)
    analyst = await build_memory_agent("analyst", task)

    team = RoundRobinGroupChat([researcher, analyst], max_turns=4)
    result = await team.run(task=task)

    # Store key findings for future conversations
    for message in result.messages:
        if len(message.content) > 100:  # Store substantive messages
            await store_finding(
                content=message.content[:500],
                agent_name=message.source,
                tags=["analysis", "api-performance"],
            )

asyncio.run(main())
```

## Function Tool Schema (AutoGen 0.4+)

Expose memory operations as callable tools using AutoGen's function tool format:

```python
from autogen_core.tools import FunctionTool
import httpx

MEMORY_URL = "http://localhost:8000"


async def search_memory(query: str, limit: int = 5, tags: list[str] | None = None) -> str:
    """
    Search long-term memory for relevant context.

    Args:
        query: Natural language search query
        limit: Maximum number of results (default: 5)
        tags: Optional tag filters to scope results (e.g. ['agent:researcher'])

    Returns:
        Formatted string of matching memories, or empty string if none found.
    """
    async with httpx.AsyncClient() as client:
        if tags:
            # Tag scope must be complete: by-tag sees every match, where ranking first
            # and filtering afterwards misses whatever falls outside the window.
            # It ignores the query and caps nothing, so cap here.
            response = await client.post(
                f"{MEMORY_URL}/api/search/by-tag",
                json={"tags": tags, "match_all": True},  # default is ANY, not ALL
            )
        else:
            response = await client.post(
                f"{MEMORY_URL}/api/search",
                json={"query": query, "n_results": max(1, min(limit, 100))},  # 422 outside 1-100
            )
        response.raise_for_status()  # else an error body reads as "no memories"
        memories = [h["memory"] for h in response.json()["results"]][:limit]

    if not memories:
        return "No relevant memories found."
    return "\n".join(f"[{', '.join(m['tags'])}] {m['content']}" for m in memories)


async def store_memory(
    content: str,
    tags: list[str] | None = None,
    memory_type: str = "note",
) -> str:
    """
    Store an important finding or fact in long-term memory.

    Args:
        content: The memory content to store
        tags: Tags to categorize the memory
        memory_type: Type classification: note, observation, decision, fact

    Returns:
        Confirmation message with the memory's content hash.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MEMORY_URL}/api/memories",
            json={"content": content, "tags": tags or [], "memory_type": memory_type},
        )
        result = response.json()

    if result.get("success"):
        return f"Memory stored successfully (hash: {result['content_hash']})"
    return f"Failed to store: {result.get('message', 'unknown error')}"


# Register as AutoGen FunctionTools
search_memory_tool = FunctionTool(search_memory, description="Search long-term memory")
store_memory_tool = FunctionTool(store_memory, description="Store a finding in long-term memory")

# Use in an agent
agent = AssistantAgent(
    name="memory_aware_agent",
    tools=[search_memory_tool, store_memory_tool],
    system_message="Search memory before answering. Store important findings after each task.",
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
)
```

## conversation_id for Long AutoGen Threads

AutoGen conversations can run for many turns, generating large amounts of similar content that would be de-duplicated by the default semantic deduplication. Use `conversation_id` to bypass deduplication for incremental storage:

```python
import uuid

# Generate once per AutoGen conversation
conversation_id = str(uuid.uuid4())


async def store_turn_memory(turn: int, speaker: str, summary: str):
    """Store a per-turn summary without semantic deduplication."""
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{MEMORY_URL}/api/memories",
            json={
                "content": f"[Turn {turn}] {speaker}: {summary}",
                "tags": ["autogen-turn", f"agent:{speaker}"],
                "memory_type": "note",
                "conversation_id": conversation_id,  # Skip dedup — store all turns
            },
        )


# Use in conversation loop
async def run_conversation_with_memory(task: str):
    team = RoundRobinGroupChat([researcher, analyst], max_turns=6)
    result = await team.run(task=task)

    for i, message in enumerate(result.messages):
        if len(message.content) > 50:
            await store_turn_memory(i, message.source, message.content[:300])

    return result
```

**Why this matters:** Without `conversation_id`, storing "Turn 1: Found X" and "Turn 2: Found Y" may be de-duplicated because they are semantically similar. With `conversation_id`, all turns are stored independently, creating a full conversation record that future agents can retrieve.

## Retrieve Cross-Agent Knowledge

After one AutoGen conversation completes, subsequent conversations can retrieve its findings:

```python
# Session 1: Researcher finds API limits
await store_finding("REST API rate limit: 100 req/min", "researcher", ["api", "limits"])

# Session 2 (days later): New agent searches for API knowledge
context = await retrieve_context("API rate limits and quotas", tags=["agent:researcher"])
print(context)
# "Relevant context from memory:
# - REST API rate limit: 100 req/min"
```
