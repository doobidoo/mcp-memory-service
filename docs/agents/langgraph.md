# LangGraph Integration Guide

Use mcp-memory-service as the persistent memory backend for LangGraph agents and StateGraphs.

## Key Differentiator: Cross-Graph Shared Memory

LangGraph's built-in `MemorySaver` is **graph-local and ephemeral** — memory is lost between runs and cannot be shared between different StateGraphs.

mcp-memory-service provides **persistent shared memory** across all graphs, runs, and even separate processes:

```
Graph A (Researcher)  ──┐
                        ├──→ mcp-memory-service ←──→ All graphs share one store
Graph B (Writer)     ──┘
Graph C (Reviewer)  ──┘
```

## Setup

```bash
pip install mcp-memory-service httpx
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
```

## Memory Tools for ReAct Agents

Define memory as `@tool` functions for use in a ReAct agent:

```python
import httpx
from langchain_core.tools import tool

MEMORY_URL = "http://localhost:8000"

@tool
async def search_memory(query: str) -> str:
    """Search long-term memory for relevant context."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MEMORY_URL}/api/search",
            json={"query": query, "n_results": 5},
        )
        hits = response.json()["results"]
        if not hits:
            return "No relevant memories found."
        return "\n".join(f"- {h['memory']['content']}" for h in hits)

@tool
async def store_memory(content: str, tags: list[str] = None) -> str:
    """Store a new memory for future retrieval."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MEMORY_URL}/api/memories",
            json={"content": content, "tags": tags or []},
        )
        result = response.json()
        return f"Stored memory: {result.get('content_hash', 'unknown')}"
```

Use in a ReAct agent:

```python
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6")

agent = create_react_agent(
    llm,
    tools=[search_memory, store_memory],
    state_modifier="You have access to long-term memory. Search memory before answering. Store important findings.",
)
```

## Memory Node in StateGraph

Add explicit memory retrieve/store nodes to a StateGraph:

```python
import httpx
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

MEMORY_URL = "http://localhost:8000"

class AgentState(TypedDict):
    messages: Annotated[list, "message history"]
    memory_context: str
    agent_id: str


async def retrieve_memory_node(state: AgentState) -> dict:
    """Retrieve this agent's memory before calling the LLM."""
    # Scope to this agent with by-tag, which returns every memory it wrote. Ranking
    # with /api/search first and filtering by tag afterwards would only see the top
    # of the list, so in a shared store a match below that window disappears.
    # by-tag ignores the query and caps nothing, so cap here.
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{MEMORY_URL}/api/search/by-tag",
            json={"tags": [f"agent:{state['agent_id']}"]},
        )
        response.raise_for_status()
        memories = [h["memory"] for h in response.json()["results"]][:5]

    if memories:
        context = "Relevant memory:\n" + "\n".join(f"- {m['content']}" for m in memories)
    else:
        context = ""

    return {"memory_context": context}


async def llm_node(state: AgentState) -> dict:
    """Call LLM with memory context injected into system message."""
    llm = ChatAnthropic(model="claude-sonnet-4-6")

    system_parts = ["You are a helpful assistant."]
    if state.get("memory_context"):
        system_parts.append(state["memory_context"])

    messages = [SystemMessage(content="\n\n".join(system_parts))] + state["messages"]
    response = await llm.ainvoke(messages)
    return {"messages": state["messages"] + [response]}


async def store_memory_node(state: AgentState) -> dict:
    """Store important information from LLM response."""
    last_response = state["messages"][-1]
    content = last_response.content if hasattr(last_response, "content") else str(last_response)

    async with httpx.AsyncClient() as client:
        await client.post(
            f"{MEMORY_URL}/api/memories",
            json={
                "content": content[:500],  # Store a summary
                "tags": [f"agent:{state['agent_id']}", "llm-response"],
                "memory_type": "observation",
            },
            headers={"X-Agent-ID": state["agent_id"]},
        )

    return {}


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("retrieve_memory", retrieve_memory_node)
graph.add_node("llm", llm_node)
graph.add_node("store_memory", store_memory_node)

graph.set_entry_point("retrieve_memory")
graph.add_edge("retrieve_memory", "llm")
graph.add_edge("llm", "store_memory")
graph.add_edge("store_memory", END)

agent = graph.compile()
```

## Cross-Graph Memory Sharing

Two independent StateGraphs sharing the same memory store:

```python
# Researcher graph — tags memories with agent:researcher
researcher_result = await researcher_agent.ainvoke({
    "messages": [HumanMessage(content="Research API rate limits")],
    "memory_context": "",
    "agent_id": "researcher",
})

# Writer graph — retrieves memories from researcher
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{MEMORY_URL}/api/search/by-tag",
        json={"tags": ["agent:researcher"]},  # Read from researcher's memory
    )
    # by-tag takes no limit: it returns every memory the researcher wrote, so cap
    # it before this reaches a context window.
    shared_context = [h["memory"] for h in response.json()["results"]][:10]

writer_result = await writer_agent.ainvoke({
    "messages": [HumanMessage(content="Write a summary of API limits")],
    "memory_context": "\n".join(m["content"] for m in shared_context),
    "agent_id": "writer",
})
```

## Incremental Conversation Summaries

Use `conversation_id` to store incremental summaries without semantic deduplication:

```python
import uuid

conversation_id = str(uuid.uuid4())

async def store_turn_summary(turn: int, summary: str):
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{MEMORY_URL}/api/memories",
            json={
                "content": f"Turn {turn}: {summary}",
                "tags": ["conversation-summary"],
                "conversation_id": conversation_id,  # Bypasses dedup for this convo
                "memory_type": "note",
            },
        )
```
