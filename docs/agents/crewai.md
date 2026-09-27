# CrewAI Integration Guide

Use mcp-memory-service as the persistent shared memory backend for CrewAI agents and crews.

## Setup

```bash
pip install mcp-memory-service crewai httpx
MCP_ALLOW_ANONYMOUS_ACCESS=true memory server --http
```

## Custom Memory Tools

Implement `BaseTool` subclasses to expose memory as CrewAI tools:

```python
import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

MEMORY_URL = "http://localhost:8000"


class SearchMemoryInput(BaseModel):
    query: str = Field(description="Natural language search query")
    tags: list[str] = Field(default=[], description="Optional tag filters (e.g. ['agent:researcher'])")
    limit: int = Field(default=5, description="Maximum number of results")


class MemorySearchTool(BaseTool):
    name: str = "Search Memory"
    description: str = (
        "Search long-term shared memory for relevant context. "
        "Use tags like 'agent:researcher' to scope results to a specific agent."
    )
    args_schema: type[BaseModel] = SearchMemoryInput

    def _run(self, query: str, tags: list[str] = None, limit: int = 5) -> str:
        import asyncio
        return asyncio.run(self._arun(query, tags or [], limit))

    async def _arun(self, query: str, tags: list[str] = None, limit: int = 5) -> str:
        # /api/search ranks by query but cannot filter tags; /api/search/by-tag filters
        # tags but ignores the query and caps nothing. With tags, take the complete
        # scope: ranking first and filtering afterwards only sees the top of the list
        # and silently drops a match that ranks below it.
        async with httpx.AsyncClient() as client:
            if tags:
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


class StoreMemoryInput(BaseModel):
    content: str = Field(description="Memory content to store")
    tags: list[str] = Field(default=[], description="Tags to categorize the memory")
    memory_type: str = Field(default="note", description="Memory type: note, observation, decision, fact")


class MemoryStoreTool(BaseTool):
    name: str = "Store Memory"
    description: str = (
        "Store an important finding, decision, or fact in long-term shared memory. "
        "Other agents in the crew can retrieve it later."
    )
    args_schema: type[BaseModel] = StoreMemoryInput
    agent_id: str = ""  # Set when creating tool instance

    def _run(self, content: str, tags: list[str] = None, memory_type: str = "note") -> str:
        import asyncio
        return asyncio.run(self._arun(content, tags or [], memory_type))

    async def _arun(self, content: str, tags: list[str] = None, memory_type: str = "note") -> str:
        headers = {"Content-Type": "application/json"}
        if self.agent_id:
            headers["X-Agent-ID"] = self.agent_id

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEMORY_URL}/api/memories",
                json={"content": content, "tags": tags or [], "memory_type": memory_type},
                headers=headers,
            )
            result = response.json()

        if result.get("success"):
            return f"Stored memory (hash: {result['content_hash']})"
        return f"Failed to store memory: {result.get('message', 'unknown error')}"
```

## Agent-Scoped and Crew-Scoped Memory

Use tag conventions to scope memories:

```python
from crewai import Agent, Task, Crew

# Researcher agent — stores findings tagged with agent:researcher
researcher = Agent(
    role="Research Analyst",
    goal="Research and document technical findings",
    backstory="Expert at finding and organizing technical information.",
    tools=[
        MemorySearchTool(),
        MemoryStoreTool(agent_id="researcher"),  # Auto-tags with agent:researcher
    ],
)

# Writer agent — retrieves from researcher, stores own work
writer = Agent(
    role="Technical Writer",
    goal="Transform research findings into clear documentation",
    backstory="Expert at synthesizing technical research into readable content.",
    tools=[
        MemorySearchTool(),
        MemoryStoreTool(agent_id="writer"),
    ],
)

research_task = Task(
    description=(
        "Research API rate limiting best practices. "
        "Store key findings using the Store Memory tool with tags ['api', 'rate-limit', 'crew:docs-team']."
    ),
    expected_output="Summary of findings stored in memory",
    agent=researcher,
)

write_task = Task(
    description=(
        "Retrieve research findings using Search Memory with tags ['agent:researcher', 'api']. "
        "Write a technical guide based on the findings."
    ),
    expected_output="Technical documentation draft",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
)

result = crew.kickoff()
```

## Cross-Crew Knowledge Sharing

Knowledge stored by one crew is instantly available to other crews:

```python
# Crew 1: Analysis team stores findings
analysis_crew = Crew(
    agents=[researcher],
    tasks=[research_task],
)
analysis_crew.kickoff()
# Memories tagged: agent:researcher, crew:analysis-team

# Crew 2: Reporting team retrieves across crew boundary
async with httpx.AsyncClient() as client:
    response = await client.post(
        f"{MEMORY_URL}/api/search/by-tag",
        json={"tags": ["crew:analysis-team"]},  # Cross-crew retrieval
    )
    # No limit on by-tag — it returns the crew's entire history; cap it.
    findings = [h["memory"] for h in response.json()["results"]][:10]
```

## Post-Task Knowledge Base Inspection

After a crew run, inspect the knowledge graph to see what was learned:

```python
import httpx

async def inspect_crew_knowledge(crew_name: str):
    async with httpx.AsyncClient() as client:
        # List all memories from this crew run
        response = await client.get(
            f"{MEMORY_URL}/api/memories",
            params={"tags": f"crew:{crew_name}", "page_size": 50},
        )
        memories = response.json()["memories"]
        print(f"Crew {crew_name} stored {len(memories)} memories")

        # Check relationships between memories (knowledge graph)
        for mem in memories[:5]:
            assoc_response = await client.get(
                f"{MEMORY_URL}/api/graph/associations/{mem['content_hash']}",
            )
            associations = assoc_response.json().get("associations", [])
            if associations:
                print(f"  Memory '{mem['content'][:50]}...' has {len(associations)} connections")

asyncio.run(inspect_crew_knowledge("docs-team"))
```
