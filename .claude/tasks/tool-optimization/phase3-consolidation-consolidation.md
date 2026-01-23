# Phase 3: Consolidation Tool Consolidation (7 → 1)

## Ziel

Konsolidiere diese 7 Consolidation-Tools zu einem `memory_consolidate` Tool:
- `consolidate_memories` (run consolidation)
- `consolidation_status` (get status)
- `consolidation_recommendations` (get recommendations)
- `scheduler_status` (scheduler info)
- `trigger_consolidation` (manual trigger)
- `pause_consolidation` (pause jobs)
- `resume_consolidation` (resume jobs)

## Neue Tool-Signatur

```python
async def manage_consolidation(
    self,
    action: Literal["run", "status", "recommend", "scheduler", "pause", "resume"],
    time_horizon: Optional[Literal["daily", "weekly", "monthly", "quarterly", "yearly"]] = None,
    immediate: bool = True
) -> Dict[str, Any]:
    """
    Unified consolidation management.
    
    Actions:
    - run: Execute consolidation (requires time_horizon)
    - status: Get consolidation system health
    - recommend: Get optimization recommendations (requires time_horizon)
    - scheduler: View scheduled jobs
    - pause: Pause jobs (optional time_horizon for specific horizon)
    - resume: Resume jobs (optional time_horizon for specific horizon)
    """
```

---

## Task 3.1: Backend Implementation

**Datei:** `src/mcp_memory_service/services/memory_service.py`

### Amp Prompt

```
Refactor consolidation operations in src/mcp_memory_service/services/memory_service.py:

1. Use finder to locate all consolidation methods:
   - consolidate_memories
   - get_consolidation_status
   - get_consolidation_recommendations
   - get_scheduler_status
   - trigger_consolidation
   - pause_consolidation
   - resume_consolidation

2. Create ONE unified method:

from typing import Literal, Optional, Dict, Any

async def manage_consolidation(
    self,
    action: Literal["run", "status", "recommend", "scheduler", "pause", "resume"],
    time_horizon: Optional[Literal["daily", "weekly", "monthly", "quarterly", "yearly"]] = None,
    immediate: bool = True
) -> Dict[str, Any]:

3. Implementation - route to existing methods:

async def manage_consolidation(
    self,
    action: Literal["run", "status", "recommend", "scheduler", "pause", "resume"],
    time_horizon: Optional[str] = None,
    immediate: bool = True
) -> Dict[str, Any]:
    """
    Unified consolidation management interface.
    
    Args:
        action: The consolidation action to perform
        time_horizon: Time horizon (required for run/recommend, optional for pause/resume)
        immediate: For 'run' action - execute immediately vs schedule
    
    Returns:
        Action-specific result dictionary
    """
    
    if action == "run":
        if not time_horizon:
            return {"error": "time_horizon required for 'run' action"}
        return await self.consolidate_memories(time_horizon)
    
    elif action == "status":
        return await self.get_consolidation_status()
    
    elif action == "recommend":
        if not time_horizon:
            return {"error": "time_horizon required for 'recommend' action"}
        return await self.get_consolidation_recommendations(time_horizon)
    
    elif action == "scheduler":
        return await self.get_scheduler_status()
    
    elif action == "pause":
        return await self.pause_consolidation(time_horizon)  # None = pause all
    
    elif action == "resume":
        return await self.resume_consolidation(time_horizon)  # None = resume all
    
    else:
        return {"error": f"Unknown action: {action}"}

4. Keep existing methods but mark as internal (prefix with _) or deprecated:

# Option A: Make internal
async def _consolidate_memories(self, time_horizon: str) -> Dict[str, Any]:
    # existing implementation
    pass

# Option B: Deprecate public interface
async def consolidate_memories(self, time_horizon: str) -> Dict[str, Any]:
    warnings.warn(
        "consolidate_memories is deprecated, use manage_consolidation(action='run', time_horizon=...)",
        DeprecationWarning,
        stacklevel=2
    )
    return await self.manage_consolidation(action="run", time_horizon=time_horizon)

5. Add type hints and docstrings.
```

---

## Task 3.2: MCP Tool Definition

**Datei:** `src/mcp_memory_service/server_impl.py`

### Amp Prompt

```
Update tool definitions in src/mcp_memory_service/server_impl.py:

1. Find these consolidation tool definitions in handle_list_tools:
   - consolidate_memories
   - consolidation_status
   - consolidation_recommendations
   - scheduler_status
   - trigger_consolidation
   - pause_consolidation
   - resume_consolidation

2. Replace ALL 7 tools with ONE unified tool:

types.Tool(
    name="memory_consolidate",
    description="""Memory consolidation management - run, monitor, and control memory optimization.

USE THIS WHEN:
- User asks about memory optimization, cleanup, or consolidation
- Need to check consolidation system health
- Want to manually trigger or schedule consolidation
- Need to pause/resume consolidation jobs

ACTIONS:
- run: Execute consolidation for a time horizon (requires time_horizon)
- status: Get consolidation system health and statistics
- recommend: Get optimization recommendations for a time horizon
- scheduler: View all scheduled consolidation jobs
- pause: Pause consolidation (all or specific horizon)
- resume: Resume paused consolidation

TIME HORIZONS:
- daily: Consolidate last 24 hours
- weekly: Consolidate last 7 days
- monthly: Consolidate last 30 days
- quarterly: Consolidate last 90 days
- yearly: Consolidate last 365 days

Examples:
{"action": "status"}
{"action": "run", "time_horizon": "weekly"}
{"action": "run", "time_horizon": "daily", "immediate": true}
{"action": "recommend", "time_horizon": "monthly"}
{"action": "scheduler"}
{"action": "pause"}
{"action": "pause", "time_horizon": "daily"}
{"action": "resume", "time_horizon": "weekly"}
""",
    inputSchema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "status", "recommend", "scheduler", "pause", "resume"],
                "description": "Consolidation action to perform"
            },
            "time_horizon": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly", "quarterly", "yearly"],
                "description": "Time horizon (required for run/recommend, optional for pause/resume)"
            },
            "immediate": {
                "type": "boolean",
                "default": true,
                "description": "For 'run' action: execute immediately vs schedule for later"
            }
        },
        "required": ["action"]
    }
)

3. Add handler method:

async def handle_memory_consolidate(self, arguments: dict) -> List[types.TextContent]:
    result = await self.memory_service.manage_consolidation(
        action=arguments["action"],
        time_horizon=arguments.get("time_horizon"),
        immediate=arguments.get("immediate", True)
    )
    return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

4. Update handle_call_tool:

elif name == "memory_consolidate":
    return await self.handle_memory_consolidate(arguments)

5. Add deprecation mapping:

"consolidate_memories": ("memory_consolidate", lambda a: {
    "action": "run",
    "time_horizon": a["time_horizon"]
}),
"consolidation_status": ("memory_consolidate", lambda a: {
    "action": "status"
}),
"consolidation_recommendations": ("memory_consolidate", lambda a: {
    "action": "recommend",
    "time_horizon": a["time_horizon"]
}),
"scheduler_status": ("memory_consolidate", lambda a: {
    "action": "scheduler"
}),
"trigger_consolidation": ("memory_consolidate", lambda a: {
    "action": "run",
    "time_horizon": a["time_horizon"],
    "immediate": a.get("immediate", True)
}),
"pause_consolidation": ("memory_consolidate", lambda a: {
    "action": "pause",
    "time_horizon": a.get("time_horizon")
}),
"resume_consolidation": ("memory_consolidate", lambda a: {
    "action": "resume",
    "time_horizon": a.get("time_horizon")
}),
```

---

## Validation

```bash
# Test all consolidation actions
uv run python -c "
import asyncio
from mcp_memory_service.services.memory_service import MemoryService

async def test():
    # Status
    r1 = await service.manage_consolidation(action='status')
    print('Status:', r1.get('status', r1))
    
    # Scheduler
    r2 = await service.manage_consolidation(action='scheduler')
    print('Scheduler:', r2)
    
    # Recommendations
    r3 = await service.manage_consolidation(action='recommend', time_horizon='weekly')
    print('Recommendations:', r3)

asyncio.run(test())
"
```

---

## Checkliste

- [ ] `manage_consolidation` Methode implementiert
- [ ] Alle 6 Actions funktionieren
- [ ] Validation für required parameters
- [ ] Alte Methoden deprecated oder internal
- [ ] `memory_consolidate` Tool definiert
- [ ] Deprecation routing für 7 alte Tools
- [ ] Tests grün
