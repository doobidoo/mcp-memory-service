# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Consolidation handler functions for MCP server.

Memory consolidation, scheduler control, status monitoring, and recommendations.
Extracted from server_impl.py Phase 2.2 refactoring.
"""

import json
import logging
import traceback
from typing import List

from mcp import types
from ...config import CONSOLIDATION_ENABLED

logger = logging.getLogger(__name__)


def _sanitize_log_value(value: object) -> str:
    """Sanitize a user-provided value for safe inclusion in log messages."""
    return str(value).replace("\n", "\\n").replace("\r", "\\r").replace("\x1b", "\\x1b")


async def handle_consolidate_memories(
    server, arguments: dict
) -> List[types.TextContent]:
    """Handle memory consolidation requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidator:
        return [
            types.TextContent(
                type="text", text="Error: Consolidation system not available"
            )
        ]

    try:
        time_horizon = arguments.get("time_horizon")
        if not time_horizon:
            return [
                types.TextContent(type="text", text="Error: time_horizon is required")
            ]

        if time_horizon not in [
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "yearly",
            "incremental",
        ]:
            return [
                types.TextContent(
                    type="text",
                    text="Error: Invalid time_horizon. Must be one of: daily, weekly, monthly, quarterly, yearly, incremental",
                )
            ]

        # time_horizon is checked against the allow-list above, so it can only be
        # one of six literals here — lazy %-formatting rather than a sanitize call.
        logger.info("Starting %s consolidation", time_horizon)

        # Extract optional theme parameter
        theme = arguments.get("theme")
        extra = {"theme": theme} if theme else {}

        # Run consolidation (with timeout for incremental)
        if time_horizon == "incremental":
            import asyncio

            INCREMENTAL_TIMEOUT_SECONDS = 10
            try:
                report = await asyncio.wait_for(
                    server.consolidator.consolidate(time_horizon, **extra),
                    timeout=INCREMENTAL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                # Advance last_run_at even on timeout to prevent infinite retry loop (#986)
                if server.consolidator.run_tracker:
                    await server.consolidator.run_tracker.record_run("incremental", 0)
                return [
                    types.TextContent(
                        type="text",
                        text="Incremental consolidation timed out (>10s). Partial progress saved.",
                    )
                ]
        else:
            report = await server.consolidator.consolidate(time_horizon, **extra)

        # Format response
        result = f"""Consolidation completed successfully!

Time Horizon: {report.time_horizon}
Duration: {(report.end_time - report.start_time).total_seconds():.2f} seconds
Memories Processed: {report.memories_processed}
Associations Discovered: {report.associations_discovered}
Clusters Created: {report.clusters_created}
Memories Compressed: {report.memories_compressed}
Memories Archived: {report.memories_archived}"""

        if report.errors:
            result += "\n\nWarnings/Errors:\n" + "\n".join(
                f"- {error}" for error in report.errors
            )

        return [types.TextContent(type="text", text=result)]

    except Exception as e:
        error_msg = f"Error during consolidation: {str(e)}"
        logger.error("%s\n%s", _sanitize_log_value(error_msg), traceback.format_exc())
        return [types.TextContent(type="text", text=error_msg)]


async def handle_consolidation_status(
    server, arguments: dict
) -> List[types.TextContent]:
    """Handle consolidation status requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidator:
        return [types.TextContent(type="text", text="Consolidation system: DISABLED")]

    try:
        # Get health check from consolidator
        health = await server.consolidator.health_check()

        # Format status report
        status_lines = [
            f"Consolidation System Status: {health.get('status', 'unknown').upper()}",
            f"Last Updated: {health.get('timestamp', 'N/A')}",
            "",
            "Component Health:",
        ]

        for component, component_health in health.get("components", {}).items():
            status = component_health.get("status", "unknown")
            status_lines.append(f"  {component}: {status.upper()}")
            if status == "unhealthy" and "error" in component_health:
                status_lines.append(f"    Error: {component_health['error']}")

        stats = health.get("statistics", {})
        status_lines.extend(
            [
                "",
                "Statistics:",
                f"  Total consolidation runs: {stats.get('total_runs', 0)}",
                f"  Successful runs: {stats.get('successful_runs', 0)}",
                f"  Total memories processed: {stats.get('total_memories_processed', 0)}",
                f"  Total associations created: {stats.get('total_associations_created', 0)}",
                f"  Total clusters created: {stats.get('total_clusters_created', 0)}",
                f"  Total memories compressed: {stats.get('total_memories_compressed', 0)}",
                f"  Total memories archived: {stats.get('total_memories_archived', 0)}",
            ]
        )

        if health.get("last_consolidation_times"):
            status_lines.extend(["", "Last Consolidation Times:"])
            for horizon, timestamp in health["last_consolidation_times"].items():
                status_lines.append(f"  {horizon}: {timestamp}")

        return [types.TextContent(type="text", text="\n".join(status_lines))]

    except Exception as e:
        error_msg = f"Error getting consolidation status: {str(e)}"
        logger.error("%s\n%s", _sanitize_log_value(error_msg), traceback.format_exc())
        return [types.TextContent(type="text", text=error_msg)]


async def handle_consolidation_recommendations(
    server, arguments: dict
) -> List[types.TextContent]:
    """Handle consolidation recommendation requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidator:
        return [
            types.TextContent(
                type="text", text="Error: Consolidation system not available"
            )
        ]

    try:
        time_horizon = arguments.get("time_horizon")
        if not time_horizon:
            return [
                types.TextContent(type="text", text="Error: time_horizon is required")
            ]

        if time_horizon not in [
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "yearly",
            "incremental",
        ]:
            return [types.TextContent(type="text", text="Error: Invalid time_horizon")]

        # Get recommendations
        recommendations = await server.consolidator.get_consolidation_recommendations(
            time_horizon
        )

        # Format response
        lines = [
            f"Consolidation Recommendations for {time_horizon} horizon:",
            "",
            f"Recommendation: {recommendations['recommendation'].upper()}",
            f"Memory Count: {recommendations['memory_count']}",
        ]

        if "reasons" in recommendations:
            lines.extend(["", "Reasons:"])
            for reason in recommendations["reasons"]:
                lines.append(f"  • {reason}")

        if "memory_types" in recommendations:
            lines.extend(["", "Memory Types:"])
            for mem_type, count in recommendations["memory_types"].items():
                lines.append(f"  {mem_type}: {count}")

        if "total_size_bytes" in recommendations:
            size_mb = recommendations["total_size_bytes"] / (1024 * 1024)
            lines.append(f"\nTotal Size: {size_mb:.2f} MB")

        if "old_memory_percentage" in recommendations:
            lines.append(
                f"Old Memory Percentage: {recommendations['old_memory_percentage']:.1f}%"
            )

        if "estimated_duration_seconds" in recommendations:
            lines.append(
                f"Estimated Duration: {recommendations['estimated_duration_seconds']:.1f} seconds"
            )

        return [types.TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        error_msg = f"Error getting consolidation recommendations: {str(e)}"
        logger.error("%s\n%s", _sanitize_log_value(error_msg), traceback.format_exc())
        return [types.TextContent(type="text", text=error_msg)]


async def handle_scheduler_status(server, arguments: dict) -> List[types.TextContent]:
    """Handle scheduler status requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidation_scheduler:
        return [
            types.TextContent(type="text", text="Consolidation scheduler: DISABLED")
        ]

    try:
        # Get scheduler status
        status = await server.consolidation_scheduler.get_scheduler_status()

        if not status["enabled"]:
            return [
                types.TextContent(
                    type="text",
                    text=f"Scheduler: DISABLED\nReason: {status.get('reason', 'Unknown')}",
                )
            ]

        # Format status report
        lines = [
            f"Consolidation Scheduler Status: {'RUNNING' if status['running'] else 'STOPPED'}",
            "",
            "Scheduled Jobs:",
        ]

        for job in status["jobs"]:
            next_run = job["next_run_time"] or "Not scheduled"
            lines.append(f"  {job['name']}: {next_run}")

        lines.extend(
            [
                "",
                "Execution Statistics:",
                f"  Total jobs executed: {status['execution_stats']['total_jobs']}",
                f"  Successful jobs: {status['execution_stats']['successful_jobs']}",
                f"  Failed jobs: {status['execution_stats']['failed_jobs']}",
            ]
        )

        if status["last_execution_times"]:
            lines.extend(["", "Last Execution Times:"])
            for horizon, timestamp in status["last_execution_times"].items():
                lines.append(f"  {horizon}: {timestamp}")

        if status["recent_jobs"]:
            lines.extend(["", "Recent Jobs:"])
            for job in status["recent_jobs"][-5:]:  # Show last 5 jobs
                duration = (job["end_time"] - job["start_time"]).total_seconds()
                lines.append(
                    f"  {job['time_horizon']} ({job['status']}): {duration:.2f}s"
                )

        return [types.TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        error_msg = f"Error getting scheduler status: {str(e)}"
        logger.error("%s\n%s", _sanitize_log_value(error_msg), traceback.format_exc())
        return [types.TextContent(type="text", text=error_msg)]


async def handle_trigger_consolidation(
    server, arguments: dict
) -> List[types.TextContent]:
    """Handle manual consolidation trigger requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidation_scheduler:
        return [
            types.TextContent(
                type="text", text="Error: Consolidation scheduler not available"
            )
        ]

    try:
        time_horizon = arguments.get("time_horizon")
        immediate = arguments.get("immediate", True)

        if not time_horizon:
            return [
                types.TextContent(type="text", text="Error: time_horizon is required")
            ]

        if time_horizon not in [
            "daily",
            "weekly",
            "monthly",
            "quarterly",
            "yearly",
            "incremental",
        ]:
            return [types.TextContent(type="text", text="Error: Invalid time_horizon")]

        # Trigger consolidation
        success = await server.consolidation_scheduler.trigger_consolidation(
            time_horizon, immediate
        )

        if success:
            action = "triggered immediately" if immediate else "scheduled for later"
            return [
                types.TextContent(
                    type="text",
                    text=f"Successfully {action} {time_horizon} consolidation",
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text", text=f"Failed to trigger {time_horizon} consolidation"
                )
            ]

    except Exception as e:
        error_msg = f"Error triggering consolidation: {str(e)}"
        logger.error("%s\n%s", _sanitize_log_value(error_msg), traceback.format_exc())
        return [types.TextContent(type="text", text=error_msg)]


async def handle_pause_consolidation(
    server, arguments: dict
) -> List[types.TextContent]:
    """Handle consolidation pause requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidation_scheduler:
        return [
            types.TextContent(
                type="text", text="Error: Consolidation scheduler not available"
            )
        ]

    try:
        time_horizon = arguments.get("time_horizon")

        # Pause consolidation
        success = await server.consolidation_scheduler.pause_consolidation(time_horizon)

        if success:
            target = time_horizon or "all"
            return [
                types.TextContent(
                    type="text", text=f"Successfully paused {target} consolidation jobs"
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text", text="Failed to pause consolidation jobs"
                )
            ]

    except Exception as e:
        error_msg = f"Error pausing consolidation: {str(e)}"
        logger.error("%s\n%s", _sanitize_log_value(error_msg), traceback.format_exc())
        return [types.TextContent(type="text", text=error_msg)]


async def handle_resume_consolidation(
    server, arguments: dict
) -> List[types.TextContent]:
    """Handle consolidation resume requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidation_scheduler:
        return [
            types.TextContent(
                type="text", text="Error: Consolidation scheduler not available"
            )
        ]

    try:
        time_horizon = arguments.get("time_horizon")

        # Resume consolidation
        success = await server.consolidation_scheduler.resume_consolidation(
            time_horizon
        )

        if success:
            target = time_horizon or "all"
            return [
                types.TextContent(
                    type="text",
                    text=f"Successfully resumed {target} consolidation jobs",
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text", text="Failed to resume consolidation jobs"
                )
            ]

    except Exception as e:
        error_msg = f"Error resuming consolidation: {str(e)}"
        logger.error("%s\n%s", _sanitize_log_value(error_msg), traceback.format_exc())
        return [types.TextContent(type="text", text=error_msg)]


async def handle_memory_consolidate(server, arguments: dict) -> List[types.TextContent]:
    """
    Unified handler for memory consolidation management.

    Routes to appropriate handler based on action parameter.
    Consolidates all consolidation operations into a single tool.
    """
    action = arguments.get("action")

    if not action:
        return [
            types.TextContent(type="text", text="Error: action parameter is required")
        ]

    # Validate action
    valid_actions = [
        "run",
        "status",
        "recommend",
        "scheduler",
        "pause",
        "resume",
        "merge",
        "harvest",
    ]
    if action not in valid_actions:
        return [
            types.TextContent(
                type="text",
                text=f"Error: Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}",
            )
        ]

    try:
        # Skip CONSOLIDATION_ENABLED check for merge and harvest (they don't need the consolidator)
        if action not in ("merge", "harvest") and not CONSOLIDATION_ENABLED:
            return [types.TextContent(type="text", text="Consolidation is disabled")]

        # Route to appropriate handler based on action
        if action == "run":
            # Run consolidation (requires time_horizon)
            time_horizon = arguments.get("time_horizon")
            if not time_horizon:
                return [
                    types.TextContent(
                        type="text",
                        text="Error: time_horizon is required for 'run' action",
                    )
                ]

            return await handle_consolidate_memories(
                server, {"time_horizon": time_horizon}
            )

        elif action == "status":
            # Get consolidation system status
            return await handle_consolidation_status(server, {})

        elif action == "recommend":
            # Get recommendations (requires time_horizon)
            time_horizon = arguments.get("time_horizon")
            if not time_horizon:
                return [
                    types.TextContent(
                        type="text",
                        text="Error: time_horizon is required for 'recommend' action",
                    )
                ]

            return await handle_consolidation_recommendations(
                server, {"time_horizon": time_horizon}
            )

        elif action == "scheduler":
            # Get scheduler status
            return await handle_scheduler_status(server, {})

        elif action == "pause":
            # Pause consolidation (optional time_horizon)
            return await handle_pause_consolidation(
                server, {"time_horizon": arguments.get("time_horizon")}
            )

        elif action == "resume":
            # Resume consolidation (optional time_horizon)
            return await handle_resume_consolidation(
                server, {"time_horizon": arguments.get("time_horizon")}
            )

        elif action == "harvest":
            # On-demand session harvest from a given path, in-process (integrity:
            # uses the server's own storage/MemoryService, so graph/dedup/provenance
            # are intact — unlike a standalone script).
            import os as _os
            from pathlib import Path as _P
            raw_path = arguments.get("path") or _os.getenv("MCP_HARVEST_SESSION_DIR", "~/.kiro/sessions/cli")
            # Path-traversal guard: the client supplies path over MCP, so confine it
            # to known session roots (mirrors the base_dir check the HTTP-blocked
            # memory_harvest had; GHSA-7crr-2r7w-cpfm). resolve() collapses ../ and
            # symlinks before the containment check.
            resolved = _P(raw_path).expanduser().resolve()
            # Allowed roots are configurable (MCP_HARVEST_ALLOWED_ROOTS, ':'-separated)
            # so operators/tests can widen them; defaults cover the session locations.
            roots_env = _os.getenv("MCP_HARVEST_ALLOWED_ROOTS", "")
            root_specs = [r for r in roots_env.split(":") if r] or [
                "~/.kiro",
                _os.getenv("MCP_HARVEST_SESSION_DIR", "~/.kiro/sessions/cli"),
                "~/local-data",
            ]
            allowed_roots = [_P(r).expanduser().resolve() for r in root_specs]
            if not any(resolved == r or resolved.is_relative_to(r) for r in allowed_roots):
                return [types.TextContent(
                    type="text",
                    text=(f"Error: harvest path is outside allowed roots ({raw_path}). "
                          f"Set MCP_HARVEST_ALLOWED_ROOTS to widen."),
                )]
            path = str(resolved)
            sessions = int(arguments.get("sessions", 50))
            use_llm = arguments.get("use_llm", True)
            dry_run = arguments.get("dry_run", False)
            force = arguments.get("force_reharvest", False)
            if not server.storage:
                return [types.TextContent(type="text", text="Error: storage not available")]
            try:
                from ...harvest.harvester import SessionHarvester
                from ...harvest.models import HarvestConfig, harvest_config_from_env
                from ...services.memory_service import MemoryService
            except Exception as e:
                return [types.TextContent(type="text", text=f"Error: harvest module unavailable ({e})")]

            ms = MemoryService(server.storage)
            harvester = SessionHarvester(project_dir=path, memory_service=ms)
            cfg = harvest_config_from_env(
                sessions=sessions, 
                dry_run=dry_run, 
                use_llm=use_llm,
                min_confidence=0.65, 
                force_reharvest=force, 
                project_path=path
            )
            results = await harvester.harvest_and_store(cfg)
            n_sessions = len(results)
            candidates = sum(len(getattr(r, "candidates", []) or []) for r in results)
            stored = sum(getattr(r, "stored", 0) or 0 for r in results)
            mode = "DRY-RUN" if dry_run else "STORED"
            return [types.TextContent(
                type="text",
                text=(f"Harvest ({mode}) from {path}\n"
                      f"sessions: {n_sessions} | candidates: {candidates} | stored: {stored}"),
            )]

        elif action == "merge":
            raw_hashes = arguments.get("content_hashes", [])
            content_hashes = list(dict.fromkeys(raw_hashes)) if isinstance(raw_hashes, list) else raw_hashes
            merged_content = arguments.get("merged_content", "")

            # Validate content_hashes
            if (
                not content_hashes
                or not isinstance(content_hashes, list)
                or len(content_hashes) < 2
            ):
                return [
                    types.TextContent(
                        type="text",
                        text="Error: content_hashes must be an array with at least 2 hashes",
                    )
                ]

            if (
                not merged_content
                or not isinstance(merged_content, str)
                or not merged_content.strip()
            ):
                return [
                    types.TextContent(
                        type="text",
                        text="Error: merged_content is required and must be a non-empty string",
                    )
                ]

            tags = arguments.get("tags", [])
            memory_type = arguments.get("memory_type")

            # Verify all hashes exist and collect source data
            missing = []
            sources = []
            for h in content_hashes:
                existing = await server.storage.get_by_hash(h)
                if existing is None:
                    missing.append(h)
                else:
                    sources.append(existing)

            if missing:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: memories not found: {', '.join(missing)}",
                    )
                ]

            # Tag union: default to union of source tags when not explicitly provided
            if not tags:
                all_tags = set()
                for src in sources:
                    src_tags = getattr(src, "tags", [])
                    if isinstance(src_tags, str):
                        src_tags = [t.strip() for t in src_tags.split(",") if t.strip()]
                    if isinstance(src_tags, list):
                        all_tags.update(src_tags)
                tags = sorted(all_tags)

            # memory_type inheritance: default to first source's type when not specified
            if not memory_type and sources:
                memory_type = getattr(sources[0], "memory_type", None)

            from mcp_memory_service.models.memory import Memory
            from mcp_memory_service.utils.hashing import generate_content_hash

            new_hash = generate_content_hash(merged_content)
            memory = Memory(
                content=merged_content,
                content_hash=new_hash,
                tags=tags if isinstance(tags, list) else [],
                memory_type=memory_type,
            )

            # Store merged content FIRST (safety: store before delete).
            # store() returns (success, message) — the second element is prose,
            # never a hash. It was named result_hash here and passed straight
            # into the response, so callers got "Memory stored successfully"
            # where they expected the merged memory's hash (#112).
            stored, store_message = await server.storage.store(
                memory, skip_semantic_dedup=True
            )

            if not stored:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Error: failed to store merged memory: {store_message}",
                    )
                ]

            # Delete source memories
            deleted = []
            failed = []
            for h in content_hashes:
                result = await server.storage.delete_memories(content_hash=h)
                if result.get("success"):
                    deleted.append(h)
                else:
                    failed.append(h)

            result = {
                "ok": len(failed) == 0,
                "content_hash": new_hash,
                "merged": content_hashes,
                "deleted": deleted,
                "failed": failed if failed else None,
            }

            return [types.TextContent(type="text", text=json.dumps(result))]

        else:
            # Should never reach here due to validation above
            return [
                types.TextContent(type="text", text=f"Error: Unknown action '{action}'")
            ]

    except Exception as e:
        error_msg = f"Error in memory_consolidate action '{_sanitize_log_value(action)}': {str(e)}"
        logger.error("%s\n%s", _sanitize_log_value(error_msg), traceback.format_exc())
        return [types.TextContent(type="text", text=error_msg)]
