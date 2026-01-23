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

import logging
import traceback
from typing import List

from mcp import types
from ...config import CONSOLIDATION_ENABLED

logger = logging.getLogger(__name__)


async def handle_consolidate_memories(server, arguments: dict) -> List[types.TextContent]:
    """Handle memory consolidation requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidator:
        return [types.TextContent(type="text", text="Error: Consolidation system not available")]

    try:
        time_horizon = arguments.get("time_horizon")
        if not time_horizon:
            return [types.TextContent(type="text", text="Error: time_horizon is required")]

        if time_horizon not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            return [types.TextContent(type="text", text="Error: Invalid time_horizon. Must be one of: daily, weekly, monthly, quarterly, yearly")]

        logger.info(f"Starting {time_horizon} consolidation")

        # Run consolidation
        report = await server.consolidator.consolidate(time_horizon)

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
            result += f"\n\nWarnings/Errors:\n" + "\n".join(f"- {error}" for error in report.errors)

        return [types.TextContent(type="text", text=result)]

    except Exception as e:
        error_msg = f"Error during consolidation: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_consolidation_status(server, arguments: dict) -> List[types.TextContent]:
    """Handle consolidation status requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidator:
        return [types.TextContent(type="text", text="Consolidation system: DISABLED")]

    try:
        # Get health check from consolidator
        health = await server.consolidator.health_check()

        # Format status report
        status_lines = [
            f"Consolidation System Status: {health['status'].upper()}",
            f"Last Updated: {health['timestamp']}",
            "",
            "Component Health:"
        ]

        for component, component_health in health['components'].items():
            status = component_health['status']
            status_lines.append(f"  {component}: {status.upper()}")
            if status == 'unhealthy' and 'error' in component_health:
                status_lines.append(f"    Error: {component_health['error']}")

        status_lines.extend([
            "",
            "Statistics:",
            f"  Total consolidation runs: {health['statistics']['total_runs']}",
            f"  Successful runs: {health['statistics']['successful_runs']}",
            f"  Total memories processed: {health['statistics']['total_memories_processed']}",
            f"  Total associations created: {health['statistics']['total_associations_created']}",
            f"  Total clusters created: {health['statistics']['total_clusters_created']}",
            f"  Total memories compressed: {health['statistics']['total_memories_compressed']}",
            f"  Total memories archived: {health['statistics']['total_memories_archived']}"
        ])

        if health['last_consolidation_times']:
            status_lines.extend([
                "",
                "Last Consolidation Times:"
            ])
            for horizon, timestamp in health['last_consolidation_times'].items():
                status_lines.append(f"  {horizon}: {timestamp}")

        return [types.TextContent(type="text", text="\n".join(status_lines))]

    except Exception as e:
        error_msg = f"Error getting consolidation status: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_consolidation_recommendations(server, arguments: dict) -> List[types.TextContent]:
    """Handle consolidation recommendation requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidator:
        return [types.TextContent(type="text", text="Error: Consolidation system not available")]

    try:
        time_horizon = arguments.get("time_horizon")
        if not time_horizon:
            return [types.TextContent(type="text", text="Error: time_horizon is required")]

        if time_horizon not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            return [types.TextContent(type="text", text="Error: Invalid time_horizon")]

        # Get recommendations
        recommendations = await server.consolidator.get_consolidation_recommendations(time_horizon)

        # Format response
        lines = [
            f"Consolidation Recommendations for {time_horizon} horizon:",
            "",
            f"Recommendation: {recommendations['recommendation'].upper()}",
            f"Memory Count: {recommendations['memory_count']}",
        ]

        if 'reasons' in recommendations:
            lines.extend([
                "",
                "Reasons:"
            ])
            for reason in recommendations['reasons']:
                lines.append(f"  • {reason}")

        if 'memory_types' in recommendations:
            lines.extend([
                "",
                "Memory Types:"
            ])
            for mem_type, count in recommendations['memory_types'].items():
                lines.append(f"  {mem_type}: {count}")

        if 'total_size_bytes' in recommendations:
            size_mb = recommendations['total_size_bytes'] / (1024 * 1024)
            lines.append(f"\nTotal Size: {size_mb:.2f} MB")

        if 'old_memory_percentage' in recommendations:
            lines.append(f"Old Memory Percentage: {recommendations['old_memory_percentage']:.1f}%")

        if 'estimated_duration_seconds' in recommendations:
            lines.append(f"Estimated Duration: {recommendations['estimated_duration_seconds']:.1f} seconds")

        return [types.TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        error_msg = f"Error getting consolidation recommendations: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_scheduler_status(server, arguments: dict) -> List[types.TextContent]:
    """Handle scheduler status requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidation_scheduler:
        return [types.TextContent(type="text", text="Consolidation scheduler: DISABLED")]

    try:
        # Get scheduler status
        status = await server.consolidation_scheduler.get_scheduler_status()

        if not status['enabled']:
            return [types.TextContent(type="text", text=f"Scheduler: DISABLED\nReason: {status.get('reason', 'Unknown')}")]

        # Format status report
        lines = [
            f"Consolidation Scheduler Status: {'RUNNING' if status['running'] else 'STOPPED'}",
            "",
            "Scheduled Jobs:"
        ]

        for job in status['jobs']:
            next_run = job['next_run_time'] or 'Not scheduled'
            lines.append(f"  {job['name']}: {next_run}")

        lines.extend([
            "",
            "Execution Statistics:",
            f"  Total jobs executed: {status['execution_stats']['total_jobs']}",
            f"  Successful jobs: {status['execution_stats']['successful_jobs']}",
            f"  Failed jobs: {status['execution_stats']['failed_jobs']}"
        ])

        if status['last_execution_times']:
            lines.extend([
                "",
                "Last Execution Times:"
            ])
            for horizon, timestamp in status['last_execution_times'].items():
                lines.append(f"  {horizon}: {timestamp}")

        if status['recent_jobs']:
            lines.extend([
                "",
                "Recent Jobs:"
            ])
            for job in status['recent_jobs'][-5:]:  # Show last 5 jobs
                duration = (job['end_time'] - job['start_time']).total_seconds()
                lines.append(f"  {job['time_horizon']} ({job['status']}): {duration:.2f}s")

        return [types.TextContent(type="text", text="\n".join(lines))]

    except Exception as e:
        error_msg = f"Error getting scheduler status: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_trigger_consolidation(server, arguments: dict) -> List[types.TextContent]:
    """Handle manual consolidation trigger requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidation_scheduler:
        return [types.TextContent(type="text", text="Error: Consolidation scheduler not available")]

    try:
        time_horizon = arguments.get("time_horizon")
        immediate = arguments.get("immediate", True)

        if not time_horizon:
            return [types.TextContent(type="text", text="Error: time_horizon is required")]

        if time_horizon not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
            return [types.TextContent(type="text", text="Error: Invalid time_horizon")]

        # Trigger consolidation
        success = await server.consolidation_scheduler.trigger_consolidation(time_horizon, immediate)

        if success:
            action = "triggered immediately" if immediate else "scheduled for later"
            return [types.TextContent(type="text", text=f"Successfully {action} {time_horizon} consolidation")]
        else:
            return [types.TextContent(type="text", text=f"Failed to trigger {time_horizon} consolidation")]

    except Exception as e:
        error_msg = f"Error triggering consolidation: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_pause_consolidation(server, arguments: dict) -> List[types.TextContent]:
    """Handle consolidation pause requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidation_scheduler:
        return [types.TextContent(type="text", text="Error: Consolidation scheduler not available")]

    try:
        time_horizon = arguments.get("time_horizon")

        # Pause consolidation
        success = await server.consolidation_scheduler.pause_consolidation(time_horizon)

        if success:
            target = time_horizon or "all"
            return [types.TextContent(type="text", text=f"Successfully paused {target} consolidation jobs")]
        else:
            return [types.TextContent(type="text", text="Failed to pause consolidation jobs")]

    except Exception as e:
        error_msg = f"Error pausing consolidation: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_resume_consolidation(server, arguments: dict) -> List[types.TextContent]:
    """Handle consolidation resume requests."""
    if not CONSOLIDATION_ENABLED or not server.consolidation_scheduler:
        return [types.TextContent(type="text", text="Error: Consolidation scheduler not available")]

    try:
        time_horizon = arguments.get("time_horizon")

        # Resume consolidation
        success = await server.consolidation_scheduler.resume_consolidation(time_horizon)

        if success:
            target = time_horizon or "all"
            return [types.TextContent(type="text", text=f"Successfully resumed {target} consolidation jobs")]
        else:
            return [types.TextContent(type="text", text="Failed to resume consolidation jobs")]

    except Exception as e:
        error_msg = f"Error resuming consolidation: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]


async def handle_memory_consolidate(server, arguments: dict) -> List[types.TextContent]:
    """
    Unified handler for memory consolidation management.

    Routes to appropriate handler based on action parameter.
    Consolidates all consolidation operations into a single tool.
    """
    action = arguments.get("action")

    if not action:
        return [types.TextContent(type="text", text="Error: action parameter is required")]

    # Validate action
    valid_actions = ["run", "status", "recommend", "scheduler", "pause", "resume"]
    if action not in valid_actions:
        return [types.TextContent(
            type="text",
            text=f"Error: Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"
        )]

    try:
        # Route to appropriate handler based on action
        if action == "run":
            # Run consolidation (requires time_horizon)
            time_horizon = arguments.get("time_horizon")
            if not time_horizon:
                return [types.TextContent(type="text", text="Error: time_horizon is required for 'run' action")]

            return await handle_consolidate_memories(server, {"time_horizon": time_horizon})

        elif action == "status":
            # Get consolidation system status
            return await handle_consolidation_status(server, {})

        elif action == "recommend":
            # Get recommendations (requires time_horizon)
            time_horizon = arguments.get("time_horizon")
            if not time_horizon:
                return [types.TextContent(type="text", text="Error: time_horizon is required for 'recommend' action")]

            return await handle_consolidation_recommendations(server, {"time_horizon": time_horizon})

        elif action == "scheduler":
            # Get scheduler status
            return await handle_scheduler_status(server, {})

        elif action == "pause":
            # Pause consolidation (optional time_horizon)
            return await handle_pause_consolidation(server, {
                "time_horizon": arguments.get("time_horizon")
            })

        elif action == "resume":
            # Resume consolidation (optional time_horizon)
            return await handle_resume_consolidation(server, {
                "time_horizon": arguments.get("time_horizon")
            })

        else:
            # Should never reach here due to validation above
            return [types.TextContent(type="text", text=f"Error: Unknown action '{action}'")]

    except Exception as e:
        error_msg = f"Error in memory_consolidate action '{action}': {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return [types.TextContent(type="text", text=error_msg)]
