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

"""APScheduler integration for autonomous consolidation operations."""

import logging
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

try:
    from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    from apscheduler.executors.asyncio import AsyncIOExecutor
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.schedulers.base import STATE_RUNNING
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

from ..compat import _sanitize_log_value
from .belief_service import BeliefService
from .consolidator import DreamInspiredConsolidator


def sessions_to_track(results) -> set:
    """Session ids that should be marked harvested (RFC-provenance R7).

    Only sessions that stored at least one memory are tracked. A session that
    stored nothing stays pending so a later run re-harvests it — this covers the
    retryable failure we care about (the LLM chain was down and every candidate
    was dropped) without risking data loss.

    Known trade-off: a deterministically empty session (nothing harvestable) also
    stays pending and is reselected each tick. `found` cannot tell the two apart
    here — a transient LLM-rewrite failure also collapses to ``found==0`` (the
    rewriter drops candidates it can't rewrite), so keying off ``found`` would
    silently discard recoverable candidates. Distinguishing the two needs the
    harvester to surface pre-rewrite extraction / rewrite-failure state on
    HarvestResult; tracked as a separate follow-up. Re-processing an empty
    session is cheaper than losing data, so this stays conservative.
    """
    def _count(r, name) -> int:
        value = getattr(r, name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    def _handled(r) -> bool:
        stored, skipped, found = _count(r, "stored"), _count(r, "skipped"), _count(r, "found")
        if stored > 0:
            return True
        # #1560: candidates the server refused because they were deleted are handled,
        # not failed — a session made only of those must not be reselected every tick.
        # Any candidate that is neither stored nor skipped keeps the session pending.
        # `found` is counted AFTER the LLM rewrite, so a candidate the rewrite lost is
        # not in it: `rewrite_dropped` (HarvestResult) is the evidence that rewriting
        # completed for every candidate. Without it a skipped-only session stays pending.
        return (skipped > 0 and _count(r, "rewrite_dropped") == 0
                and stored + skipped >= found)

    return {
        r.session_id
        for r in results
        if getattr(r, "session_id", None) and _handled(r)
    }


def _classify_sync_failure(
    exc: Exception,
    handshake_complete: bool,
    event_failure_message: str | None = None,
    probe: bool = False,
) -> tuple[str, str]:
    """Map sync exceptions to the stable status and user-facing message."""
    if event_failure_message is not None:
        return "error", event_failure_message

    from ..storage.base import EmbeddingModelMismatchError
    from ..storage.sync.protocol import SyncProtocolCompatibilityError

    if isinstance(exc, (EmbeddingModelMismatchError, SyncProtocolCompatibilityError)):
        return "incompatible", str(exc)
    if handshake_complete:
        return "error", "Sync cycle failed after capability negotiation."
    if probe:
        return (
            "unavailable",
            "Capability probe failed; check peer connectivity and credentials.",
        )
    return (
        "unavailable",
        "Capability negotiation failed; check peer connectivity and credentials.",
    )


class ConsolidationScheduler:
    """
    Scheduler for autonomous consolidation operations.
    
    Integrates with APScheduler to run consolidation operations at specified intervals
    based on time horizons (daily, weekly, monthly, quarterly, yearly).
    """
    
    def __init__(
        self, 
        consolidator: DreamInspiredConsolidator,
        schedule_config: Dict[str, str],
        enabled: bool = True
    ):
        self.consolidator = consolidator
        self.schedule_config = schedule_config
        self.enabled = enabled
        self.logger = logging.getLogger(__name__)
        
        # Job execution tracking
        self.job_history = []
        self.last_execution_times = {}
        self.execution_stats = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0
        }
        self._sync_peer_status: Dict[str, Dict[str, Any]] = {}
        self._sync_peer_operation_versions: dict[str, int] = {}
        self._sync_peer_probe_versions: dict[str, int] = {}
        
        # Initialize scheduler if APScheduler is available
        if APSCHEDULER_AVAILABLE and enabled:
            self.scheduler = AsyncIOScheduler(
                jobstores={'default': MemoryJobStore()},
                executors={'default': AsyncIOExecutor()},
                job_defaults={
                    'coalesce': True,  # Combine multiple pending executions
                    'max_instances': 1,  # Only one instance of each job at a time
                    'misfire_grace_time': 3600  # 1 hour grace period for missed jobs
                }
            )
            
            # Add event listeners
            self.scheduler.add_listener(self._job_executed_listener, EVENT_JOB_EXECUTED)
            self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)
        else:
            self.scheduler = None
            if not APSCHEDULER_AVAILABLE:
                self.logger.warning("APScheduler not available - consolidation scheduling disabled")
            elif not enabled:
                self.logger.info("Consolidation scheduling disabled by configuration")

        # Wire scheduler reference into the health monitor (if present)
        health_monitor = getattr(self.consolidator, 'health_monitor', None)
        if health_monitor is not None and hasattr(health_monitor, 'attach_scheduler'):
            health_monitor.attach_scheduler(self)
    
    async def start(self) -> bool:
        """Start the consolidation scheduler."""
        if not self.scheduler:
            return False
        
        try:
            # Add consolidation jobs based on configuration
            self._schedule_consolidation_jobs()

            # Add scheduled session-harvest job (opt-in via MCP_HARVEST_SCHEDULE)
            self._schedule_harvest_job()

            # Add scheduled delta-sync cycle job (opt-in via MCP_SYNC_SCHEDULE) — Phase 4d, ADR-0028
            self._schedule_sync_job()

            # Start the scheduler
            self.scheduler.start()
            self.logger.info("Consolidation scheduler started successfully")
            
            # Log scheduled jobs
            jobs = self.scheduler.get_jobs()
            for job in jobs:
                self.logger.info("Scheduled job: %s - next run: %s", _sanitize_log_value(job.id), _sanitize_log_value(job.next_run_time))
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to start consolidation scheduler: %s", _sanitize_log_value(e))
            return False
    
    async def stop(self) -> bool:
        """Stop the consolidation scheduler."""
        if not self.scheduler:
            return True
        
        try:
            self.scheduler.shutdown(wait=True)
            self.logger.info("Consolidation scheduler stopped")
            return True
        except Exception as e:
            self.logger.error("Error stopping consolidation scheduler: %s", _sanitize_log_value(e))
            return False
    
    def _schedule_consolidation_jobs(self):
        """Schedule consolidation jobs based on configuration."""
        time_horizons = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        
        for horizon in time_horizons:
            schedule_spec = self.schedule_config.get(horizon, 'disabled')
            
            if schedule_spec == 'disabled':
                self.logger.debug("Consolidation for %s horizon is disabled", _sanitize_log_value(horizon))
                continue
            
            try:
                trigger = self._create_trigger(horizon, schedule_spec)
                if trigger:
                    job_id = f"consolidation_{horizon}"
                    self.scheduler.add_job(
                        func=self._run_consolidation_job,
                        trigger=trigger,
                        args=[horizon],
                        id=job_id,
                        name=f"Consolidation - {horizon.title()}",
                        replace_existing=True
                    )
                    self.logger.info("Scheduled %s consolidation: %s", _sanitize_log_value(horizon), _sanitize_log_value(schedule_spec))
                
            except Exception as e:
                self.logger.error("Error scheduling %s consolidation: %s", _sanitize_log_value(horizon), _sanitize_log_value(e))
    

    def _schedule_harvest_job(self):
        """Schedule autonomous session harvest (opt-in via MCP_HARVEST_SCHEDULE).

        The harvest handler (memory_harvest) is local-only and never exposed over
        remote transports (confused-deputy protection). The scheduler runs
        in-process on the server host, which already has the filesystem access
        the harvester needs — so autonomous harvest belongs here, next to the
        consolidation cadence it piggybacks on, rather than as an external cron
        calling a blocked tool.

        MCP_HARVEST_SCHEDULE accepts an interval like "6h", "30m", "90s", or a
        plain number of hours ("6"). Unset/blank/"disabled" → no job (default).
        """
        schedule_spec = os.getenv("MCP_HARVEST_SCHEDULE", "").strip()
        if not schedule_spec or schedule_spec.lower() == "disabled":
            self.logger.debug("Scheduled session harvest disabled (MCP_HARVEST_SCHEDULE unset)")
            return

        seconds = self._parse_interval_seconds(schedule_spec)
        if not seconds or seconds <= 0:
            self.logger.error(
                "Invalid MCP_HARVEST_SCHEDULE=%r — expected e.g. '6h', '30m', '90s' or hours; skipping",
                schedule_spec,
            )
            return

        try:
            self.scheduler.add_job(
                func=self._run_scheduled_harvest,
                trigger=IntervalTrigger(seconds=seconds),
                id="session_harvest",
                name="Scheduled Session Harvest",
                replace_existing=True,
            )
            self.logger.info("Scheduled session harvest every %ss (MCP_HARVEST_SCHEDULE=%s)", seconds, schedule_spec)
        except Exception as e:
            self.logger.error(
                "Error scheduling session harvest: %s", _sanitize_log_value(e)
            )

    def _schedule_sync_job(self):
        """Schedule the delta-sync cycle job (opt-in via MCP_SYNC_SCHEDULE). Phase 4d, ADR-0028.

        Mirrors _schedule_harvest_job: in-process interval
        job on the consolidation scheduler, off by default (unset/blank/"disabled" → no job).
        Runs pull + push against each peer in MCP_SYNC_PEERS. Only spokes set peers; the hub
        sets none and thus schedules an effectively empty cycle (passive pivot, ADR-0027).
        """
        if not self.scheduler:
            self.logger.debug("Sync scheduling skipped - scheduler not available")
            return

        schedule_spec = os.getenv("MCP_SYNC_SCHEDULE", "").strip()
        if not schedule_spec or schedule_spec.lower() == "disabled":
            self.logger.debug("Delta-sync scheduling disabled (MCP_SYNC_SCHEDULE unset)")
            return

        seconds = self._parse_interval_seconds(schedule_spec)
        if not seconds or seconds <= 0:
            self.logger.error(
                "Invalid MCP_SYNC_SCHEDULE=%r — expected e.g. '15m', '30m', '6h', '90s' or hours; skipping",
                schedule_spec,
            )
            return

        try:
            # Probe peers once after the service starts without delaying service
            # startup. Every scheduled cycle repeats the handshake before data I/O.
            self.scheduler.add_job(
                func=self._probe_sync_peers,
                trigger=DateTrigger(run_date=datetime.now(timezone.utc)),
                id="delta_sync_probe",
                name="Delta-Sync Capability Probe",
                replace_existing=True,
                max_instances=1,
            )
            self.scheduler.add_job(
                func=self._run_sync_cycle,
                trigger=IntervalTrigger(seconds=seconds),
                id="delta_sync",
                name="Delta-Sync Cycle (pull + push)",
                replace_existing=True,
            )
            self.logger.info("Scheduled delta-sync every %ss (MCP_SYNC_SCHEDULE=%s)", seconds, schedule_spec)
        except Exception as e:
            self.logger.error("Error scheduling delta-sync: %s", _sanitize_log_value(e))

    def _begin_sync_peer_operation(self, peer_id: str) -> int:
        """Return a token that lets newer probe/cycle results supersede older ones."""
        operation_version = self._sync_peer_operation_versions.get(peer_id, 0) + 1
        self._sync_peer_operation_versions[peer_id] = operation_version
        return operation_version

    def _set_sync_peer_status(
        self,
        peer_id: str,
        status: str,
        message: str,
        compatibility: dict[str, Any] | None = None,
        *,
        last_sync_at: str | None = None,
        operation_version: int | None = None,
        probe_version: int | None = None,
    ) -> None:
        """Store the latest safe-to-expose delta-sync result for one peer."""
        # Startup probes have their own generation. Once a cycle has started,
        # its result owns the peer status, even if a slow probe reaches it later.
        if probe_version is not None and (
            probe_version != self._sync_peer_probe_versions.get(peer_id)
            or self._sync_peer_operation_versions.get(peer_id, 0) > 0
        ):
            return
        if (
            operation_version is not None
            and operation_version != self._sync_peer_operation_versions.get(peer_id)
        ):
            return
        previous = self._sync_peer_status.get(peer_id, {})
        result = {
            **previous,
            "peer_id": peer_id,
            "status": status,
            "message": message,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "negotiated_sync_protocol": previous.get("negotiated_sync_protocol"),
            "event_schema_version": previous.get("event_schema_version"),
            "peer_version": previous.get("peer_version"),
            "last_sync_at": last_sync_at or previous.get("last_sync_at"),
        }
        if compatibility is not None:
            result.update({
                "negotiated_sync_protocol": compatibility.get("negotiated_sync_protocol"),
                "event_schema_version": compatibility.get("event_schema_version"),
                "peer_version": compatibility.get("version"),
            })
        self._sync_peer_status[peer_id] = result

    def get_sync_peer_status(self) -> list[dict[str, Any]]:
        """Return the most recent startup/cycle result for each delta-sync peer."""
        return [deepcopy(self._sync_peer_status[key]) for key in sorted(self._sync_peer_status)]

    @staticmethod
    def delta_sync_configuration() -> dict[str, Any]:
        """Describe configured peers without creating clients or exposing credentials."""
        schedule = os.getenv("MCP_SYNC_SCHEDULE", "").strip()
        peer_ids = sorted({p.strip() for p in os.getenv("MCP_SYNC_PEERS", "").split(",") if p.strip()})
        raw_url = os.getenv("MCP_SYNC_PEER_URL", "").strip()
        try:
            parts = urlsplit(raw_url)
            valid_url = parts.scheme in ("http", "https") and bool(parts.hostname)
            # Remove userinfo, query credentials, and fragments from the display URL.
            url = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", "")) if valid_url else ""
        except ValueError:
            url = ""
        return {
            "configured": bool(peer_ids or schedule.lower() not in ("", "disabled")),
            "schedule": schedule or "disabled",
            "enabled": False,
            "next_run_at": None,
            "interval_seconds": None,
            "status": "disabled",
            "peers": [{"peer_id": peer_id, "url": url, "status": "pending" if url else "error",
                       "message": "Waiting for first cycle" if url else "MCP_SYNC_PEER_URL is missing or invalid"}
                      for peer_id in peer_ids],
        }

    def get_delta_sync_status(self) -> dict[str, Any]:
        """Expose actual scheduling and the latest in-memory cycle snapshots."""
        result = self.delta_sync_configuration()
        job = self.scheduler.get_job("delta_sync") if self.scheduler else None
        next_run = getattr(job, "next_run_time", None)
        result["enabled"] = bool(job and next_run and self.scheduler.state == STATE_RUNNING)
        result["next_run_at"] = next_run.isoformat() if result["enabled"] else None
        if job is not None:
            result["interval_seconds"] = int(job.trigger.interval.total_seconds())
        for peer in result["peers"]:
            peer.update(deepcopy(self._sync_peer_status.get(peer["peer_id"], {})))
            peer["health"] = peer.get("cycle_status", "pending")
            if peer["status"] in ("error", "incompatible", "unavailable"):
                peer["health"] = "error"
        states = {peer["health"] for peer in result["peers"]}
        result["status"] = ("syncing" if "syncing" in states else "error" if "error" in states
                            else "disabled" if not result["enabled"] else "pending" if not states or "pending" in states
                            else "synced")
        return result

    async def _broadcast_delta_sync_event(
        self,
        peer_id: str,
        pulled: Optional[int],
        pushed: Optional[int],
        time_taken_seconds: float,
        success: bool,
    ) -> None:
        """Best-effort notification; web failures must not stop scheduled sync."""
        try:
            # Lazy import: web imports consolidation, so consolidation must not
            # import web at module level (same pattern as storage/hybrid.py).
            from ..web.sse import sse_manager, create_sync_delta_completed_event
        except ImportError:
            self.logger.debug(
                "SSE unavailable; skipping sync_delta_completed broadcast"
            )
            return
        try:
            event = create_sync_delta_completed_event(
                peer_id=str(peer_id),
                pulled=int(pulled) if pulled is not None else None,
                pushed=int(pushed) if pushed is not None else None,
                time_taken_seconds=float(time_taken_seconds or 0.0),
                success=success,
            )
            await sse_manager.broadcast_event(event)
            self.logger.debug(
                "Broadcast sync_delta_completed for peer %s (pulled=%s, pushed=%s, success=%s)",
                _sanitize_log_value(peer_id),
                pulled,
                pushed,
                success,
            )
        except Exception as exc:
            self.logger.debug(
                "Failed to broadcast sync_delta_completed for peer %s: %s",
                _sanitize_log_value(peer_id), _sanitize_log_value(exc),
            )

    async def _probe_sync_peers(self) -> None:
        """Best-effort startup capability probe; never blocks application startup."""
        storage = getattr(self.consolidator, "storage", None)
        if storage is None:
            self.logger.debug("Delta-sync capability probe skipped: storage unavailable")
            return

        for peer_id, peer in self._resolve_sync_peers():
            probe_version = self._sync_peer_probe_versions.get(peer_id, 0) + 1
            self._sync_peer_probe_versions[peer_id] = probe_version
            try:
                await peer.initialize()
                compatibility = await peer.check_sync_compatibility(storage)
                self._set_sync_peer_status(
                    peer_id,
                    "compatible",
                    "Peer capabilities are compatible",
                    compatibility,
                    probe_version=probe_version,
                )
            except Exception as exc:
                status, message = _classify_sync_failure(
                    exc, handshake_complete=False, probe=True
                )
                self._set_sync_peer_status(
                    peer_id,
                    status,
                    message,
                    probe_version=probe_version,
                )
                self.logger.warning(
                    "Delta-sync capability probe failed for peer %s: %s",
                    _sanitize_log_value(peer_id), _sanitize_log_value(exc),
                )
            finally:
                try:
                    await peer.close()
                except Exception as exc:
                    self.logger.debug(
                        "Could not close sync peer %s after capability probe: %s",
                        _sanitize_log_value(peer_id), _sanitize_log_value(exc),
                    )

    def _resolve_sync_peers(self):
        """Build (peer_id, RemoteHTTPStorage) list from MCP_SYNC_PEERS. Option 1 (single hub):
        peer_id -> base URL + auth from existing envs. Returns [] when unset (hub/no-peer)."""
        raw = os.getenv("MCP_SYNC_PEERS", "").strip()
        if not raw:
            return []
        from ..storage.remote_http import RemoteHTTPStorage
        # Dedicated var for the PEER's API key — NOT MCP_API_KEY. MCP_API_KEY on this
        # process ALSO turns on auth enforcement for THIS server's own /mcp endpoint
        # (middleware: `if not API_KEY: allow`), which would break every local MCP client.
        # Keep the sync client's hub credential separate from the server's own auth gate.
        api_key = os.getenv("MCP_SYNC_PEER_API_KEY") or None
        basic_user = os.getenv("MCP_SYNC_BASIC_USER") or None
        basic_pass = os.getenv("MCP_SYNC_BASIC_PASS") or None
        base_url = os.getenv("MCP_SYNC_PEER_URL", "").strip()
        peers = []
        for peer_id in (p.strip() for p in raw.split(",") if p.strip()):
            if not base_url:
                self.logger.error("MCP_SYNC_PEERS set but MCP_SYNC_PEER_URL missing; skipping %s", peer_id)
                continue
            auth_style = "x-api-key" if (api_key and basic_user) else ("bearer" if api_key else "x-api-key")
            peer = RemoteHTTPStorage(
                base_url=base_url, api_key=api_key, auth_style=auth_style,
                basic_user=basic_user, basic_pass=basic_pass,
            )
            peers.append((peer_id, peer))
        return peers

    async def _run_sync_cycle(self):
        """Run one delta-sync cycle: for each peer, pull then push. Phase 4d, ADR-0028.

        Resilient (R5): a peer error is logged and the cycle continues; never re-raises
        (the scheduler must survive). Cursors guarantee resumption next interval.
        """
        storage = getattr(self.consolidator, "storage", None)
        if storage is None:
            self.logger.warning("Delta-sync cycle skipped: consolidator has no storage")
            return
        peers = self._resolve_sync_peers()
        if not peers:
            self.logger.debug("Delta-sync cycle: no peers configured (MCP_SYNC_PEERS unset)")
            return
        from ..storage.sync.orchestrator import push_to_peer, sync_from_peer
        for peer_id, peer in peers:
            await self._run_sync_peer_cycle(
                storage, peer_id, peer, sync_from_peer, push_to_peer
            )

    async def _run_sync_peer_cycle(
        self,
        storage: Any,
        peer_id: str,
        peer: Any,
        sync_from_peer: Any,
        push_to_peer: Any,
    ) -> None:
        """Negotiate, exchange events, and record one peer's cycle result."""
        operation_version = self._begin_sync_peer_operation(peer_id)
        handshake_complete = False
        compatibility = None
        event_failure_message = None
        cycle_start = datetime.now(timezone.utc).isoformat()
        cycle = {"cycle_status": "syncing", "last_cycle_started_at": cycle_start,
                 "last_cycle_at": None, "last_error": None, "pull": None, "push": None,
                 "backlog": None}
        self._sync_peer_status.setdefault(peer_id, {"peer_id": peer_id}).update(cycle)
        try:
            await peer.initialize()
            compatibility = await peer.check_sync_compatibility(storage)
            handshake_complete = True
            self._set_sync_peer_status(
                peer_id,
                "compatible",
                "Peer capabilities are compatible",
                compatibility,
                operation_version=operation_version,
            )
            pull = await sync_from_peer(storage, peer, peer_id=peer_id, limit=100)
            cycle["pull"] = {"received": getattr(pull, "events_received", None),
                             "applied": pull.events_applied, "cursor": getattr(pull, "final_cursor", None)}
            push = await push_to_peer(storage, peer, peer_id=peer_id, limit=100)
            cycle["push"] = {"pushed": push.events_pushed, "cursor": getattr(push, "final_seq", None)}
            pull_failed = getattr(pull, "events_failed", 0) or 0
            push_failed = getattr(push, "events_failed", 0) or 0
            if pull_failed or push_failed:
                event_failure_message = (
                    "Peer sync reported failed events "
                    f"(pull={pull_failed}, push={push_failed})."
                )
                raise RuntimeError(event_failure_message)
            self._set_sync_peer_status(
                peer_id,
                "compatible",
                "Peer sync cycle completed",
                compatibility,
                last_sync_at=datetime.now(timezone.utc).isoformat(),
                operation_version=operation_version,
            )
            self.logger.info(
                "Delta-sync %s: pulled %s applied, pushed %s",
                _sanitize_log_value(peer_id),
                getattr(pull, "events_applied", "?"),
                getattr(push, "events_pushed", "?"),
            )
            cycle["cycle_status"] = "synced"
        except Exception as exc:
            self.execution_stats['failed_jobs'] += 1
            status, message = _classify_sync_failure(
                exc, handshake_complete, event_failure_message
            )
            cycle.update(cycle_status="error", last_error=message)
            self._set_sync_peer_status(
                peer_id,
                status,
                message,
                compatibility,
                operation_version=operation_version,
            )
            self.logger.error(
                "Delta-sync cycle failed for peer %s: %s",
                _sanitize_log_value(peer_id), _sanitize_log_value(exc),
            )
        finally:
            if operation_version == self._sync_peer_operation_versions.get(peer_id):
                cycle["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
                self._sync_peer_status[peer_id].update(cycle)
                # A direction that raised may have committed earlier pages;
                # without a returned summary its count is unknown, not zero.
                pull_result = cycle["pull"] or {}
                push_result = cycle["push"] or {}
                await self._broadcast_delta_sync_event(
                    peer_id=peer_id,
                    pulled=pull_result.get("received"),
                    pushed=push_result.get("pushed"),
                    time_taken_seconds=(
                        datetime.now(timezone.utc) - datetime.fromisoformat(cycle_start)
                    ).total_seconds(),
                    success=cycle["cycle_status"] == "synced",
                )
            try:
                await peer.close()
            except Exception as exc:
                # Best-effort cleanup: a failing close() must not mask the cycle result
                # or abort the remaining peers. Nothing actionable to recover here.
                self.logger.debug(
                    "Could not close sync peer %s after cycle: %s",
                    _sanitize_log_value(peer_id), _sanitize_log_value(exc),
                )

    @staticmethod
    def _parse_interval_seconds(spec: str) -> Optional[int]:
        """Parse an interval spec into seconds. Accepts '6h', '30m', '90s', or bare hours."""
        spec = spec.strip().lower()
        try:
            if spec.endswith("h"):
                return int(float(spec[:-1]) * 3600)
            if spec.endswith("m"):
                return int(float(spec[:-1]) * 60)
            if spec.endswith("s"):
                return int(float(spec[:-1]))
            return int(float(spec) * 3600)  # bare number = hours
        except (ValueError, TypeError):
            return None

    async def _run_scheduled_harvest(self):
        """Execute an autonomous session harvest, in-process, and store results.

        Mirrors the memory_harvest handler but runs on the server's own cadence:
        reads MCP_HARVEST_SESSION_DIR, harvests the delta (the harvest tracker in
        harvest_and_store skips already-processed sessions), and bridges results
        into the observation/belief pipeline via auto_commit.
        """
        storage = getattr(self.consolidator, "storage", None)
        if storage is None:
            self.logger.warning("Scheduled harvest skipped: consolidator has no storage")
            return

        try:
            from ..harvest.harvester import SessionHarvester
            from ..harvest.models import harvest_config_from_env
            from ..services.memory_service import MemoryService
        except Exception as e:
            self.logger.warning(
                "Scheduled harvest skipped: harvest module unavailable (%s)",
                _sanitize_log_value(e),
            )
            return

        session_dir = os.path.expanduser(os.getenv("MCP_HARVEST_SESSION_DIR", "~/.kiro/sessions/cli"))
        job_start = datetime.now()
        self.logger.info("Starting scheduled session harvest from %s", session_dir)
        try:
            from ..harvest.models import HarvestConfig
            page_size = int(os.getenv("MCP_HARVEST_SCHEDULE_SESSIONS", "50"))
            use_llm = os.getenv("MCP_HARVEST_SCHEDULE_USE_LLM", "true").lower() in ("true", "1", "yes")
            # harvest_and_store stores via MemoryService.store_memory — pass the
            # service wrapper, not the raw storage backend.
            memory_service = MemoryService(storage)
            harvester = SessionHarvester(project_dir=session_dir, memory_service=memory_service)

            # Idempotency: read the harvest-tracker and skip already-harvested
            # sessions, mirroring the memory_harvest handler so scheduled runs
            # don't re-process (and duplicate) sessions every cycle.
            already = await self._read_harvest_tracker(memory_service)
            all_config = HarvestConfig(sessions=9999, project_path=session_dir)
            all_sessions = harvester._resolve_sessions(all_config)
            pending = [s for s in all_sessions if harvester._session_id(s) not in already]
            if not pending:
                self.logger.info("Scheduled harvest: all %d sessions already harvested", len(all_sessions))
                return

            config = harvest_config_from_env(
                sessions=page_size,
                dry_run=False,
                use_llm=use_llm,
                project_path=session_dir,
                session_ids=[harvester._session_id(s) for s in pending[:page_size]],
            )
            results = await harvester.harvest_and_store(config)
            stored = sum(getattr(r, "stored", 0) or 0 for r in results)
            found = sum(getattr(r, "found", 0) or 0 for r in results)

            # Update tracker only with sessions that actually stored something
            # (RFC-provenance R7): a session harvested with stored==0 stays
            # pending so a later run re-harvests it instead of skipping forever.
            new_ids = sessions_to_track(results)
            if new_ids:
                await self._update_harvest_tracker(memory_service, already | new_ids)

            self.execution_stats['successful_jobs'] += 1
            self.last_execution_times['harvest'] = job_start
            duration = (datetime.now() - job_start).total_seconds()
            self.logger.info(
                "Completed scheduled harvest in %.2fs: %d sessions, %d found, %d stored (%d pending remain)",
                duration, len(results), found, stored, max(0, len(pending) - page_size),
            )
        except Exception as e:
            # Never re-raise: a failing harvest must not tear down the scheduler
            # or the consolidation jobs sharing it.
            self.execution_stats['failed_jobs'] += 1
            self.logger.error("Scheduled session harvest failed: %s", e)

    async def _read_harvest_tracker(self, memory_service) -> set:
        """Read the set of already-harvested session ids from the tracker memory."""
        try:
            tracker = await memory_service.list_memories(page=1, page_size=1, tags=["harvest-tracker"])
            for mem in tracker.get("memories", []):
                content = mem.get("content", "")
                if content.startswith("harvested_sessions:"):
                    ids_str = content.split(":", 1)[1]
                    return {s for s in ids_str.split(",") if s}
        except Exception:
            pass  # first run or tracker missing — treat as empty
        return set()

    async def _update_harvest_tracker(self, memory_service, all_ids: set):
        """Upsert the harvest-tracker memory (delete old + store merged set)."""
        try:
            old = await memory_service.list_memories(page=1, page_size=1, tags=["harvest-tracker"])
            for mem in old.get("memories", []):
                await memory_service.storage.delete(mem["content_hash"])
            await memory_service.store_memory(
                content=f"harvested_sessions:{','.join(sorted(all_ids))}",
                tags=["harvest-tracker"],
                memory_type="observation",
                metadata={"count": len(all_ids)},
            )
        except Exception as e:
            self.logger.warning("Failed to update harvest tracker: %s", e)

    def _create_trigger(self, horizon: str, schedule_spec: str):
        """Create APScheduler trigger from schedule specification."""
        try:
            if horizon == 'daily':
                # Daily format: "HH:MM" (e.g., "02:00")
                hour, minute = map(int, schedule_spec.split(':'))
                return CronTrigger(hour=hour, minute=minute)
            
            elif horizon == 'weekly':
                # Weekly format: "DAY HH:MM" (e.g., "SUN 03:00")
                day_time = schedule_spec.split(' ')
                if len(day_time) != 2:
                    raise ValueError(f"Invalid weekly schedule format: {schedule_spec}")
                
                day_map = {
                    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3, 
                    'FRI': 4, 'SAT': 5, 'SUN': 6
                }
                
                day = day_map.get(day_time[0].upper())
                if day is None:
                    raise ValueError(f"Invalid day: {day_time[0]}")
                
                hour, minute = map(int, day_time[1].split(':'))
                return CronTrigger(day_of_week=day, hour=hour, minute=minute)
            
            elif horizon == 'monthly':
                # Monthly format: "DD HH:MM" (e.g., "01 04:00")
                day_time = schedule_spec.split(' ')
                if len(day_time) != 2:
                    raise ValueError(f"Invalid monthly schedule format: {schedule_spec}")
                
                day = int(day_time[0])
                hour, minute = map(int, day_time[1].split(':'))
                return CronTrigger(day=day, hour=hour, minute=minute)
            
            elif horizon == 'quarterly':
                # Quarterly format: "MM-DD HH:MM" (e.g., "01-01 05:00")
                # Run on the first day of quarters (Jan, Apr, Jul, Oct)
                parts = schedule_spec.split(' ')
                if len(parts) != 2:
                    raise ValueError(f"Invalid quarterly schedule format: {schedule_spec}")
                
                month_day = parts[0].split('-')
                if len(month_day) != 2:
                    raise ValueError(f"Invalid quarterly date format: {parts[0]}")
                
                day = int(month_day[1])
                hour, minute = map(int, parts[1].split(':'))
                
                # Quarters: Jan(1), Apr(4), Jul(7), Oct(10)
                return CronTrigger(month='1,4,7,10', day=day, hour=hour, minute=minute)
            
            elif horizon == 'yearly':
                # Yearly format: "MM-DD HH:MM" (e.g., "01-01 06:00")
                parts = schedule_spec.split(' ')
                if len(parts) != 2:
                    raise ValueError(f"Invalid yearly schedule format: {schedule_spec}")
                
                month_day = parts[0].split('-')
                if len(month_day) != 2:
                    raise ValueError(f"Invalid yearly date format: {parts[0]}")
                
                month = int(month_day[0])
                day = int(month_day[1])
                hour, minute = map(int, parts[1].split(':'))
                
                return CronTrigger(month=month, day=day, hour=hour, minute=minute)
            
            else:
                self.logger.error("Unknown time horizon: %s", _sanitize_log_value(horizon))
                return None
                
        except Exception as e:
            self.logger.error("Error creating trigger for %s with spec '%s': %s", _sanitize_log_value(horizon), _sanitize_log_value(schedule_spec), _sanitize_log_value(e))
            return None
    
    async def _run_consolidation_job(self, time_horizon: str):
        """Execute a consolidation job for the specified time horizon."""
        job_start_time = datetime.now()
        self.logger.info("Starting scheduled %s consolidation", _sanitize_log_value(time_horizon))
        
        try:
            # Run the consolidation
            report = await self.consolidator.consolidate(time_horizon)
            
            # Run belief derivation (opt-in via MCP_BELIEFS_ENABLED)
            belief_stats = {}
            beliefs_enabled = os.getenv("MCP_BELIEFS_ENABLED", "false").lower() in ("true", "1", "yes")
            if beliefs_enabled and hasattr(self.consolidator, 'storage'):
                try:
                    belief_svc = BeliefService(self.consolidator.storage)
                    belief_stats = await belief_svc.derive_beliefs()
                except Exception as be:
                    self.logger.warning("Belief derivation error (non-fatal): %s", _sanitize_log_value(be))

            
            # Record successful execution
            self.execution_stats['successful_jobs'] += 1
            self.last_execution_times[time_horizon] = job_start_time
            
            # Add to job history
            job_record = {
                'time_horizon': time_horizon,
                'start_time': job_start_time,
                'end_time': datetime.now(),
                'status': 'success',
                'memories_processed': report.memories_processed,
                'associations_discovered': report.associations_discovered,
                'clusters_created': report.clusters_created,
                'memories_compressed': report.memories_compressed,
                'memories_archived': report.memories_archived,
                'beliefs': belief_stats,
                'errors': report.errors
            }
            
            self._add_job_to_history(job_record)
            
            # Log success
            duration = (job_record['end_time'] - job_record['start_time']).total_seconds()
            self.logger.info(
                f"Completed {time_horizon} consolidation successfully in {duration:.2f}s: "
                f"{report.memories_processed} memories processed, "
                f"{report.associations_discovered} associations, "
                f"{report.clusters_created} clusters, "
                f"{report.memories_compressed} compressed, "
                f"{report.memories_archived} archived"
            )
            
        except Exception as e:
            # Record failed execution
            self.execution_stats['failed_jobs'] += 1
            
            job_record = {
                'time_horizon': time_horizon,
                'start_time': job_start_time,
                'end_time': datetime.now(),
                'status': 'failed',
                'error': str(e),
                'memories_processed': 0,
                'associations_discovered': 0,
                'clusters_created': 0,
                'memories_compressed': 0,
                'memories_archived': 0,
                'errors': [str(e)]
            }
            
            self._add_job_to_history(job_record)
            
            self.logger.error("Failed %s consolidation: %s", _sanitize_log_value(time_horizon), _sanitize_log_value(e))
            raise
        finally:
            # Consolidation can change counts even if a later step fails.
            try:
                from ..web.sse import SSEEvent, sse_manager

                await sse_manager.broadcast_event(SSEEvent(
                    event_type="consolidation_completed",
                    data={"time_horizon": time_horizon, "success": job_record["status"] == "success"},
                ))
            except Exception as exc:
                self.logger.debug("Unable to notify dashboard of consolidation: %s", _sanitize_log_value(exc))

    def _add_job_to_history(self, job_record: Dict[str, Any]):
        """Add job record to history with size limit."""
        self.job_history.append(job_record)
        
        # Keep only last 100 job records
        if len(self.job_history) > 100:
            self.job_history = self.job_history[-100:]
    
    def _job_executed_listener(self, event):
        """Handle job execution events."""
        self.execution_stats['total_jobs'] += 1
        self.logger.debug("Job executed: %s", _sanitize_log_value(event.job_id))
    
    def _job_error_listener(self, event):
        """Handle job error events."""
        self.logger.error("Job error: %s - %s", _sanitize_log_value(event.job_id), _sanitize_log_value(event.exception))
    
    async def trigger_consolidation(self, time_horizon: str, immediate: bool = True) -> bool:
        """Manually trigger a consolidation job."""
        if not self.scheduler:
            self.logger.error("Scheduler not available")
            return False
        
        try:
            if immediate:
                # Run immediately
                await self._run_consolidation_job(time_horizon)
                return True
            else:
                # Schedule to run in 1 minute
                job_id = f"manual_consolidation_{time_horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                trigger = IntervalTrigger(seconds=60)  # Run once after 60 seconds
                
                self.scheduler.add_job(
                    func=self._run_consolidation_job,
                    trigger=trigger,
                    args=[time_horizon],
                    id=job_id,
                    name=f"Manual Consolidation - {time_horizon.title()}",
                    max_instances=1
                )
                
                self.logger.info("Scheduled manual %s consolidation", _sanitize_log_value(time_horizon))
                return True
                
        except Exception as e:
            self.logger.error("Error triggering %s consolidation: %s", _sanitize_log_value(time_horizon), _sanitize_log_value(e))
            return False
    
    async def get_scheduler_status(self) -> Dict[str, Any]:
        """Get scheduler status and job information."""
        if not self.scheduler:
            return {
                'enabled': False,
                'reason': 'APScheduler not available or disabled'
            }
        
        jobs = self.scheduler.get_jobs()
        job_info = []
        
        for job in jobs:
            job_info.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'enabled': True,
            'running': self.scheduler.running,
            'jobs': job_info,
            'execution_stats': self.execution_stats.copy(),
            'last_execution_times': {
                horizon: time.isoformat() for horizon, time in self.last_execution_times.items()
            },
            'recent_jobs': self.job_history[-10:]  # Last 10 jobs
        }
    
    async def update_schedule(self, new_schedule_config: Dict[str, str]) -> bool:
        """Update the consolidation schedule."""
        if not self.scheduler:
            return False
        
        try:
            # Remove existing consolidation jobs
            job_ids = [f"consolidation_{horizon}" for horizon in ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']]
            
            for job_id in job_ids:
                if self.scheduler.get_job(job_id):
                    self.scheduler.remove_job(job_id)
            
            # Update configuration
            self.schedule_config = new_schedule_config
            
            # Re-schedule jobs
            self._schedule_consolidation_jobs()
            
            self.logger.info("Consolidation schedule updated successfully")
            return True
            
        except Exception as e:
            self.logger.error("Error updating consolidation schedule: %s", _sanitize_log_value(e))
            return False
    
    async def pause_consolidation(self, time_horizon: Optional[str] = None) -> bool:
        """Pause consolidation jobs (all or specific horizon)."""
        if not self.scheduler:
            return False
        
        try:
            if time_horizon:
                job_id = f"consolidation_{time_horizon}"
                job = self.scheduler.get_job(job_id)
                if job:
                    self.scheduler.pause_job(job_id)
                    self.logger.info("Paused %s consolidation", _sanitize_log_value(time_horizon))
                else:
                    self.logger.warning("No job found for %s consolidation", _sanitize_log_value(time_horizon))
            else:
                # Pause all consolidation jobs
                jobs = self.scheduler.get_jobs()
                for job in jobs:
                    if job.id.startswith('consolidation_'):
                        self.scheduler.pause_job(job.id)
                
                self.logger.info("Paused all consolidation jobs")
            
            return True
            
        except Exception as e:
            self.logger.error("Error pausing consolidation: %s", _sanitize_log_value(e))
            return False
    
    async def resume_consolidation(self, time_horizon: Optional[str] = None) -> bool:
        """Resume consolidation jobs (all or specific horizon)."""
        if not self.scheduler:
            return False
        
        try:
            if time_horizon:
                job_id = f"consolidation_{time_horizon}"
                job = self.scheduler.get_job(job_id)
                if job:
                    self.scheduler.resume_job(job_id)
                    self.logger.info("Resumed %s consolidation", _sanitize_log_value(time_horizon))
                else:
                    self.logger.warning("No job found for %s consolidation", _sanitize_log_value(time_horizon))
            else:
                # Resume all consolidation jobs
                jobs = self.scheduler.get_jobs()
                for job in jobs:
                    if job.id.startswith('consolidation_'):
                        self.scheduler.resume_job(job.id)
                
                self.logger.info("Resumed all consolidation jobs")
            
            return True
            
        except Exception as e:
            self.logger.error("Error resuming consolidation: %s", _sanitize_log_value(e))
            return False
