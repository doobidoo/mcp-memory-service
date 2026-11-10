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
FastAPI application for MCP Memory Service HTTP/SSE interface.

Provides REST API and Server-Sent Events using SQLite-vec backend.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from html import escape
from typing import Optional, Any

from ..compat import _sanitize_log_value

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.types import Receive, Scope, Send

try:
    from .. import __version__
except (ImportError, AttributeError):
    __version__ = "0.0.0.dev0"
from ..config import (
    HTTP_PORT,
    HTTP_HOST,
    HTTP_ROOT_PATH,
    CORS_ORIGINS,
    MDNS_ENABLED,
    HTTPS_ENABLED,
    OAUTH_ENABLED,
    METRICS_ENABLED,
    CONSOLIDATION_ENABLED,
    CONSOLIDATION_CONFIG,
    CONSOLIDATION_SCHEDULE,
    BACKUP_ENABLED
)
from .dependencies import set_storage, create_storage_backend
from .api.health import router as health_router
from .api.memories import router as memories_router
from .api.search import router as search_router
from .api.events import router as events_router
from .api.sync import router as sync_router
from .api.manage import router as manage_router
from .api.analytics import router as analytics_router
from .api.documents import router as documents_router
from .api.mcp import router as mcp_router
from .api.consolidation import router as consolidation_router
from .api.backup import router as backup_router
from .api.quality import router as quality_router
from .api.server import router as server_router
from .api.configuration import router as configuration_router
from .api.oauth_status import router as oauth_status_router
from .api.conflicts import router as conflicts_router
from .api.harvest import router as harvest_router
from .api.sync_events import router as sync_events_router
from .sse import sse_manager

logger = logging.getLogger(__name__)

_DASHBOARD_CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net https://esm.sh "
    "'sha256-+9dJjjHd5V3OuLS4ZtFtbrWm8oLQHO5xVJLfAzAGoAg='; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self'; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
)
_SAME_ORIGIN_DASHBOARD_CSP = (
    "default-src 'self'; script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self'; font-src 'self'; "
    "connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
)


class RootPathStaticFiles(StaticFiles):
    """Serve a mounted directory when the proxy strips the external root path."""

    def __init__(self, *, directory: str, external_root_path: str) -> None:
        super().__init__(directory=directory)
        self.external_root_path = external_root_path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        root_path = scope.get("root_path", "")
        if (self.external_root_path and root_path.startswith(self.external_root_path)
                and not scope["path"].startswith(self.external_root_path + "/")):
            scope = dict(scope)
            scope["root_path"] = root_path[len(self.external_root_path):]
        await super().__call__(scope, receive, send)

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            if path == "index.html":
                response.headers["Content-Security-Policy"] = _DASHBOARD_CSP
            elif path == "sse_test.html":
                response.headers["Content-Security-Policy"] = _SAME_ORIGIN_DASHBOARD_CSP
        response.headers["Cache-Control"] = "no-cache"
        return response

# Global storage instance
storage: Optional["MemoryStorage"] = None

# Global mDNS advertiser instance
mdns_advertiser: Optional[Any] = None

# Global OAuth cleanup task
oauth_cleanup_task: Optional[asyncio.Task] = None

# Global consolidation instances
consolidation_scheduler: Optional["ConsolidationScheduler"] = None

# Global backup scheduler instance
backup_scheduler: Optional["BackupScheduler"] = None

# Env vars that each enable an opt-in interval job on the consolidation scheduler.
# When any is set (not blank / not "disabled"), the scheduler is started even if the
# consolidation cadences themselves are all disabled — otherwise a host that only
# enables one of these would never get a scheduler and the job would silently never run.
# One entry per line so independent features can append their own without editing a
# shared literal (keeps feature PRs orthogonal).
_OPTIN_SCHEDULE_ENV_VARS = (
    "MCP_HARVEST_SCHEDULE",
    "MCP_SYNC_SCHEDULE",  # delta-sync Phase 4d
)


async def oauth_cleanup_background_task():
    """Background task to periodically clean up expired OAuth tokens and codes."""
    from .oauth.storage import get_oauth_storage

    while True:
        try:
            # Clean up expired tokens every 5 minutes
            await asyncio.sleep(300)  # 5 minutes

            storage = get_oauth_storage()
            cleanup_stats = await storage.cleanup_expired()
            if cleanup_stats["expired_codes_cleaned"] > 0 or cleanup_stats["expired_tokens_cleaned"] > 0:
                logger.info("OAuth cleanup: removed %s codes, %s tokens",
                            _sanitize_log_value(cleanup_stats['expired_codes_cleaned']),
                            _sanitize_log_value(cleanup_stats['expired_tokens_cleaned']))

        except asyncio.CancelledError:
            logger.info("OAuth cleanup task cancelled")
            break
        except Exception as e:
            logger.error("Error in OAuth cleanup task: %s", _sanitize_log_value(e))
            # Continue running even if there's an error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global storage, mdns_advertiser, oauth_cleanup_task, consolidation_scheduler, backup_scheduler, document_cleanup_task

    # Startup
    logger.info("Starting MCP Memory Service HTTP interface...")
    try:
        storage = await create_storage_backend()
        set_storage(storage)  # Set the global storage instance

        # Initialize consolidation system if enabled — OR if any opt-in interval job
        # (see _OPTIN_SCHEDULE_ENV_VARS) is configured. The scheduler needs a consolidator
        # instance to attach jobs to, so a host that only enables e.g. MCP_SYNC_SCHEDULE
        # (with consolidation disabled) must still reach this block, build the consolidator,
        # and start the scheduler — otherwise the opt-in job silently never runs (Greptile P1).
        _optin_jobs_on = any(
            (os.getenv(v, '').strip().lower() not in ('', 'disabled'))
            for v in _OPTIN_SCHEDULE_ENV_VARS
        )
        if CONSOLIDATION_ENABLED or _optin_jobs_on:
            try:
                from ..consolidation.base import ConsolidationConfig
                from ..consolidation.consolidator import DreamInspiredConsolidator
                from ..consolidation.scheduler import ConsolidationScheduler
                from ..api import set_consolidator, set_scheduler

                # Create consolidation config
                config = ConsolidationConfig(**CONSOLIDATION_CONFIG)

                # Initialize consolidator with storage
                consolidator = DreamInspiredConsolidator(storage, config)
                logger.info("Dream-inspired consolidator initialized")

                # Set global consolidator for API access
                set_consolidator(consolidator)

                # Start the scheduler when consolidation cadences OR any opt-in job is enabled.
                #
                # If we only got here because an opt-in job (sync/harvest) is on while
                # consolidation is DISABLED, the scheduler must NOT register consolidation
                # jobs — passing the raw CONSOLIDATION_SCHEDULE would re-activate cadences
                # (e.g. a leftover MCP_SCHEDULE_WEEKLY) that can archive/delete memories the
                # user turned off (Greptile P1). Neutralize every cadence to 'disabled' in
                # that case; the opt-in jobs schedule themselves from their own env vars.
                if CONSOLIDATION_ENABLED:
                    effective_schedule = CONSOLIDATION_SCHEDULE
                else:
                    effective_schedule = {k: 'disabled' for k in CONSOLIDATION_SCHEDULE}

                _consolidation_on = any(schedule != 'disabled' for schedule in effective_schedule.values())
                if _consolidation_on or _optin_jobs_on:
                    consolidation_scheduler = ConsolidationScheduler(
                        consolidator,
                        effective_schedule,
                        enabled=True
                    )

                    # Start the scheduler
                    if await consolidation_scheduler.start():
                        logger.info("Consolidation scheduler started successfully")
                        # Set global scheduler for API access
                        set_scheduler(consolidation_scheduler)
                    else:
                        logger.warning("Failed to start consolidation scheduler")
                        consolidation_scheduler = None
                else:
                    logger.info("Consolidation scheduler disabled (all schedules set to 'disabled')")

            except Exception as e:
                logger.error("Failed to initialize consolidation system: %s", _sanitize_log_value(e))
                consolidation_scheduler = None
        else:
            logger.info("Consolidation system disabled")

        # Start SSE manager
        await sse_manager.start()
        logger.info("SSE Manager started")

        # Start OAuth cleanup task if enabled
        if OAUTH_ENABLED:
            oauth_cleanup_task = asyncio.create_task(oauth_cleanup_background_task())
            logger.info("OAuth cleanup background task started")
        
        # Start mDNS service advertisement if enabled
        if MDNS_ENABLED:
            try:
                from ..discovery.mdns_service import ServiceAdvertiser
                mdns_advertiser = ServiceAdvertiser(
                    host=HTTP_HOST,
                    port=HTTP_PORT,
                    https_enabled=HTTPS_ENABLED
                )
                success = await mdns_advertiser.start()
                if success:
                    logger.info("mDNS service advertisement started")
                else:
                    logger.warning("Failed to start mDNS service advertisement")
                    mdns_advertiser = None
            except ImportError:
                logger.warning("mDNS support not available (zeroconf not installed)")
                mdns_advertiser = None
            except Exception as e:
                logger.error("Error starting mDNS advertisement: %s", _sanitize_log_value(e))
                mdns_advertiser = None
        else:
            logger.info("mDNS service advertisement disabled")

        # Start backup scheduler if enabled
        if BACKUP_ENABLED:
            try:
                from ..backup.scheduler import get_backup_scheduler
                backup_scheduler = get_backup_scheduler()
                await backup_scheduler.start()
                logger.info("Backup scheduler started")
            except ImportError:
                logger.warning("Backup scheduler not available (backup module not installed)")
                backup_scheduler = None
            except Exception as e:
                logger.error("Error starting backup scheduler: %s", _sanitize_log_value(e))
                backup_scheduler = None
        else:
            logger.info("Backup scheduler disabled")

        # Start document upload session cleanup task
        try:
            from .api.documents import cleanup_sessions_loop
            document_cleanup_task = asyncio.create_task(cleanup_sessions_loop())
            logger.info("Document session cleanup task started")
        except Exception as e:
            logger.error("Error starting document session cleanup: %s", _sanitize_log_value(e))
            document_cleanup_task = None

    except Exception as e:
        logger.error("Failed to initialize storage: %s", _sanitize_log_value(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Memory Service HTTP interface...")

    # Stop consolidation scheduler
    if consolidation_scheduler:
        try:
            await consolidation_scheduler.stop()
            logger.info("Consolidation scheduler stopped")
        except Exception as e:
            logger.error("Error stopping consolidation scheduler: %s", _sanitize_log_value(e))

    # Stop backup scheduler
    if backup_scheduler:
        try:
            await backup_scheduler.stop()
            logger.info("Backup scheduler stopped")
        except Exception as e:
            logger.error("Error stopping backup scheduler: %s", _sanitize_log_value(e))

    # Stop mDNS advertisement
    if mdns_advertiser:
        try:
            await mdns_advertiser.stop()
            logger.info("mDNS service advertisement stopped")
        except Exception as e:
            logger.error("Error stopping mDNS advertisement: %s", _sanitize_log_value(e))

    # Stop OAuth cleanup task
    if oauth_cleanup_task:
        try:
            oauth_cleanup_task.cancel()
            await oauth_cleanup_task
            logger.info("OAuth cleanup task stopped")
        except asyncio.CancelledError:
            logger.info("OAuth cleanup task cancelled successfully")
        except Exception as e:
            logger.error("Error stopping OAuth cleanup task: %s", _sanitize_log_value(e))

    # Stop document session cleanup task
    if document_cleanup_task:
        try:
            document_cleanup_task.cancel()
            await document_cleanup_task
            logger.info("Document session cleanup task stopped")
        except asyncio.CancelledError:
            logger.info("Document session cleanup task cancelled successfully")
        except Exception as e:
            logger.error("Error stopping document session cleanup task: %s", _sanitize_log_value(e))

    # Stop SSE manager
    await sse_manager.stop()
    logger.info("SSE Manager stopped")

    if storage:
        await storage.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="MCP Memory Service",
        description="HTTP REST API and SSE interface for semantic memory storage",
        version=__version__,
        lifespan=lifespan,
        root_path=HTTP_ROOT_PATH,
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # CORS middleware
    if '*' in CORS_ORIGINS:
        logger.warning(
            "Wildcard CORS origin ('*') detected. Any website can read API responses. "
            "Set MCP_CORS_ORIGINS to restrict access (e.g. 'http://localhost:8000')."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials='*' not in CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request body-size cap (disk/memory exhaustion + buffer-overflow guard).
    # Global cap for all routes, tighter cap for /oauth/*, document ingestion
    # exempt. Added last so it runs first (outermost) on the inbound path.
    from .body_limit import BodySizeLimitMiddleware
    app.add_middleware(BodySizeLimitMiddleware)

    # Include API routers
    logger.info("Including API routers...")
    app.include_router(health_router, prefix="/api", tags=["health"])
    logger.info("✓ Included health router with %s routes", _sanitize_log_value(len(health_router.routes)))
    app.include_router(memories_router, prefix="/api", tags=["memories"])
    logger.info("✓ Included memories router with %s routes", _sanitize_log_value(len(memories_router.routes)))
    app.include_router(search_router, prefix="/api", tags=["search"])
    logger.info("✓ Included search router with %s routes", _sanitize_log_value(len(search_router.routes)))
    app.include_router(manage_router, prefix="/api/manage", tags=["management"])
    logger.info("✓ Included manage router with %s routes", _sanitize_log_value(len(manage_router.routes)))
    app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])
    logger.info("✓ Included analytics router with %s routes", _sanitize_log_value(len(analytics_router.routes)))
    app.include_router(events_router, prefix="/api", tags=["events"])
    logger.info("✓ Included events router with %s routes", _sanitize_log_value(len(events_router.routes)))
    app.include_router(sync_router, prefix="/api", tags=["sync"])
    logger.info("✓ Included sync router with %s routes", _sanitize_log_value(len(sync_router.routes)))
    app.include_router(backup_router, prefix="/api", tags=["backup"])
    logger.info("✓ Included backup router with %s routes", _sanitize_log_value(len(backup_router.routes)))
    app.include_router(quality_router, prefix="/api/quality", tags=["quality"])
    logger.info("✓ Included quality router with %s routes", _sanitize_log_value(len(quality_router.routes)))
    try:
        app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
        logger.info("✓ Included documents router with %s routes", _sanitize_log_value(len(documents_router.routes)))
    except Exception as e:
        logger.error("✗ Failed to include documents router: %s", _sanitize_log_value(e))
        import traceback  # inline import: only reached on this router-import failure
        logger.error(traceback.format_exc())

    # Include consolidation router
    app.include_router(consolidation_router, tags=["consolidation"])
    logger.info("✓ Included consolidation router with %s routes", _sanitize_log_value(len(consolidation_router.routes)))

    # Include server management router
    app.include_router(server_router, prefix="/api/server", tags=["server-management"])
    logger.info("✓ Included server router with %s routes", _sanitize_log_value(len(server_router.routes)))

    # Include configuration router
    app.include_router(configuration_router, prefix="/api", tags=["configuration"])
    logger.info("✓ Included configuration router with %s routes", _sanitize_log_value(len(configuration_router.routes)))

    # Include OAuth status router (always available, returns disabled status if OAuth off)
    app.include_router(oauth_status_router, prefix="/api", tags=["oauth-status"])
    logger.info("✓ Included OAuth status router with %s routes", _sanitize_log_value(len(oauth_status_router.routes)))

    # Include conflicts router (P3 conflict detection)
    app.include_router(conflicts_router, prefix="/api", tags=["conflicts"])
    logger.info("✓ Included conflicts router with %s routes", _sanitize_log_value(len(conflicts_router.routes)))

    # Include session harvest router (Issue #630)
    app.include_router(harvest_router, tags=["harvest"])
    logger.info("✓ Included harvest router with %s routes", _sanitize_log_value(len(harvest_router.routes)))

    # Include delta-sync event feed router (#1345 Phase 4 — pull/push transport)
    app.include_router(sync_events_router, prefix="/api", tags=["sync"])
    logger.info("✓ Included sync_events router with %s routes", _sanitize_log_value(len(sync_events_router.routes)))

    # Include MCP protocol router
    app.include_router(mcp_router, tags=["mcp-protocol"])

    # Include OAuth routers if enabled
    if OAUTH_ENABLED:
        from .oauth.discovery import router as oauth_discovery_router
        from .oauth.registration import router as oauth_registration_router
        from .oauth.authorization import router as oauth_authorization_router

        app.include_router(oauth_discovery_router, tags=["oauth-discovery"])
        app.include_router(oauth_registration_router, prefix="/oauth", tags=["oauth"])
        app.include_router(oauth_authorization_router, prefix="/oauth", tags=["oauth"])

        logger.info("OAuth 2.1 endpoints enabled")
    else:
        logger.info("OAuth 2.1 endpoints disabled")

    # Include Prometheus metrics router if enabled (issue #1097, opt-in).
    # Read the module-level flag (not the import-time local) so tests that
    # monkeypatch ``web.app.METRICS_ENABLED`` and rebuild the app are honoured,
    # exactly the way OAUTH_ENABLED gates its routers above.
    if METRICS_ENABLED:
        from .api.metrics import router as metrics_router
        app.include_router(metrics_router, tags=["metrics"])
        logger.info("Prometheus /metrics endpoint enabled")
    else:
        logger.info("Prometheus /metrics endpoint disabled")

    # Serve static files (dashboard)
    static_path = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_path):
        app.mount(
            "/static",
            RootPathStaticFiles(
                directory=static_path,
                external_root_path=HTTP_ROOT_PATH,
            ),
            name="static",
        )
    
    @app.get("/api-overview", include_in_schema=False)
    async def api_overview():
        """Redirect legacy API overview bookmarks to Swagger UI."""
        # Relative URL preserves reverse-proxy path prefixes.
        return RedirectResponse(url="api/docs", status_code=308)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Serve the dashboard homepage."""
        # Serve the migrated interactive dashboard instead of hardcoded template
        try:
            # Path to the migrated dashboard HTML file
            dashboard_path = os.path.join(os.path.dirname(__file__), "static", "index.html")

            if os.path.exists(dashboard_path):
                # Read and serve the migrated dashboard
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    dashboard_html = f.read()
                base_href = escape(f"{HTTP_ROOT_PATH}/" if HTTP_ROOT_PATH else "/", quote=True)
                dashboard_html = dashboard_html.replace(
                    '<base href="../">', f'<base href="{base_href}">', 1
                )
                return HTMLResponse(
                    content=dashboard_html,
                    headers={"Content-Security-Policy": _DASHBOARD_CSP, "Cache-Control": "no-cache"},
                )
            logger.warning("Dashboard file not found: %s", dashboard_path)
        except Exception as e:
            logger.warning("Error loading migrated dashboard: %s", _sanitize_log_value(e))
        # The API keeps working without the static files; say so instead of a 500 (#1926).
        return HTMLResponse(
            content="<!doctype html><title>Dashboard unavailable</title>"
            "<p>Dashboard files not found. The REST API is still available under /api.</p>",
            status_code=503,
        )

    @app.get("/api/languages")
    async def get_available_languages():
        """Get list of available translation languages by scanning i18n directory."""
        try:
            i18n_path = os.path.join(os.path.dirname(__file__), "static", "i18n")

            if not os.path.exists(i18n_path):
                return {"languages": ["en"]}  # Fallback to English only

            # Scan for .json files in i18n directory
            languages = []
            for filename in os.listdir(i18n_path):
                if filename.endswith('.json'):
                    # Extract language code (e.g., "en.json" -> "en")
                    lang_code = filename[:-5]  # Remove .json extension
                    languages.append(lang_code)

            # Sort languages alphabetically
            languages.sort()

            # Ensure at least English is available
            if not languages:
                languages = ["en"]

            return {"languages": languages}

        except Exception as e:
            logger.error("Error scanning i18n directory: %s", _sanitize_log_value(e))
            return {"languages": ["en"]}  # Fallback to English on error

    return app


# Create the app instance
app = create_app()


# Storage getter is now in dependencies.py
