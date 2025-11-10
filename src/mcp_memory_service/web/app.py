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
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .. import __version__
from ..config import (
    HTTP_PORT,
    HTTP_HOST,
    CORS_ORIGINS,
    DATABASE_PATH,
    EMBEDDING_MODEL_NAME,
    MDNS_ENABLED,
    HTTPS_ENABLED,
    OAUTH_ENABLED,
    CONSOLIDATION_ENABLED,
    CONSOLIDATION_CONFIG,
    CONSOLIDATION_SCHEDULE
)
from .dependencies import set_storage, get_storage, create_storage_backend
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
from .sse import sse_manager

logger = logging.getLogger(__name__)

# Global storage instance
storage: Optional["MemoryStorage"] = None

# Global mDNS advertiser instance
mdns_advertiser: Optional[Any] = None

# Global OAuth cleanup task
oauth_cleanup_task: Optional[asyncio.Task] = None

# Global consolidation instances
consolidator: Optional["DreamInspiredConsolidator"] = None
consolidation_scheduler: Optional["ConsolidationScheduler"] = None


async def oauth_cleanup_background_task():
    """Background task to periodically clean up expired OAuth tokens and codes."""
    from .oauth.storage import oauth_storage

    while True:
        try:
            # Clean up expired tokens every 5 minutes
            await asyncio.sleep(300)  # 5 minutes

            cleanup_stats = await oauth_storage.cleanup_expired()
            if cleanup_stats["expired_codes_cleaned"] > 0 or cleanup_stats["expired_tokens_cleaned"] > 0:
                logger.info(f"OAuth cleanup: removed {cleanup_stats['expired_codes_cleaned']} codes, "
                           f"{cleanup_stats['expired_tokens_cleaned']} tokens")

        except asyncio.CancelledError:
            logger.info("OAuth cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in OAuth cleanup task: {e}")
            # Continue running even if there's an error


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global storage, mdns_advertiser, oauth_cleanup_task, consolidator, consolidation_scheduler

    # Startup
    logger.info("Starting MCP Memory Service HTTP interface...")
    try:
        storage = await create_storage_backend()
        set_storage(storage)  # Set the global storage instance

        # Initialize consolidation system if enabled
        if CONSOLIDATION_ENABLED:
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

                # Initialize scheduler if any schedules are enabled
                if any(schedule != 'disabled' for schedule in CONSOLIDATION_SCHEDULE.values()):
                    consolidation_scheduler = ConsolidationScheduler(
                        consolidator,
                        CONSOLIDATION_SCHEDULE,
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
                logger.error(f"Failed to initialize consolidation system: {e}")
                consolidator = None
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
                logger.error(f"Error starting mDNS advertisement: {e}")
                mdns_advertiser = None
        else:
            logger.info("mDNS service advertisement disabled")
            
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
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
            logger.error(f"Error stopping consolidation scheduler: {e}")

    # Stop mDNS advertisement
    if mdns_advertiser:
        try:
            await mdns_advertiser.stop()
            logger.info("mDNS service advertisement stopped")
        except Exception as e:
            logger.error(f"Error stopping mDNS advertisement: {e}")

    # Stop OAuth cleanup task
    if oauth_cleanup_task:
        try:
            oauth_cleanup_task.cancel()
            await oauth_cleanup_task
            logger.info("OAuth cleanup task stopped")
        except asyncio.CancelledError:
            logger.info("OAuth cleanup task cancelled successfully")
        except Exception as e:
            logger.error(f"Error stopping OAuth cleanup task: {e}")

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
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    logger.info("Including API routers...")
    app.include_router(health_router, prefix="/api", tags=["health"])
    logger.info(f"✓ Included health router with {len(health_router.routes)} routes")
    app.include_router(memories_router, prefix="/api", tags=["memories"])
    logger.info(f"✓ Included memories router with {len(memories_router.routes)} routes")
    app.include_router(search_router, prefix="/api", tags=["search"])
    logger.info(f"✓ Included search router with {len(search_router.routes)} routes")
    app.include_router(manage_router, prefix="/api/manage", tags=["management"])
    logger.info(f"✓ Included manage router with {len(manage_router.routes)} routes")
    app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])
    logger.info(f"✓ Included analytics router with {len(analytics_router.routes)} routes")
    app.include_router(events_router, prefix="/api", tags=["events"])
    logger.info(f"✓ Included events router with {len(events_router.routes)} routes")
    app.include_router(sync_router, prefix="/api", tags=["sync"])
    logger.info(f"✓ Included sync router with {len(sync_router.routes)} routes")
    try:
        app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
        logger.info(f"✓ Included documents router with {len(documents_router.routes)} routes")
    except Exception as e:
        logger.error(f"✗ Failed to include documents router: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Include consolidation router
    app.include_router(consolidation_router, tags=["consolidation"])
    logger.info(f"✓ Included consolidation router with {len(consolidation_router.routes)} routes")

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

    # Serve static files (dashboard)
    static_path = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_path):
        app.mount("/static", StaticFiles(directory=static_path), name="static")
    
    def get_api_overview_html():
        """Generate the API overview HTML template."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>MCP Memory Service v""" + __version__ + """</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                :root {
                    --primary: #3b82f6;
                    --primary-dark: #2563eb;
                    --secondary: #8b5cf6;
                    --success: #10b981;
                    --warning: #f59e0b;
                    --danger: #ef4444;
                    --dark: #1e293b;
                    --gray: #64748b;
                    --light: #f8fafc;
                    --white: #ffffff;
                    --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
                    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                    color: var(--dark);
                    line-height: 1.6;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem;
                }
                
                header {
                    text-align: center;
                    margin-bottom: 3rem;
                    padding: 2rem;
                    background: var(--white);
                    border-radius: 1rem;
                    box-shadow: var(--shadow-lg);
                }
                
                .logo {
                    display: inline-flex;
                    align-items: center;
                    gap: 1rem;
                    margin-bottom: 1rem;
                }
                
                .logo-icon {
                    width: 60px;
                    height: 60px;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    border-radius: 1rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: var(--white);
                    font-size: 2rem;
                    font-weight: bold;
                }
                
                h1 {
                    font-size: 2.5rem;
                    font-weight: 800;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    margin-bottom: 0.5rem;
                }
                
                .subtitle {
                    color: var(--gray);
                    font-size: 1.25rem;
                    margin-bottom: 1rem;
                }
                
                .version-badge {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    background: var(--success);
                    color: var(--white);
                    padding: 0.25rem 1rem;
                    border-radius: 2rem;
                    font-size: 0.875rem;
                    font-weight: 600;
                }
                
                .stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                    margin-bottom: 3rem;
                }
                
                .stat-card {
                    background: var(--white);
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    box-shadow: var(--shadow);
                    text-align: center;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }
                
                .stat-card:hover {
                    transform: translateY(-2px);
                    box-shadow: var(--shadow-lg);
                }
                
                .stat-value {
                    font-size: 2rem;
                    font-weight: 700;
                    color: var(--primary);
                    margin-bottom: 0.25rem;
                }
                
                .stat-label {
                    color: var(--gray);
                    font-size: 0.875rem;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }
                
                .endpoint-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 3rem;
                }
                
                .endpoint-card {
                    background: var(--white);
                    border-radius: 0.75rem;
                    box-shadow: var(--shadow);
                    overflow: hidden;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }
                
                .endpoint-card:hover {
                    transform: translateY(-4px);
                    box-shadow: var(--shadow-lg);
                }
                
                .endpoint-header {
                    padding: 1.5rem;
                    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
                    color: var(--white);
                }
                
                .endpoint-header h3 {
                    font-size: 1.25rem;
                    margin-bottom: 0.5rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                .endpoint-icon {
                    font-size: 1.5rem;
                }
                
                .endpoint-description {
                    opacity: 0.9;
                    font-size: 0.875rem;
                }
                
                .endpoint-list {
                    padding: 1.5rem;
                }
                
                .endpoint-item {
                    padding: 0.75rem;
                    border-radius: 0.5rem;
                    margin-bottom: 0.5rem;
                    background: var(--light);
                    transition: background-color 0.2s ease;
                    cursor: pointer;
                }
                
                .endpoint-item:hover {
                    background: #e2e8f0;
                }
                
                .method {
                    display: inline-block;
                    padding: 0.125rem 0.5rem;
                    border-radius: 0.25rem;
                    font-size: 0.75rem;
                    font-weight: 700;
                    margin-right: 0.5rem;
                    text-transform: uppercase;
                }
                
                .method-get { background: var(--success); color: var(--white); }
                .method-post { background: var(--primary); color: var(--white); }
                .method-delete { background: var(--danger); color: var(--white); }
                
                .endpoint-path {
                    font-family: 'Courier New', monospace;
                    font-size: 0.875rem;
                    color: var(--dark);
                }
                
                .endpoint-desc {
                    font-size: 0.75rem;
                    color: var(--gray);
                    margin-top: 0.25rem;
                }
                
                .action-buttons {
                    display: flex;
                    gap: 1rem;
                    justify-content: center;
                    margin-bottom: 3rem;
                }
                
                .btn {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.75rem 1.5rem;
                    border-radius: 0.5rem;
                    font-weight: 600;
                    text-decoration: none;
                    transition: all 0.2s ease;
                    border: none;
                    cursor: pointer;
                }
                
                .btn-primary {
                    background: var(--primary);
                    color: var(--white);
                }
                
                .btn-primary:hover {
                    background: var(--primary-dark);
                    transform: translateY(-2px);
                    box-shadow: var(--shadow-lg);
                }
                
                .btn-secondary {
                    background: var(--white);
                    color: var(--primary);
                    border: 2px solid var(--primary);
                }
                
                .btn-secondary:hover {
                    background: var(--primary);
                    color: var(--white);
                    transform: translateY(-2px);
                    box-shadow: var(--shadow-lg);
                }
                
                footer {
                    text-align: center;
                    padding: 2rem;
                    color: var(--gray);
                }
                
                .tech-stack {
                    display: flex;
                    justify-content: center;
                    gap: 2rem;
                    margin-top: 1rem;
                    flex-wrap: wrap;
                }
                
                .tech-badge {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.5rem 1rem;
                    background: var(--white);
                    border-radius: 0.5rem;
                    box-shadow: var(--shadow);
                    font-size: 0.875rem;
                    font-weight: 600;
                }

                .nav-buttons {
                    display: flex;
                    gap: 1rem;
                    margin-top: 1rem;
                    justify-content: center;
                }

                .nav-btn {
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.75rem 1.5rem;
                    background: var(--primary);
                    color: var(--white);
                    text-decoration: none;
                    border-radius: 0.5rem;
                    font-weight: 600;
                    transition: background-color 0.2s;
                    box-shadow: var(--shadow);
                }

                .nav-btn:hover {
                    background: var(--primary-dark);
                    text-decoration: none;
                    color: var(--white);
                }

                .nav-btn.secondary {
                    background: var(--gray);
                }

                .nav-btn.secondary:hover {
                    background: #475569;
                }

                .loading {
                    display: inline-block;
                    width: 1rem;
                    height: 1rem;
                    border: 2px solid var(--light);
                    border-top-color: var(--primary);
                    border-radius: 50%;
                    animation: spin 0.6s linear infinite;
                }
                
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
                
                @media (max-width: 768px) {
                    .container { padding: 1rem; }
                    h1 { font-size: 2rem; }
                    .endpoint-grid { grid-template-columns: 1fr; }
                    .stats { grid-template-columns: 1fr; }
                    .action-buttons {
                        flex-direction: column;
                        align-items: center;
                        gap: 0.75rem;
                    }
                    .btn {
                        width: 100%;
                        max-width: 300px;
                        justify-content: center;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div class="logo">
                        <div class="logo-icon">🧠</div>
                        <div>
                            <h1>MCP Memory Service</h1>
                            <p class="subtitle" id="subtitle">Intelligent Semantic Memory with <span id="backend-name">Loading...</span></p>
                        </div>
                    </div>
                    <div class="version-badge">
                        <span>✅</span> <span id="version-display">Loading...</span> - Latest Release
                    </div>
                    <div class="nav-buttons">
                        <a href="/" class="nav-btn">
                            <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M10,20V14H14V20H19V12H22L12,3L2,12H5V20H10Z"/>
                            </svg>
                            Interactive Dashboard
                        </a>
                        <a href="/api/docs" class="nav-btn secondary" target="_blank">
                            <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M14,17H7V15H14M17,13H7V11H17M17,9H7V7H17M19,3H5C3.89,3 3,3.89 3,5V19A2,2 0 0,0 5,21H19A2,2 0 0,0 21,19V5C21,3.89 20.1,3 19,3Z"/>
                            </svg>
                            Swagger UI
                        </a>
                    </div>
                </header>
                
                <div class="stats" id="stats">
                    <div class="stat-card">
                        <div class="stat-value"><span class="loading"></span></div>
                        <div class="stat-label">Total Memories</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value"><span class="loading"></span></div>
                        <div class="stat-label">Embedding Model</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value"><span class="loading"></span></div>
                        <div class="stat-label">Server Status</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value"><span class="loading"></span></div>
                        <div class="stat-label">Response Time</div>
                    </div>
                </div>
                
                <div class="action-buttons">
                    <a href="/api/docs" class="btn btn-primary">
                        <span>📚</span> Interactive API Docs
                    </a>
                    <a href="/api/redoc" class="btn btn-secondary">
                        <span>📖</span> ReDoc Documentation
                    </a>
                    <a href="https://github.com/doobidoo/mcp-memory-service" class="btn btn-secondary" target="_blank">
                        <span>🚀</span> GitHub Repository
                    </a>
                </div>
                
                <div class="endpoint-grid">
                    <div class="endpoint-card">
                        <div class="endpoint-header">
                            <h3><span class="endpoint-icon">💾</span> Memory Management</h3>
                            <p class="endpoint-description">Store, retrieve, and manage semantic memories</p>
                        </div>
                        <div class="endpoint-list">
                            <div class="endpoint-item" onclick="window.location.href='/api/docs#/memories/store_memory_api_memories_post'">
                                <span class="method method-post">POST</span>
                                <span class="endpoint-path">/api/memories</span>
                                <div class="endpoint-desc">Store a new memory with automatic embedding generation</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/docs#/memories/list_memories_api_memories_get'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/memories</span>
                                <div class="endpoint-desc">List all memories with pagination support</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/docs#/memories/get_memory_api_memories__content_hash__get'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/memories/{hash}</span>
                                <div class="endpoint-desc">Retrieve a specific memory by content hash</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/docs#/memories/delete_memory_api_memories__content_hash__delete'">
                                <span class="method method-delete">DELETE</span>
                                <span class="endpoint-path">/api/memories/{hash}</span>
                                <div class="endpoint-desc">Delete a memory and its embeddings</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="endpoint-card">
                        <div class="endpoint-header">
                            <h3><span class="endpoint-icon">🔍</span> Search Operations</h3>
                            <p class="endpoint-description">Powerful semantic and tag-based search</p>
                        </div>
                        <div class="endpoint-list">
                            <div class="endpoint-item" onclick="window.location.href='/api/docs#/search/semantic_search_api_search_post'">
                                <span class="method method-post">POST</span>
                                <span class="endpoint-path">/api/search</span>
                                <div class="endpoint-desc">Semantic similarity search using embeddings</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/docs#/search/tag_search_api_search_by_tag_post'">
                                <span class="method method-post">POST</span>
                                <span class="endpoint-path">/api/search/by-tag</span>
                                <div class="endpoint-desc">Search memories by tags (AND/OR logic)</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/docs#/search/time_search_api_search_by_time_post'">
                                <span class="method method-post">POST</span>
                                <span class="endpoint-path">/api/search/by-time</span>
                                <div class="endpoint-desc">Natural language time-based queries</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/docs#/search/find_similar_api_search_similar__content_hash__get'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/search/similar/{hash}</span>
                                <div class="endpoint-desc">Find memories similar to a specific one</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="endpoint-card">
                        <div class="endpoint-header">
                            <h3><span class="endpoint-icon">📡</span> Real-time Events</h3>
                            <p class="endpoint-description">Server-Sent Events for live updates</p>
                        </div>
                        <div class="endpoint-list">
                            <div class="endpoint-item" onclick="window.location.href='/api/events'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/events</span>
                                <div class="endpoint-desc">Subscribe to real-time memory events stream</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/events/stats'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/events/stats</span>
                                <div class="endpoint-desc">View SSE connection statistics</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/static/sse_test.html'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/static/sse_test.html</span>
                                <div class="endpoint-desc">Interactive SSE testing interface</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="endpoint-card">
                        <div class="endpoint-header">
                            <h3><span class="endpoint-icon">🏥</span> Health & Status</h3>
                            <p class="endpoint-description">Monitor service health and performance</p>
                        </div>
                        <div class="endpoint-list">
                            <div class="endpoint-item" onclick="window.location.href='/api/health'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/health</span>
                                <div class="endpoint-desc">Quick health check endpoint</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/health/detailed'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/health/detailed</span>
                                <div class="endpoint-desc">Detailed health with database statistics</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/docs'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/docs</span>
                                <div class="endpoint-desc">Interactive Swagger UI documentation</div>
                            </div>
                            <div class="endpoint-item" onclick="window.location.href='/api/redoc'">
                                <span class="method method-get">GET</span>
                                <span class="endpoint-path">/api/redoc</span>
                                <div class="endpoint-desc">Alternative ReDoc documentation</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <footer>
                    <p>Powered by cutting-edge technology</p>
                    <div class="tech-stack">
                        <div class="tech-badge">
                            <span>🐍</span> FastAPI
                        </div>
                        <div class="tech-badge">
                            <span>🗄️</span> SQLite-vec
                        </div>
                        <div class="tech-badge">
                            <span>🧠</span> Sentence Transformers
                        </div>
                        <div class="tech-badge">
                            <span>🔥</span> PyTorch
                        </div>
                        <div class="tech-badge">
                            <span>🌐</span> mDNS Discovery
                        </div>
                    </div>
                    <p style="margin-top: 2rem; opacity: 0.8;">
                        © 2025 MCP Memory Service | Apache 2.0 License
                    </p>
                </footer>
            </div>
            
            <script>
                // Fetch and display live stats
                async function updateStats() {
                    try {
                        const healthResponse = await fetch('/api/health');
                        const health = await healthResponse.json();
                        
                        const detailedResponse = await fetch('/api/health/detailed');
                        const detailed = await detailedResponse.json();
                        
                        const stats = document.getElementById('stats');
                        stats.innerHTML = `
                            <div class="stat-card">
                                <div class="stat-value">${detailed.statistics?.total_memories || 0}</div>
                                <div class="stat-label">Total Memories</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">all-MiniLM-L6-v2</div>
                                <div class="stat-label">Embedding Model</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value" style="color: var(--success);">● Healthy</div>
                                <div class="stat-label">Server Status</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">&lt;1ms</div>
                                <div class="stat-label">Response Time</div>
                            </div>
                        `;
                    } catch (error) {
                        console.error('Failed to fetch stats:', error);
                    }
                }
                
                // Update stats on page load
                updateStats();
                
                // Update stats every 30 seconds
                setInterval(updateStats, 30000);
            </script>

            <script>
                // Dynamic content loading for API overview
                function getBackendDisplayName(backend) {
                    const backendMap = {
                        'sqlite-vec': 'SQLite-vec',
                        'sqlite_vec': 'SQLite-vec',
                        'cloudflare': 'Cloudflare D1 + Vectorize',
                        'hybrid': 'Hybrid (SQLite-vec + Cloudflare)'
                    };
                    return backendMap[backend] || backend || 'Unknown Backend';
                }

                async function loadDynamicInfo() {
                    try {
                        // Load detailed health information
                        const response = await fetch('/api/health/detailed');
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}`);
                        }
                        const healthData = await response.json();

                        // Update version display
                        const versionEl = document.getElementById('version-display');
                        if (versionEl && healthData.version) {
                            versionEl.textContent = `v${healthData.version}`;
                        }

                        // Update backend name and subtitle
                        const backendNameEl = document.getElementById('backend-name');
                        const subtitleEl = document.getElementById('subtitle');

                        if (healthData.storage && healthData.storage.backend) {
                            const backendDisplay = getBackendDisplayName(healthData.storage.backend);

                            if (backendNameEl) {
                                backendNameEl.textContent = backendDisplay;
                            }

                            if (subtitleEl) {
                                subtitleEl.innerHTML = `Intelligent Semantic Memory with <span id="backend-name">${backendDisplay}</span>`;
                            }
                        }

                    } catch (error) {
                        console.error('Error loading dynamic info:', error);

                        // Fallback values on error
                        const versionEl = document.getElementById('version-display');
                        const backendNameEl = document.getElementById('backend-name');
                        const subtitleEl = document.getElementById('subtitle');

                        if (versionEl) {
                            versionEl.textContent = 'v?.?.?';
                        }

                        if (backendNameEl) {
                            backendNameEl.textContent = 'Unknown Backend';
                        }

                        if (subtitleEl) {
                            subtitleEl.innerHTML = 'Intelligent Semantic Memory with <span id="backend-name">Unknown Backend</span>';
                        }
                    }
                }

                // Load dynamic content when page loads
                document.addEventListener('DOMContentLoaded', loadDynamicInfo);
            </script>
        </body>
        </html>
        """

    @app.get("/api-overview", response_class=HTMLResponse)
    async def api_overview():
        """Serve the API documentation overview page."""
        return get_api_overview_html()

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
                    return f.read()
            else:
                # Fallback to original template if dashboard not found
                return html_template
        except Exception as e:
            # Error fallback to original template
            logger.warning(f"Error loading migrated dashboard: {e}")
            return html_template
    
    return app


# Create the app instance
app = create_app()


# Storage getter is now in dependencies.py