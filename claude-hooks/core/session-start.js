/**
 * Claude Code Session Start Hook
 * Automatically injects relevant memories at the beginning of each session
 */

const fs = require('fs').promises;
const path = require('path');

// Import utilities
const { detectProjectContext } = require('../utilities/project-detector');
const { scoreMemoryRelevance } = require('../utilities/memory-scorer');
const { formatMemoriesForContext } = require('../utilities/context-formatter');
const { detectContextShift, extractCurrentContext, determineRefreshStrategy } = require('../utilities/context-shift-detector');
const { analyzeGitContext, buildGitContextQuery } = require('../utilities/git-analyzer');
const { MemoryClient } = require('../utilities/memory-client');

/**
 * Load hook configuration
 */
async function loadConfig() {
    try {
        const configPath = path.join(__dirname, '../config.json');
        const configData = await fs.readFile(configPath, 'utf8');
        return JSON.parse(configData);
    } catch (error) {
        console.warn('[Memory Hook] Using default configuration:', error.message);
        return {
            memoryService: {
                protocol: 'auto',
                preferredProtocol: 'mcp',
                fallbackEnabled: true,
                http: {
                    endpoint: 'https://localhost:8443',
                    apiKey: 'test-key-123',
                    healthCheckTimeout: 3000,
                    useDetailedHealthCheck: false
                },
                mcp: {
                    serverCommand: ['uv', 'run', 'memory', 'server'],
                    serverWorkingDir: null,
                    connectionTimeout: 5000,
                    toolCallTimeout: 10000
                },
                defaultTags: ['claude-code', 'auto-generated'],
                maxMemoriesPerSession: 8,
                injectAfterCompacting: false
            },
            projectDetection: {
                gitRepository: true,
                packageFiles: ['package.json', 'pyproject.toml', 'Cargo.toml'],
                frameworkDetection: true,
                languageDetection: true
            },
            output: {
                verbose: true, // Default to verbose for backward compatibility
                showMemoryDetails: false, // Hide detailed memory scoring by default
                showProjectDetails: true, // Show project detection by default
                showScoringDetails: false, // Hide detailed scoring breakdown
                cleanMode: false // Default to normal output
            }
        };
    }
}

/**
 * Query memory service for health information (supports both HTTP and MCP)
 */
async function queryMemoryHealth(memoryClient) {
    try {
        const healthResult = await memoryClient.getHealthStatus();
        return healthResult;
    } catch (error) {
        return {
            success: false,
            error: error.message,
            fallback: true
        };
    }
}

/**
 * Parse health data into storage info structure (supports both HTTP and MCP responses)
 */
function parseHealthDataToStorageInfo(healthData) {
    try {
        // Handle MCP tool response format
        if (healthData.content && Array.isArray(healthData.content)) {
            const textContent = healthData.content.find(c => c.type === 'text')?.text;
            if (textContent) {
                try {
                    // Parse JSON from MCP response
                    const parsedData = JSON.parse(textContent.replace(/'/g, '"').replace(/True/g, 'true').replace(/False/g, 'false').replace(/None/g, 'null'));
                    return parseHealthDataToStorageInfo(parsedData);
                } catch (parseError) {
                    console.warn('[Memory Hook] Could not parse MCP health response:', parseError.message);
                    return getUnknownStorageInfo();
                }
            }
        }

        // Handle direct health data object
        const storage = healthData.storage || healthData || {};
        const system = healthData.system || {};
        const statistics = healthData.statistics || healthData.stats || {};
        
        // Determine icon based on backend
        let icon = '💾';
        switch (storage.backend?.toLowerCase()) {
            case 'sqlite-vec':
            case 'sqlite_vec':
                icon = '🪶';
                break;
            case 'chromadb':
            case 'chroma':
                icon = '📦';
                break;
            case 'cloudflare':
                icon = '☁️';
                break;
        }
        
        // Build description with status
        const backendName = storage.backend ? storage.backend.replace('_', '-') : 'Unknown';
        const statusText = storage.status === 'connected' ? 'Connected' : 
                          storage.status === 'disconnected' ? 'Disconnected' : 
                          storage.status || 'Unknown';
        
        const description = `${backendName} (${statusText})`;
        
        // Build location info
        let location = storage.database_path || storage.location || 'Unknown location';
        if (location.length > 50) {
            location = '...' + location.substring(location.length - 47);
        }
        
        // Determine type (local/remote/cloud)
        let type = 'unknown';
        if (storage.backend === 'cloudflare') {
            type = 'cloud';
        } else if (storage.database_path && storage.database_path.startsWith('/')) {
            type = 'local';
        } else if (location.includes('://')) {
            type = 'remote';
        } else {
            type = 'local';
        }
        
        return {
            backend: storage.backend || 'unknown',
            type: type,
            location: location,
            description: description,
            icon: icon,
            // Rich health data
            health: {
                status: storage.status,
                totalMemories: statistics.total_memories || storage.total_memories || 0,
                databaseSizeMB: statistics.database_size_mb || storage.database_size_mb || 0,
                uniqueTags: statistics.unique_tags || storage.unique_tags || 0,
                embeddingModel: storage.embedding_model || 'Unknown',
                platform: system.platform,
                uptime: healthData.uptime_seconds,
                accessible: storage.accessible
            }
        };
        
    } catch (error) {
        return getUnknownStorageInfo();
    }
}

/**
 * Get unknown storage info structure
 */
function getUnknownStorageInfo() {
    return {
        backend: 'unknown',
        type: 'unknown',
        location: 'Health parse error',
        description: 'Unknown Storage',
        icon: '❓',
        health: { status: 'error', totalMemories: 0 }
    };
}

/**
 * Detect storage backend configuration (fallback method)
 */
function detectStorageBackendFallback(config) {
    try {
        // Check environment variable first
        const envBackend = process.env.MCP_MEMORY_STORAGE_BACKEND?.toLowerCase();
        const endpoint = config.memoryService?.endpoint || 'https://localhost:8443';
        
        // Parse endpoint to determine if local or remote
        const url = new URL(endpoint);
        const isLocal = url.hostname === 'localhost' || url.hostname === '127.0.0.1' || url.hostname.endsWith('.local');
        
        let storageInfo = {
            backend: 'unknown',
            type: 'unknown',
            location: endpoint,
            description: 'Unknown Storage',
            icon: '💾',
            health: { status: 'unknown', totalMemories: 0 }
        };
        
        if (envBackend) {
            switch (envBackend) {
                case 'sqlite_vec':
                    storageInfo = {
                        backend: 'sqlite_vec',
                        type: 'local',
                        location: process.env.MCP_MEMORY_SQLITE_PATH || '~/.mcp-memory/memories.db',
                        description: 'SQLite-vec (Config)',
                        icon: '🪶',
                        health: { status: 'unknown', totalMemories: 0 }
                    };
                    break;
                    
                case 'chromadb':
                case 'chroma':
                    const chromaHost = process.env.MCP_MEMORY_CHROMADB_HOST;
                    const chromaPath = process.env.MCP_MEMORY_CHROMA_PATH;
                    
                    if (chromaHost) {
                        // Remote ChromaDB
                        const chromaPort = process.env.MCP_MEMORY_CHROMADB_PORT || '8000';
                        const ssl = process.env.MCP_MEMORY_CHROMADB_SSL === 'true';
                        const protocol = ssl ? 'https' : 'http';
                        storageInfo = {
                            backend: 'chromadb',
                            type: 'remote',
                            location: `${protocol}://${chromaHost}:${chromaPort}`,
                            description: 'ChromaDB (Remote Config)',
                            icon: '🌐',
                            health: { status: 'unknown', totalMemories: 0 }
                        };
                    } else {
                        // Local ChromaDB
                        storageInfo = {
                            backend: 'chromadb',
                            type: 'local',
                            location: chromaPath || '~/.mcp-memory/chroma',
                            description: 'ChromaDB (Config)',
                            icon: '📦',
                            health: { status: 'unknown', totalMemories: 0 }
                        };
                    }
                    break;
                    
                case 'cloudflare':
                    const accountId = process.env.CLOUDFLARE_ACCOUNT_ID;
                    storageInfo = {
                        backend: 'cloudflare',
                        type: 'cloud',
                        location: accountId ? `Account: ${accountId.substring(0, 8)}...` : 'Cloudflare Workers',
                        description: 'Cloudflare Vector (Config)',
                        icon: '☁️',
                        health: { status: 'unknown', totalMemories: 0 }
                    };
                    break;
            }
        } else {
            // Fallback: infer from endpoint
            if (isLocal) {
                storageInfo = {
                    backend: 'local_service',
                    type: 'local',
                    location: endpoint,
                    description: 'Local MCP Service',
                    icon: '💾',
                    health: { status: 'unknown', totalMemories: 0 }
                };
            } else {
                storageInfo = {
                    backend: 'remote_service',
                    type: 'remote',
                    location: endpoint,
                    description: 'Remote MCP Service',
                    icon: '🌐',
                    health: { status: 'unknown', totalMemories: 0 }
                };
            }
        }
        
        return storageInfo;
        
    } catch (error) {
        return {
            backend: 'unknown',
            type: 'unknown',
            location: 'Configuration Error',
            description: 'Unknown Storage',
            icon: '❓',
            health: { status: 'error', totalMemories: 0 }
        };
    }
}

/**
 * Query memory service for relevant memories (supports both HTTP and MCP)
 */
async function queryMemoryService(memoryClient, query) {
    try {
        // Add timeout for each individual query (2 seconds max)
        const queryTimeout = new Promise((resolve) =>
            setTimeout(() => resolve([]), 2000)
        );

        let memories = [];

        // Use time-based queries for recent memories
        const queryPromise = query.timeFilter ?
            memoryClient.queryMemoriesByTime(`${query.semanticQuery} ${query.timeFilter}`, query.limit) :
            memoryClient.queryMemories(query.semanticQuery, query.limit);

        memories = await Promise.race([queryPromise, queryTimeout]);

        return memories || [];
    } catch (error) {
        console.warn('[Memory Hook] Memory query error:', error.message);
        return [];
    }
}

// ANSI Colors for console output
const CONSOLE_COLORS = {
    RESET: '\x1b[0m',
    BRIGHT: '\x1b[1m',
    DIM: '\x1b[2m',
    CYAN: '\x1b[36m',
    GREEN: '\x1b[32m',
    BLUE: '\x1b[34m',
    YELLOW: '\x1b[33m',
    GRAY: '\x1b[90m',
    RED: '\x1b[31m'
};

/**
 * Main session start hook function with enhanced visual output
 */
async function onSessionStart(context) {
    // Global timeout wrapper to prevent hook from hanging
    const HOOK_TIMEOUT = 8000; // 8 seconds (leave 2s buffer for cleanup)
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Hook timeout - completing early')), HOOK_TIMEOUT);
    });

    try {
        return await Promise.race([
            executeSessionStart(context),
            timeoutPromise
        ]);
    } catch (error) {
        if (error.message.includes('Hook timeout')) {
            console.log(`${CONSOLE_COLORS.YELLOW}⏱️  Memory Hook${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}Completed with timeout (normal for slow connections)${CONSOLE_COLORS.RESET}`);
            return;
        }
        throw error;
    }
}

/**
 * Main execution logic (wrapped by timeout)
 */
async function executeSessionStart(context) {
    try {
        // Load configuration first to check verbosity settings
        const config = await loadConfig();
        const verbose = config.output?.verbose !== false; // Default to true
        const cleanMode = config.output?.cleanMode === true; // Default to false
        const showMemoryDetails = config.output?.showMemoryDetails === true;
        const showProjectDetails = config.output?.showProjectDetails !== false; // Default to true
        
        if (verbose && !cleanMode) {
            console.log(`${CONSOLE_COLORS.CYAN}🧠 Memory Hook${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Initializing session awareness...`);
        }
        
        // Check if this is triggered by a compacting event and skip if configured to do so
        if (context.trigger === 'compacting' || context.event === 'memory-compacted') {
            if (!config.memoryService.injectAfterCompacting) {
                console.log(`${CONSOLE_COLORS.YELLOW}⏸️  Memory Hook${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Skipping injection after compacting`);
                return;
            }
            console.log(`${CONSOLE_COLORS.GREEN}▶️  Memory Hook${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Proceeding with injection after compacting`);
        }
        
        // For non-session-start events, use smart timing to decide if refresh is needed
        if (context.trigger !== 'session-start' && context.trigger !== 'start') {
            const currentContext = extractCurrentContext(context.conversationState || {}, context.workingDirectory);
            const previousContext = context.previousContext || context.conversationState?.previousContext;
            
            if (previousContext) {
                const shiftDetection = detectContextShift(currentContext, previousContext);
                
                if (!shiftDetection.shouldRefresh) {
                    console.log(`${CONSOLE_COLORS.GRAY}⏸️  Memory Hook${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}No context shift detected, skipping${CONSOLE_COLORS.RESET}`);
                    return;
                }
                
                console.log(`${CONSOLE_COLORS.BLUE}🔄 Memory Hook${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Context shift: ${shiftDetection.description}`);
            }
        }
        
        // Detect project context
        const projectContext = await detectProjectContext(context.workingDirectory || process.cwd());
        if (verbose && showProjectDetails && !cleanMode) {
            console.log(`${CONSOLE_COLORS.BLUE}📂 Project${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.BRIGHT}${projectContext.name}${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}(${projectContext.language})${CONSOLE_COLORS.RESET}`);
        }
        
        // Initialize memory client and detect storage backend
        const showStorageSource = config.memoryService?.showStorageSource !== false; // Default to true
        const sourceDisplayMode = config.memoryService?.sourceDisplayMode || 'brief';
        let memoryClient = null;
        let storageInfo = null;
        let connectionInfo = null;

        if (showStorageSource && verbose && !cleanMode) {
            // Initialize unified memory client for health check and memory queries
            try {
                memoryClient = new MemoryClient(config.memoryService);
                const connection = await memoryClient.connect();
                connectionInfo = memoryClient.getConnectionInfo();

                if (verbose && showMemoryDetails && !cleanMode && connectionInfo?.activeProtocol) {
                    console.log(`${CONSOLE_COLORS.CYAN}🔗 Connection${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Using ${CONSOLE_COLORS.BRIGHT}${connectionInfo.activeProtocol.toUpperCase()}${CONSOLE_COLORS.RESET} protocol`);
                }

                const healthResult = await queryMemoryHealth(memoryClient);
                
                    if (healthResult.success) {
                        storageInfo = parseHealthDataToStorageInfo(healthResult.data);

                        // Display based on mode with rich health information
                        if (sourceDisplayMode === 'detailed') {
                            console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${CONSOLE_COLORS.BRIGHT}${storageInfo.description}${CONSOLE_COLORS.RESET}`);
                            console.log(`${CONSOLE_COLORS.CYAN}📍 Location${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}${storageInfo.location}${CONSOLE_COLORS.RESET}`);
                            if (storageInfo.health.totalMemories > 0) {
                                console.log(`${CONSOLE_COLORS.CYAN}📊 Database${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GREEN}${storageInfo.health.totalMemories} memories${CONSOLE_COLORS.RESET}, ${CONSOLE_COLORS.YELLOW}${storageInfo.health.databaseSizeMB}MB${CONSOLE_COLORS.RESET}, ${CONSOLE_COLORS.BLUE}${storageInfo.health.uniqueTags} tags${CONSOLE_COLORS.RESET}`);
                            }
                        } else if (sourceDisplayMode === 'brief') {
                            const memoryCount = storageInfo.health.totalMemories > 0 ? ` • ${storageInfo.health.totalMemories} memories` : '';
                            const sizeInfo = storageInfo.health.databaseSizeMB > 0 ? ` • ${storageInfo.health.databaseSizeMB}MB` : '';
                            console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${CONSOLE_COLORS.BRIGHT}${storageInfo.description}${CONSOLE_COLORS.RESET}${memoryCount}${sizeInfo}`);
                            if (storageInfo.location && sourceDisplayMode === 'brief') {
                                console.log(`${CONSOLE_COLORS.CYAN}📍 Path${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}${storageInfo.location}${CONSOLE_COLORS.RESET}`);
                            }
                        } else if (sourceDisplayMode === 'icon-only') {
                            console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${storageInfo.backend} • ${storageInfo.health.totalMemories} memories`);
                        }
                    } else {
                        // Fallback to environment/config detection when MCP health check fails
                        if (verbose && showMemoryDetails && !cleanMode) {
                            console.log(`${CONSOLE_COLORS.YELLOW}⚠️  MCP Health Check${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}${healthResult.error}, using config fallback${CONSOLE_COLORS.RESET}`);
                        }

                        storageInfo = detectStorageBackendFallback(config);

                        if (sourceDisplayMode === 'detailed') {
                            console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${CONSOLE_COLORS.BRIGHT}${storageInfo.description}${CONSOLE_COLORS.RESET}`);
                            console.log(`${CONSOLE_COLORS.CYAN}📍 Location${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}${storageInfo.location}${CONSOLE_COLORS.RESET}`);
                        } else if (sourceDisplayMode === 'brief') {
                            console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${CONSOLE_COLORS.BRIGHT}${storageInfo.description}${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}(${storageInfo.location})${CONSOLE_COLORS.RESET}`);
                        } else if (sourceDisplayMode === 'icon-only') {
                            console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${storageInfo.backend}`);
                        }
                    }
            } catch (error) {
                // Memory client connection failed, fall back to environment detection
                if (verbose && showMemoryDetails && !cleanMode) {
                    console.log(`${CONSOLE_COLORS.YELLOW}⚠️  Memory Connection${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}${error.message}, using environment fallback${CONSOLE_COLORS.RESET}`);
                }

                storageInfo = detectStorageBackendFallback(config);

                if (sourceDisplayMode === 'brief') {
                    console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${CONSOLE_COLORS.BRIGHT}${storageInfo.description}${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}(${storageInfo.location})${CONSOLE_COLORS.RESET}`);
                }
            }
        } else {
            // Health check disabled, use config fallback
            storageInfo = detectStorageBackendFallback(config);

            if (sourceDisplayMode === 'detailed') {
                console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${CONSOLE_COLORS.BRIGHT}${storageInfo.description}${CONSOLE_COLORS.RESET}`);
                console.log(`${CONSOLE_COLORS.CYAN}📍 Location${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}${storageInfo.location}${CONSOLE_COLORS.RESET}`);
            } else if (sourceDisplayMode === 'brief') {
                console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${CONSOLE_COLORS.BRIGHT}${storageInfo.description}${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}(${storageInfo.location})${CONSOLE_COLORS.RESET}`);
            } else if (sourceDisplayMode === 'icon-only') {
                console.log(`${CONSOLE_COLORS.CYAN}💾 Storage${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${storageInfo.icon} ${storageInfo.backend}`);
            }
        }

        // Analyze git context if enabled
        const gitAnalysisEnabled = config.gitAnalysis?.enabled !== false; // Default to true
        const showGitAnalysis = config.output?.showGitAnalysis !== false; // Default to true
        let gitContext = null;
        
        if (gitAnalysisEnabled) {
            if (verbose && showGitAnalysis && !cleanMode) {
                console.log(`${CONSOLE_COLORS.CYAN}📊 Git Analysis${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Analyzing repository context...`);
            }
            
            gitContext = await analyzeGitContext(context.workingDirectory || process.cwd(), {
                commitLookback: config.gitAnalysis?.commitLookback || 14,
                maxCommits: config.gitAnalysis?.maxCommits || 20,
                includeChangelog: config.gitAnalysis?.includeChangelog !== false,
                verbose: showGitAnalysis && showMemoryDetails && !cleanMode
            });
            
            if (gitContext && verbose && showGitAnalysis && !cleanMode) {
                const { commits, changelogEntries, repositoryActivity, developmentKeywords } = gitContext;
                console.log(`${CONSOLE_COLORS.CYAN}📊 Git Context${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${commits.length} commits, ${changelogEntries?.length || 0} changelog entries`);
                
                if (showMemoryDetails) {
                    const topKeywords = developmentKeywords.keywords.slice(0, 5).join(', ');
                    if (topKeywords) {
                        console.log(`${CONSOLE_COLORS.CYAN}🔑 Keywords${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.YELLOW}${topKeywords}${CONSOLE_COLORS.RESET}`);
                    }
                }
            }
        }
        
        // Initialize memory client for memory queries if not already connected
        if (!memoryClient) {
            try {
                // Add quick timeout for initial connection
                const connectionTimeout = new Promise((_, reject) =>
                    setTimeout(() => reject(new Error('Quick connection timeout')), 2000)
                );

                memoryClient = new MemoryClient(config.memoryService);
                await Promise.race([
                    memoryClient.connect(),
                    connectionTimeout
                ]);
                connectionInfo = memoryClient.getConnectionInfo();
            } catch (error) {
                if (verbose && !cleanMode) {
                    console.log(`${CONSOLE_COLORS.YELLOW}⚠️  Memory Connection${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}Failed to connect for memory queries: ${error.message}${CONSOLE_COLORS.RESET}`);
                }
                memoryClient = null;
            }
        }

        // Multi-phase memory retrieval for better recency prioritization
        const allMemories = [];
        const maxMemories = config.memoryService.maxMemoriesPerSession;
        const recentFirstMode = config.memoryService.recentFirstMode !== false; // Default to true
        const recentRatio = config.memoryService.recentMemoryRatio || 0.6;
        const recentTimeWindow = config.memoryService.recentTimeWindow || 'last-week';
        const fallbackTimeWindow = config.memoryService.fallbackTimeWindow || 'last-month';
        const showPhaseDetails = config.output?.showPhaseDetails !== false; // Default to true
        
        if (recentFirstMode) {
            // Phase 0: Git Context Phase (NEW - highest priority for repository-aware memories)
            if (gitContext && gitContext.developmentKeywords.keywords.length > 0) {
                const maxGitMemories = config.gitAnalysis?.maxGitMemories || 3;
                const gitQueries = buildGitContextQuery(projectContext, gitContext.developmentKeywords, context.userMessage);
                
                if (verbose && showPhaseDetails && !cleanMode && gitQueries.length > 0) {
                    console.log(`${CONSOLE_COLORS.GREEN}⚡ Phase 0${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Git-aware memory search (${maxGitMemories} slots, ${gitQueries.length} queries)`);
                }
                
                // Execute git-context queries
                for (const gitQuery of gitQueries.slice(0, 2)) { // Limit to top 2 queries for performance
                    if (allMemories.length >= maxGitMemories) break;
                    
                    const gitMemories = await queryMemoryService(memoryClient, {
                        semanticQuery: gitQuery.semanticQuery,
                        limit: Math.min(maxGitMemories - allMemories.length, 3),
                        timeFilter: 'last-2-weeks' // Focus on recent memories for git context
                    });
                    
                    if (gitMemories && gitMemories.length > 0) {
                        // Mark these memories as git-context derived for scoring
                        const markedMemories = gitMemories.map(mem => ({
                            ...mem,
                            _gitContextType: gitQuery.type,
                            _gitContextSource: gitQuery.source,
                            _gitContextWeight: config.gitAnalysis?.gitContextWeight || 1.2
                        }));
                        
                        // Avoid duplicates from previous git queries
                        const newGitMemories = markedMemories.filter(newMem => 
                            !allMemories.some(existing => 
                                existing.content && newMem.content && 
                                existing.content.substring(0, 100) === newMem.content.substring(0, 100)
                            )
                        );
                        
                        allMemories.push(...newGitMemories);
                        
                        if (verbose && showMemoryDetails && !cleanMode && newGitMemories.length > 0) {
                            console.log(`${CONSOLE_COLORS.GREEN}  📋 Git Query${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} [${gitQuery.type}] found ${newGitMemories.length} memories`);
                        }
                    }
                }
            }
            
            // Phase 1: Recent memories - high priority
            const remainingSlotsAfterGit = Math.max(0, maxMemories - allMemories.length);
            if (remainingSlotsAfterGit > 0) {
                // Build enhanced semantic query with git context
                let recentSemanticQuery = context.userMessage ? 
                    `recent ${projectContext.name} ${context.userMessage}` :
                    `recent ${projectContext.name} development decisions insights`;
                
                // Add git context if available
                if (projectContext.git?.branch) {
                    recentSemanticQuery += ` ${projectContext.git.branch}`;
                }
                if (projectContext.git?.lastCommit) {
                    recentSemanticQuery += ` latest changes commit`;
                }
                
                // Add development keywords from git analysis
                if (gitContext && gitContext.developmentKeywords.keywords.length > 0) {
                    const topKeywords = gitContext.developmentKeywords.keywords.slice(0, 3).join(' ');
                    recentSemanticQuery += ` ${topKeywords}`;
                }
                const recentQuery = {
                    semanticQuery: recentSemanticQuery,
                    limit: Math.max(Math.floor(remainingSlotsAfterGit * recentRatio), 2), // Adjusted for remaining slots
                    timeFilter: recentTimeWindow
                };
                
                if (verbose && showMemoryDetails && showPhaseDetails && !cleanMode) {
                    console.log(`${CONSOLE_COLORS.BLUE}🕒 Phase 1${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Searching recent memories (${recentTimeWindow}, ${recentQuery.limit} slots)`);
                }
                
                const recentMemories = await queryMemoryService(memoryClient, recentQuery);
                
                // Filter out duplicates from git context phase
                if (recentMemories && recentMemories.length > 0) {
                    const newRecentMemories = recentMemories.filter(newMem => 
                        !allMemories.some(existing => 
                            existing.content && newMem.content && 
                            existing.content.substring(0, 100) === newMem.content.substring(0, 100)
                        )
                    );
                    
                    allMemories.push(...newRecentMemories);
                }
            }
            
            // Phase 2: Important tagged memories - fill remaining slots  
            const remainingSlots = maxMemories - allMemories.length;
            if (remainingSlots > 0) {
                // Build enhanced query for important memories
                let importantSemanticQuery = `${projectContext.name} important decisions architecture`;
                if (projectContext.language && projectContext.language !== 'Unknown') {
                    importantSemanticQuery += ` ${projectContext.language}`;
                }
                if (projectContext.frameworks?.length > 0) {
                    importantSemanticQuery += ` ${projectContext.frameworks.join(' ')}`;
                }
                
                const importantQuery = {
                    tags: [
                        projectContext.name,
                        'key-decisions',
                        'architecture', 
                        'claude-code-reference'
                    ].filter(Boolean),
                    semanticQuery: importantSemanticQuery,
                    limit: remainingSlots,
                    timeFilter: 'last-2-weeks'
                };
                
                if (verbose && showMemoryDetails && showPhaseDetails && !cleanMode) {
                    console.log(`${CONSOLE_COLORS.BLUE}🎯 Phase 2${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Searching important tagged memories (${remainingSlots} slots)`);
                }
                
                const importantMemories = await queryMemoryService(memoryClient, importantQuery);
                
                // Avoid duplicates by checking content similarity  
                const newMemories = (importantMemories || []).filter(newMem => 
                    !allMemories.some(existing => 
                        existing.content && newMem.content && 
                        existing.content.substring(0, 100) === newMem.content.substring(0, 100)
                    )
                );
                
                allMemories.push(...newMemories);
            }
            
            // Phase 3: Fallback to general project context if still need more
            const stillRemaining = maxMemories - allMemories.length;
            if (stillRemaining > 0 && allMemories.length < 3) {
                const fallbackQuery = {
                    semanticQuery: `${projectContext.name} project context`,
                    limit: stillRemaining,
                    timeFilter: fallbackTimeWindow
                };
                
                if (verbose && showMemoryDetails && showPhaseDetails && !cleanMode) {
                    console.log(`${CONSOLE_COLORS.BLUE}🔄 Phase 3${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Fallback general context (${stillRemaining} slots, ${fallbackTimeWindow})`);
                }
                
                const fallbackMemories = await queryMemoryService(memoryClient, fallbackQuery);
                
                const newFallbackMemories = (fallbackMemories || []).filter(newMem => 
                    !allMemories.some(existing => 
                        existing.content && newMem.content && 
                        existing.content.substring(0, 100) === newMem.content.substring(0, 100)
                    )
                );
                
                allMemories.push(...newFallbackMemories);
            }
        } else {
            // Legacy single-phase approach 
            const memoryQuery = {
                tags: [
                    projectContext.name,
                    `language:${projectContext.language}`,
                    'key-decisions',
                    'architecture',
                    'recent-insights',
                    'claude-code-reference'
                ].filter(Boolean),
                semanticQuery: context.userMessage ? 
                    `${projectContext.name} ${context.userMessage}` :
                    `${projectContext.name} project context decisions architecture`,
                limit: maxMemories,
                timeFilter: 'last-2-weeks'
            };
            
            const legacyMemories = await queryMemoryService(memoryClient, memoryQuery);
            
            allMemories.push(...(legacyMemories || []));
        }
        
        // Skip memory retrieval if no memory client available
        if (!memoryClient) {
            if (verbose && !cleanMode) {
                console.log(`${CONSOLE_COLORS.YELLOW}⚠️  Memory Retrieval${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}Skipped due to connection failure${CONSOLE_COLORS.RESET}`);
            }
            // Skip memory operations but don't return - still complete the hook
            if (verbose && showMemoryDetails && !cleanMode) {
                console.log(`${CONSOLE_COLORS.YELLOW}📭 Memory Search${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}No memory service available${CONSOLE_COLORS.RESET}`);
            }
        }

        // Use the collected memories from all phases
        const memories = allMemories.slice(0, maxMemories);
        
        if (memories.length > 0) {
            // Analyze memory recency for better reporting
            const now = new Date();
            const recentCount = memories.filter(m => {
                if (!m.created_at_iso) return false;
                const memDate = new Date(m.created_at_iso);
                const daysDiff = (now - memDate) / (1000 * 60 * 60 * 24);
                return daysDiff <= 7; // Within last week
            }).length;
            
            if (verbose && !cleanMode) {
                const recentText = recentCount > 0 ? ` ${CONSOLE_COLORS.GREEN}(${recentCount} recent)${CONSOLE_COLORS.RESET}` : '';
                console.log(`${CONSOLE_COLORS.GREEN}📚 Memory Search${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Found ${CONSOLE_COLORS.BRIGHT}${memories.length}${CONSOLE_COLORS.RESET} relevant memories${recentText}`);
            }
            
            // Score memories for relevance (with enhanced recency weighting)
            let scoredMemories = scoreMemoryRelevance(memories, projectContext, {
                verbose: showMemoryDetails,
                enhanceRecency: recentFirstMode
            });

            // Apply git context weight boost to git-derived memories
            scoredMemories = scoredMemories.map(memory => {
                if (memory._gitContextWeight && memory._gitContextWeight !== 1.0) {
                    return {
                        ...memory,
                        relevanceScore: Math.min(1.0, memory.relevanceScore * memory._gitContextWeight)
                    };
                }
                return memory;
            }).sort((a, b) => b.relevanceScore - a.relevanceScore); // Re-sort after boost

            // Show top scoring memories with recency info
            if (verbose && showMemoryDetails && scoredMemories.length > 0 && !cleanMode) {
                const topMemories = scoredMemories.slice(0, 3);
                const memoryInfo = topMemories.map(m => {
                    const score = `${(m.relevanceScore * 100).toFixed(0)}%`;
                    let recencyFlag = '';
                    if (m.created_at_iso) {
                        const daysDiff = (now - new Date(m.created_at_iso)) / (1000 * 60 * 60 * 24);
                        if (daysDiff <= 1) recencyFlag = '🕒';
                        else if (daysDiff <= 7) recencyFlag = '📅';
                    }
                    return `${score}${recencyFlag}`;
                }).join(', ');
                console.log(`${CONSOLE_COLORS.CYAN}🎯 Scoring${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Top relevance: ${CONSOLE_COLORS.YELLOW}${memoryInfo}${CONSOLE_COLORS.RESET}`);
            }
            
            // Determine refresh strategy based on context
            const strategy = context.trigger && context.previousContext ? 
                determineRefreshStrategy(detectContextShift(
                    extractCurrentContext(context.conversationState || {}, context.workingDirectory),
                    context.previousContext
                )) : {
                    maxMemories: config.memoryService.maxMemoriesPerSession,
                    includeScore: false,
                    message: '🧠 Loading relevant memory context...'
                };
            
            // Take top scored memories based on strategy
            const maxMemories = Math.min(strategy.maxMemories || config.memoryService.maxMemoriesPerSession, scoredMemories.length);
            const topMemories = scoredMemories.slice(0, maxMemories);
            
            // Show actual memory processing info (moved from deduplication)
            if (verbose && showMemoryDetails && !cleanMode) {
                const totalCollected = allMemories.length;
                const actualUsed = Math.min(maxMemories, scoredMemories.length);
                if (totalCollected > actualUsed) {
                    console.log(`[Context Formatter] Selected ${actualUsed} from ${totalCollected} collected memories`);
                }
                console.log(`${CONSOLE_COLORS.CYAN}🔄 Processing${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${actualUsed} memories selected`);
            }
            
            // Format memories for context injection with strategy-based options
            const contextMessage = formatMemoriesForContext(topMemories, projectContext, {
                includeScore: strategy.includeScore || false,
                groupByCategory: maxMemories > 3,
                maxMemories: maxMemories,
                includeTimestamp: true,
                maxContentLength: config.contextFormatting?.maxContentLength || 500,
                maxContentLengthCLI: config.contextFormatting?.maxContentLengthCLI || 400,
                maxContentLengthCategorized: config.contextFormatting?.maxContentLengthCategorized || 350,
                storageInfo: showStorageSource ? (storageInfo || detectStorageBackend(config)) : null
            });
            
            // Inject context into session
            if (context.injectSystemMessage) {
                await context.injectSystemMessage(contextMessage);
                if (!cleanMode) {
                    console.log(`${CONSOLE_COLORS.GREEN}✅ Memory Hook${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Context injected ${CONSOLE_COLORS.GRAY}(${maxMemories} memories)${CONSOLE_COLORS.RESET}`);
                }
            } else if (verbose && !cleanMode) {
                // Fallback: log context for manual copying with styling
                console.log(`\n${CONSOLE_COLORS.CYAN}╭──────────────────────────────────────────╮${CONSOLE_COLORS.RESET}`);
                console.log(`${CONSOLE_COLORS.CYAN}│${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.BRIGHT}Memory Context for Manual Copy${CONSOLE_COLORS.RESET}          ${CONSOLE_COLORS.CYAN}│${CONSOLE_COLORS.RESET}`);
                console.log(`${CONSOLE_COLORS.CYAN}╰──────────────────────────────────────────╯${CONSOLE_COLORS.RESET}`);
                // Clean output to remove session-start-hook wrapper tags
                const cleanedMessage = contextMessage.replace(/<\/?session-start-hook>/g, '');
                console.log(cleanedMessage);
                console.log(`${CONSOLE_COLORS.CYAN}╰──────────────────────────────────────────╯${CONSOLE_COLORS.RESET}\n`);
            }
        } else if (verbose && showMemoryDetails && !cleanMode) {
            console.log(`${CONSOLE_COLORS.YELLOW}📭 Memory Search${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}No relevant memories found${CONSOLE_COLORS.RESET}`);
        }

    } catch (error) {
        console.error(`${CONSOLE_COLORS.RED}❌ Memory Hook Error${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${error.message}`);
        // Fail gracefully - don't prevent session from starting
    } finally {
        // Ensure MCP client cleanup even on error
        try {
            if (memoryClient && typeof memoryClient.disconnect === 'function') {
                await memoryClient.disconnect();
            }
        } catch (error) {
            // Ignore cleanup errors silently
        }
    }
}

/**
 * Hook metadata for Claude Code
 */
module.exports = {
    name: 'memory-awareness-session-start',
    version: '2.3.0',
    description: 'Automatically inject relevant memories at session start with git-aware repository context',
    trigger: 'session-start',
    handler: onSessionStart,
    config: {
        async: true,
        timeout: 15000, // Increased timeout for git analysis
        priority: 'high'
    }
};

// Direct execution support for testing
if (require.main === module) {
    // Test the hook with mock context
    const mockContext = {
        workingDirectory: process.cwd(),
        sessionId: 'test-session',
        injectSystemMessage: async (message) => {
            const lines = message.split('\n');
            const maxLength = Math.min(80, Math.max(25, ...lines.map(l => l.length)));
            const border = '─'.repeat(maxLength - 2);
            
            console.log(`\n${CONSOLE_COLORS.CYAN}╭─${border}─╮${CONSOLE_COLORS.RESET}`);
            console.log(`${CONSOLE_COLORS.CYAN}│${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.BRIGHT}🧠 Injected Memory Context${CONSOLE_COLORS.RESET}${' '.repeat(maxLength - 27)} ${CONSOLE_COLORS.CYAN}│${CONSOLE_COLORS.RESET}`);
            console.log(`${CONSOLE_COLORS.CYAN}╰─${border}─╯${CONSOLE_COLORS.RESET}`);
            console.log(message);
            console.log(`${CONSOLE_COLORS.CYAN}╰─${border}─╯${CONSOLE_COLORS.RESET}`);
        }
    };
    
    onSessionStart(mockContext)
        .then(() => {
            // Test completed quietly
        })
        .catch(error => console.error(`${CONSOLE_COLORS.RED}❌ Hook test failed:${CONSOLE_COLORS.RESET} ${error.message}`));
}