/**
 * Claude Code Session Start Hook
 * Automatically injects relevant memories at the beginning of each session
 */

const fs = require('fs').promises;
const path = require('path');

// Import utilities
const { detectProjectContext } = require('../utilities/project-detector');
const { scoreMemoryRelevance, analyzeMemoryAgeDistribution, calculateAdaptiveGitWeight } = require('../utilities/memory-scorer');
const { formatMemoriesForContext } = require('../utilities/context-formatter');
const { detectContextShift, extractCurrentContext, determineRefreshStrategy } = require('../utilities/context-shift-detector');
const { analyzeGitContext, buildGitContextQuery } = require('../utilities/git-analyzer');
const { MemoryClient } = require('../utilities/memory-client');
const { getVersionInfo, formatVersionDisplay } = require('../utilities/version-checker');
const { detectUserOverrides, logOverride } = require('../utilities/user-override-detector');

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
                preferredProtocol: 'http',
                fallbackEnabled: true,
                http: {
                    endpoint: 'http://127.0.0.1:8889',
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
        
        // Build location info (use cwd as better fallback than "Unknown")
        let location = storage.database_path || storage.location || process.cwd();
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
        const endpoint = config.memoryService?.http?.endpoint || 'http://127.0.0.1:8889';
        
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
 * Query memory service using code execution (token-efficient)
 */
async function queryMemoryServiceViaCode(query, config) {
    const startTime = Date.now();
    const enableMetrics = config?.codeExecution?.enableMetrics !== false;

    try {
        const { execSync } = require('child_process');

        // Escape query strings for safe shell execution
        const escapeForPython = (str) => str.replace(/"/g, '\\"').replace(/\n/g, '\\n');

        // Build Python code for memory search
        // Use v8.19.0+ Code Execution Interface API for optimal performance
        // Note: time_filter not supported in Code Execution API, only in MCP tools
        const pythonCode = `
import sys
import json
from datetime import datetime
from mcp_memory_service.api import search

try:
    # Execute search with semantic query and limit (time filtering done server-side)
    results = search("${escapeForPython(query.semanticQuery || '')}", limit=${query.limit || 8})

    # Format compact output
    output = {
        'success': True,
        'memories': [
            {
                'hash': m.hash,
                'preview': m.preview,
                'tags': list(m.tags),
                'created': m.created,
                'created_at': m.created,
                'created_at_iso': datetime.fromtimestamp(m.created).isoformat(),
                'score': m.score,
                'content': m.preview  # Use preview as content for compatibility
            }
            for m in results.memories
        ],
        'total': results.total,
        'method': 'code_execution'
    }
    print(json.dumps(output))
    sys.exit(0)
except Exception as e:
    print(json.dumps({'success': False, 'error': str(e), 'method': 'code_execution'}))
    sys.exit(1)
`;

        // Get Python path from config
        const pythonPath = config?.codeExecution?.pythonPath || 'python3';
        const timeout = config?.codeExecution?.timeout || 5000;

        // Execute Python code with timeout (suppress warnings to avoid stderr failures)
        const result = execSync(`${pythonPath} -W ignore -c "${pythonCode.replace(/"/g, '\\"')}"`, {
            encoding: 'utf-8',
            timeout: timeout,
            stdio: ['pipe', 'pipe', 'pipe']
        });

        const parsed = JSON.parse(result);

        if (parsed.success) {
            const executionTime = Date.now() - startTime;

            // Calculate token savings estimate
            const memoriesRetrieved = (parsed.memories || []).length;
            const mcpTokens = 1200 + (memoriesRetrieved * 300); // Conservative MCP estimate
            const codeTokens = 20 + (memoriesRetrieved * 25); // Code execution tokens
            const tokensSaved = mcpTokens - codeTokens;
            const reductionPercent = ((tokensSaved / mcpTokens) * 100).toFixed(1);

            // Store metrics for reporting
            if (enableMetrics) {
                parsed._metrics = {
                    executionTime,
                    memoriesRetrieved,
                    mcpTokensEstimate: mcpTokens,
                    codeTokensEstimate: codeTokens,
                    tokensSaved,
                    reductionPercent
                };
            }

            return parsed.memories || [];
        } else {
            throw new Error(parsed.error || 'Code execution failed');
        }
    } catch (error) {
        // Silently return null to trigger MCP fallback
        // Error logging suppressed - fallback is expected when module not installed
        return null;
    }
}

/**
 * Query memory service for relevant memories (supports code execution with MCP fallback)
 */
async function queryMemoryService(memoryClient, query, config) {
    const startTime = Date.now();

    try {
        // Check if code execution is enabled
        const codeExecutionEnabled = config?.codeExecution?.enabled !== false; // Default true
        const fallbackToMCP = config?.codeExecution?.fallbackToMCP !== false; // Default true
        const enableMetrics = config?.codeExecution?.enableMetrics !== false;

        // Phase 1: Try code execution first (75% token reduction)
        if (codeExecutionEnabled) {
            const codeResult = await queryMemoryServiceViaCode(query, config);

            if (codeResult !== null) {
                const executionTime = Date.now() - startTime;

                // Extract metrics if available
                const metrics = codeResult._metrics || {};

                // Success! Log token savings
                if (config?.output?.verbose && config?.output?.showMemoryDetails && enableMetrics) {
                    const tokenInfo = metrics.reductionPercent ?
                        ` ${CONSOLE_COLORS.GRAY}(${metrics.reductionPercent}% reduction, ${metrics.tokensSaved} tokens saved)${CONSOLE_COLORS.RESET}` :
                        ` ${CONSOLE_COLORS.GRAY}(75% reduction)${CONSOLE_COLORS.RESET}`;
                    console.log(`${CONSOLE_COLORS.GREEN}⚡ Code Execution${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Token-efficient path${tokenInfo}`);
                }

                return codeResult;
            }
        }

        // Phase 2: Fallback to MCP tools if code execution failed
        if (fallbackToMCP && memoryClient) {
            if (config?.output?.verbose && config?.output?.showMemoryDetails) {
                console.log(`${CONSOLE_COLORS.YELLOW}↩️  MCP Fallback${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}Using standard MCP tools${CONSOLE_COLORS.RESET}`);
            }

            // Add timeout for each individual query (2 seconds max)
            const queryTimeout = new Promise((resolve) =>
                setTimeout(() => resolve([]), 2000)
            );

            let memories = [];

            // Use time-based queries with semantic filtering for relevant recent memories
            const queryPromise = query.timeFilter ?
                memoryClient.queryMemoriesByTime(query.timeFilter, query.limit, query.semanticQuery) :
                memoryClient.queryMemories(query.semanticQuery, query.limit);

            memories = await Promise.race([queryPromise, queryTimeout]);

            return memories || [];
        }

        return [];
    } catch (error) {
        console.warn('[Memory Hook] Memory query error:', error.message);
        return [];
    }
}

/**
 * Calculate content similarity between two normalized strings
 * Uses word overlap similarity with 80% threshold for deduplication
 */
function calculateContentSimilarity(str1, str2) {
    if (!str1 || !str2) return 0;
    if (str1 === str2) return 1;

    // Use simple word overlap similarity
    const words1 = new Set(str1.split(/\s+/).filter(w => w.length > 3));
    const words2 = new Set(str2.split(/\s+/).filter(w => w.length > 3));

    if (words1.size === 0 && words2.size === 0) return 1;
    if (words1.size === 0 || words2.size === 0) return 0;

    const intersection = new Set([...words1].filter(w => words2.has(w)));
    const union = new Set([...words1, ...words2]);

    return intersection.size / union.size;
}

/**
 * Check if a new memory is a duplicate of any existing memory
 * Uses 80% similarity threshold and never deduplicates cluster memories
 */
function isDuplicateMemory(newMemory, existingMemories) {
    if (!newMemory || !newMemory.content) return false;

    // Never deduplicate cluster memories - they are unique consolidations
    if (newMemory.memory_type === 'compressed_cluster') {
        return false;
    }

    // Normalize new memory content
    const normalizedNew = (newMemory.content || '').toLowerCase()
        .replace(/# session summary.*?\n/gi, '')
        .replace(/\*\*date\*\*:.*?\n/gi, '')
        .replace(/\*\*project\*\*:.*?\n/gi, '')
        .replace(/\s+/g, ' ')
        .trim();

    // Skip if content is too short
    if (normalizedNew.length < 20) return false;

    // Check similarity against all existing memories
    for (const existing of existingMemories) {
        if (!existing || !existing.content) continue;

        // Normalize existing memory content
        const normalizedExisting = (existing.content || '').toLowerCase()
            .replace(/# session summary.*?\n/gi, '')
            .replace(/\*\*date\*\*:.*?\n/gi, '')
            .replace(/\*\*project\*\*:.*?\n/gi, '')
            .replace(/\s+/g, ' ')
            .trim();

        // Calculate similarity
        const similarity = calculateContentSimilarity(normalizedNew, normalizedExisting);

        // 80% similarity threshold
        if (similarity > 0.8) {
            return true;
        }
    }

    return false;
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
    // Config specifies 10s, we use 9.5s to leave 0.5s buffer for cleanup
    // With 1 git query + 1 recent query, expect ~9.5s total (4.5s each due to Python cold-start)
    const HOOK_TIMEOUT = 9500; // 9.5 seconds (reduced Phase 0 from 2 to 1 query)
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

        // Check for user overrides (#skip / #remember)
        const overrides = detectUserOverrides(context.userMessage);
        if (overrides.forceSkip) {
            logOverride('skip');
            return;
        }
        // Note: forceRemember for session-start could force retrieval even without context shift
        // Currently we just log and continue - could be enhanced later
        if (overrides.forceRemember && verbose && !cleanMode) {
            console.log(`${CONSOLE_COLORS.CYAN}💾 Memory Hook${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Force retrieval requested (#remember)`);
        }

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
            const projectDisplay = `${CONSOLE_COLORS.BRIGHT}${projectContext.name}${CONSOLE_COLORS.RESET}`;
            const typeDisplay = projectContext.language !== 'Unknown' ? ` ${CONSOLE_COLORS.GRAY}(${projectContext.language})${CONSOLE_COLORS.RESET}` : '';
            console.log(`${CONSOLE_COLORS.BLUE}📂 Project Detector${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Analyzing ${projectDisplay}${typeDisplay}`);
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

        // Display version information
        const showVersionInfo = config.versionCheck?.enabled !== false; // Default to true
        if (showVersionInfo && verbose && !cleanMode) {
            try {
                const versionInfo = await getVersionInfo(context.workingDirectory || process.cwd(), {
                    checkPyPI: config.versionCheck?.checkPyPI !== false,
                    timeout: config.versionCheck?.timeout || 2000
                });

                const versionDisplay = formatVersionDisplay(versionInfo, CONSOLE_COLORS);
                console.log(versionDisplay);
            } catch (error) {
                // Silently fail - version check is informational, not critical
                if (verbose && showMemoryDetails) {
                    console.warn(`[Memory Hook] Version check failed: ${error.message}`);
                }
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

        // Extract memory scoring configuration
        const scoringWeights = config.memoryScoring?.weights || {};
        const timeDecayRate = config.memoryScoring?.timeDecayRate || 0.1;
        const enableConversationContext = config.memoryScoring?.enableConversationContext || false;
        const minRelevanceScore = config.memoryScoring?.minRelevanceScore || 0.3;
        const showPhaseDetails = config.output?.showPhaseDetails !== false && config.output?.style !== 'balanced'; // Hide in balanced mode

        if (recentFirstMode) {
            // Phase 0: Git Context Phase (NEW - highest priority for repository-aware memories)
            if (gitContext && gitContext.developmentKeywords.keywords.length > 0) {
                const maxGitMemories = config.gitAnalysis?.maxGitMemories || 3;
                const gitQueries = buildGitContextQuery(projectContext, gitContext.developmentKeywords, context.userMessage);

                if (verbose && showPhaseDetails && !cleanMode && gitQueries.length > 0) {
                    console.log(`${CONSOLE_COLORS.GREEN}⚡ Phase 0${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Git-aware memory search (${maxGitMemories} slots, 1 of ${gitQueries.length} queries for 8s timeout)`);
                }
                
                // Execute git-context queries
                for (const gitQuery of gitQueries.slice(0, 1)) { // Limit to top 1 query to stay within 8s timeout
                    if (allMemories.length >= maxGitMemories) break;

                    const gitMemories = await queryMemoryService(memoryClient, {
                        semanticQuery: gitQuery.semanticQuery,
                        limit: Math.min(maxGitMemories - allMemories.length, 3),
                        timeFilter: 'last-2-weeks' // Focus on recent memories for git context
                    }, config);
                    
                    if (gitMemories && gitMemories.length > 0) {
                        // Mark these memories as git-context derived for scoring
                        const markedMemories = gitMemories.map(mem => ({
                            ...mem,
                            _gitContextType: gitQuery.type,
                            _gitContextSource: gitQuery.source,
                            _gitContextWeight: config.gitAnalysis?.gitContextWeight || 1.2
                        }));
                        
                        // Filter out duplicates using similarity-based deduplication (80% threshold)
                        const newGitMemories = markedMemories.filter(newMem =>
                            !isDuplicateMemory(newMem, allMemories)
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

                const recentMemories = await queryMemoryService(memoryClient, recentQuery, config);

                // Filter out duplicates using similarity-based deduplication (80% threshold)
                if (recentMemories && recentMemories.length > 0) {
                    const newRecentMemories = recentMemories.filter(newMem =>
                        !isDuplicateMemory(newMem, allMemories)
                    );

                    allMemories.push(...newRecentMemories);
                }
            }
            
            // Phase 2: Important tagged memories - fill remaining slots
            const remainingSlots = maxMemories - allMemories.length;
            if (remainingSlots > 0) {
                // Build tag list for important memories
                const importantTags = [
                    projectContext.name,
                    'key-decisions',
                    'architecture',
                    'claude-code-reference'
                ].filter(Boolean);

                const timeFilter = 'last-2-weeks';

                if (verbose && showMemoryDetails && showPhaseDetails && !cleanMode) {
                    console.log(`${CONSOLE_COLORS.BLUE}🎯 Phase 2${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Searching important tagged memories (${remainingSlots} slots)`);
                }

                // Use new tag-time filtering method for efficient recency prioritization
                const importantMemories = memoryClient ?
                    await memoryClient.queryMemoriesByTagsAndTime(importantTags, timeFilter, remainingSlots, false) :
                    [];

                // Filter out duplicates using similarity-based deduplication (80% threshold)
                const newMemories = (importantMemories || []).filter(newMem =>
                    !isDuplicateMemory(newMem, allMemories)
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

                const fallbackMemories = await queryMemoryService(memoryClient, fallbackQuery, config);

                // Filter out duplicates using similarity-based deduplication (80% threshold)
                const newFallbackMemories = (fallbackMemories || []).filter(newMem =>
                    !isDuplicateMemory(newMem, allMemories)
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
            
            const legacyMemories = await queryMemoryService(memoryClient, memoryQuery, config);

            allMemories.push(...(legacyMemories || []));
        }

        // Query specifically for consolidated cluster memories (always load 2-3 most recent)
        // With regular consolidation (weekly/monthly), clusters should be < 2 weeks old
        // last-month (30 days) provides buffer for irregular consolidation schedules
        if (memoryClient) {
            // Use tag-based search instead of semantic search for reliable cluster detection
            // All consolidated clusters have the 'cluster' tag
            const clusterMemories = await memoryClient.queryMemoriesByTagsAndTime(
                ['cluster'],           // Tag-based filter (all clusters have this tag)
                'last-month',         // Time window: sufficient for regularly created clusters
                3                     // Limit: load 2-3 most recent clusters
            );

            // Filter to only keep compressed_cluster types and add to allMemories
            const validClusters = (clusterMemories || []).filter(m => m.memory_type === 'compressed_cluster');

            // Explicit logging about cluster availability
            if (validClusters.length > 0 && verbose && showMemoryDetails && !cleanMode) {
                console.log(`${CONSOLE_COLORS.MAGENTA}📦 Cluster Search${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} Found ${CONSOLE_COLORS.BRIGHT}${validClusters.length}${CONSOLE_COLORS.RESET} consolidated memories`);
            } else if (validClusters.length === 0 && verbose && showMemoryDetails && !cleanMode) {
                console.log(`${CONSOLE_COLORS.GRAY}📦 Cluster Search${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}No clusters found (run consolidation to create)${CONSOLE_COLORS.RESET}`);
            }

            allMemories.push(...validClusters);
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
            
            // Analyze memory age distribution for adaptive weight adjustment
            const ageAnalysis = analyzeMemoryAgeDistribution(memories, { verbose: showMemoryDetails && !cleanMode });

            // Apply auto-calibration if enabled
            const autoCalibrate = config.memoryScoring?.autoCalibrate !== false; // Default true
            let adjustedWeights = { ...scoringWeights };

            if (autoCalibrate && ageAnalysis.isStale && ageAnalysis.recommendedAdjustments.timeDecay) {
                adjustedWeights = {
                    ...adjustedWeights,
                    timeDecay: ageAnalysis.recommendedAdjustments.timeDecay,
                    tagRelevance: ageAnalysis.recommendedAdjustments.tagRelevance
                };

                if (verbose && showMemoryDetails && !cleanMode) {
                    console.log(`${CONSOLE_COLORS.CYAN}🎯 Auto-Calibration${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}${ageAnalysis.recommendedAdjustments.reason}${CONSOLE_COLORS.RESET}`);
                    console.log(`${CONSOLE_COLORS.CYAN}   Adjusted Weights${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} timeDecay: ${adjustedWeights.timeDecay.toFixed(2)}, tagRelevance: ${adjustedWeights.tagRelevance.toFixed(2)}`);
                }
            }

            // Score memories for relevance (with enhanced recency weighting and auto-calibrated weights)
            let scoredMemories = scoreMemoryRelevance(memories, projectContext, {
                verbose: showMemoryDetails,
                enhanceRecency: recentFirstMode,
                weights: adjustedWeights,
                timeDecayRate: timeDecayRate,
                includeConversationContext: enableConversationContext
            });

            // Calculate adaptive git context weight
            // v8.5.1+ Dynamic git weight based on memory age and commit activity
            const configuredGitWeight = config.gitAnalysis?.gitContextWeight || 1.2;
            const adaptiveGitEnabled = config.gitAnalysis?.adaptiveGitWeight !== false; // Default true

            let gitWeightResult;
            if (adaptiveGitEnabled && gitContext) {
                gitWeightResult = calculateAdaptiveGitWeight(
                    gitContext,
                    ageAnalysis,
                    configuredGitWeight,
                    { verbose: showMemoryDetails && !cleanMode }
                );
            } else {
                gitWeightResult = { weight: configuredGitWeight, reason: 'Adaptive git weight disabled', adjusted: false };
            }

            const gitWeight = gitWeightResult.weight;

            // Show git weight info
            if (verbose && showMemoryDetails && !cleanMode) {
                if (gitWeightResult.adjusted) {
                    console.log(`${CONSOLE_COLORS.CYAN}⚙️  Adaptive Git Weight${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}${gitWeightResult.reason}${CONSOLE_COLORS.RESET}`);
                }
                if (configuredGitWeight > 1.5 && !gitWeightResult.adjusted) {
                    console.log(`${CONSOLE_COLORS.YELLOW}⚠️  Git Weight${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}High git context weight (${gitWeight.toFixed(1)}x) may prioritize git-related memories excessively${CONSOLE_COLORS.RESET}`);
                    console.log(`${CONSOLE_COLORS.YELLOW}   Recommended${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}1.1-1.3x for balanced recency${CONSOLE_COLORS.RESET}`);
                }
            }

            // Apply git context weight boost to git-derived memories

            scoredMemories = scoredMemories.map(memory => {
                if (memory._gitContextWeight && memory._gitContextWeight !== 1.0) {
                    const originalScore = memory.relevanceScore;
                    const boostedScore = Math.min(1.0, originalScore * memory._gitContextWeight);

                    // Store original score for transparency
                    return {
                        ...memory,
                        _originalScore: originalScore,
                        relevanceScore: boostedScore,
                        _wasBoosted: true
                    };
                }
                return memory;
            }).sort((a, b) => b.relevanceScore - a.relevanceScore); // Re-sort after boost

            // Filter out zero-scored memories (project affinity filtered)
            const preFilterCount = scoredMemories.length;
            scoredMemories = scoredMemories.filter(m => m.relevanceScore > 0);
            if (verbose && showMemoryDetails && !cleanMode && preFilterCount !== scoredMemories.length) {
                console.log(`[Memory Filter] Removed ${preFilterCount - scoredMemories.length} unrelated memories (no project affinity)`);
            }

            // Show top scoring memories with recency info and detailed breakdown
            if (verbose && showMemoryDetails && scoredMemories.length > 0 && !cleanMode) {
                const topMemories = scoredMemories.slice(0, 3);
                const memoryInfo = topMemories.map((m, idx) => {
                    const score = `${(m.relevanceScore * 100).toFixed(0)}%`;
                    let recencyFlag = '';
                    let ageText = '';
                    if (m.created_at_iso) {
                        const daysDiff = (now - new Date(m.created_at_iso)) / (1000 * 60 * 60 * 24);
                        if (daysDiff <= 1) {
                            recencyFlag = '🕒';
                            ageText = 'today';
                        } else if (daysDiff <= 7) {
                            recencyFlag = '📅';
                            ageText = `${Math.floor(daysDiff)}d ago`;
                        } else if (daysDiff <= 30) {
                            ageText = `${Math.floor(daysDiff)}d ago`;
                        } else {
                            ageText = `${Math.floor(daysDiff)}d ago`;
                        }
                    }

                    // Show detailed breakdown for top memory (only if explicitly enabled)
                    if (idx === 0 && m.scoreBreakdown) {
                        const bd = m.scoreBreakdown;
                        const showBreakdown = config.output?.showScoringBreakdown === true;

                        if (showBreakdown) {
                            console.log(`${CONSOLE_COLORS.CYAN}  📊 Top Memory Breakdown${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.DIM}→${CONSOLE_COLORS.RESET}`);
                            console.log(`${CONSOLE_COLORS.CYAN}    • Time Decay${CONSOLE_COLORS.RESET}: ${(bd.timeDecay * 100).toFixed(0)}% ${CONSOLE_COLORS.GRAY}(${ageText})${CONSOLE_COLORS.RESET}`);
                            console.log(`${CONSOLE_COLORS.CYAN}    • Tag Match${CONSOLE_COLORS.RESET}: ${(bd.tagRelevance * 100).toFixed(0)}%`);
                            console.log(`${CONSOLE_COLORS.CYAN}    • Content${CONSOLE_COLORS.RESET}: ${(bd.contentRelevance * 100).toFixed(0)}%`);
                            console.log(`${CONSOLE_COLORS.CYAN}    • Quality${CONSOLE_COLORS.RESET}: ${(bd.contentQuality * 100).toFixed(0)}%`);
                            if (bd.recencyBonus > 0) {
                                console.log(`${CONSOLE_COLORS.CYAN}    • Recency Bonus${CONSOLE_COLORS.RESET}: ${CONSOLE_COLORS.GREEN}+${(bd.recencyBonus * 100).toFixed(0)}%${CONSOLE_COLORS.RESET}`);
                            }
                            // Show git context boost if applied
                            if (m._wasBoosted && m._originalScore) {
                                const boostAmount = ((m.relevanceScore - m._originalScore) * 100).toFixed(0);
                                console.log(`${CONSOLE_COLORS.CYAN}    • Git Boost${CONSOLE_COLORS.RESET}: ${CONSOLE_COLORS.YELLOW}+${boostAmount}%${CONSOLE_COLORS.RESET} ${CONSOLE_COLORS.GRAY}(${(m._originalScore * 100).toFixed(0)}% → ${(m.relevanceScore * 100).toFixed(0)}%)${CONSOLE_COLORS.RESET}`);
                            }
                        } else if (config.logging?.enableDebug) {
                            // Log to debug file instead of console
                            const debugMsg = `[Memory Scorer] Top memory breakdown: TimeDecay=${(bd.timeDecay * 100).toFixed(0)}%, TagMatch=${(bd.tagRelevance * 100).toFixed(0)}%, Content=${(bd.contentRelevance * 100).toFixed(0)}%, Quality=${(bd.contentQuality * 100).toFixed(0)}%, RecencyBonus=${(bd.recencyBonus * 100).toFixed(0)}%`;
                            console.log(debugMsg);
                        }
                    }

                    return ageText ? `${score}${recencyFlag} (${ageText})` : `${score}${recencyFlag}`;
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
                storageInfo: showStorageSource ? (storageInfo || detectStorageBackend(config)) : null,
                adaptiveTruncation: config.output?.adaptiveTruncation !== false,
                contentLengthConfig: config.contentLength
            });
            
            // Inject context into session
            if (context.injectSystemMessage) {
                await context.injectSystemMessage(contextMessage);
                // Note: Don't console.log here - injectSystemMessage handles display
                // console.log would cause duplicate output in Claude Code


                // Write detailed session context log file (Option 3)
                try {
                    const os = require('os');
                    const logPath = path.join(os.homedir(), '.claude', 'last-session-context.txt');
                    const recencyPercent = memories.length > 0 ? ((recentCount / memories.length) * 100).toFixed(0) : 0;

                    let logContent = `Session Started: ${new Date().toISOString()}\n`;
                    logContent += `Session ID: ${context.sessionId || 'unknown'}\n\n`;
                    logContent += `=== Project Context ===\n`;
                    logContent += `Project: ${projectContext.name}\n`;
                    logContent += `Language: ${projectContext.language}\n`;
                    if (projectContext.frameworks && projectContext.frameworks.length > 0) {
                        logContent += `Frameworks: ${projectContext.frameworks.join(', ')}\n`;
                    }
                    if (projectContext.git) {
                        logContent += `Git Branch: ${projectContext.git.branch || 'unknown'}\n`;
                    }
                    logContent += `\n=== Storage Backend ===\n`;
                    if (storageInfo) {
                        logContent += `Backend: ${storageInfo.backend}\n`;
                        logContent += `Type: ${storageInfo.type}\n`;
                        logContent += `Location: ${storageInfo.location}\n`;
                        if (storageInfo.health.totalMemories > 0) {
                            logContent += `Total Memories in DB: ${storageInfo.health.totalMemories}\n`;
                        }
                    }
                    logContent += `\n=== Memory Statistics ===\n`;
                    logContent += `Memories Loaded: ${memories.length}\n`;
                    logContent += `Recent (last week): ${recentCount} (${recencyPercent}%)\n`;

                    if (gitContext) {
                        logContent += `\n=== Git Context ===\n`;
                        logContent += `Commits Analyzed: ${gitContext.commits.length}\n`;
                        logContent += `Changelog Entries: ${gitContext.changelogEntries?.length || 0}\n`;
                        logContent += `Top Keywords: ${gitContext.developmentKeywords.keywords.slice(0, 5).join(', ')}\n`;
                    }

                    if (topMemories.length > 0) {
                        logContent += `\n=== Top Loaded Memories ===\n`;
                        topMemories.slice(0, 3).forEach((m, idx) => {
                            const preview = m.content ? m.content.substring(0, 150).replace(/\n/g, ' ') : 'No content';
                            const ageInfo = m.created_at_iso ? ` (${Math.floor((now - new Date(m.created_at_iso)) / (1000 * 60 * 60 * 24))}d ago)` : '';
                            logContent += `\n${idx + 1}. Score: ${(m.relevanceScore * 100).toFixed(0)}%${ageInfo}\n`;
                            logContent += `   ${preview}...\n`;
                        });
                    }

                    await fs.writeFile(logPath, logContent, 'utf8');
                } catch (error) {
                    // Silently fail - log file is nice-to-have, not critical
                    if (verbose && showMemoryDetails) {
                        console.warn(`[Memory Hook] Failed to write log file: ${error.message}`);
                    }
                }

                // Write status line cache file (Option 4)
                try {
                    const cachePath = path.join(__dirname, '../utilities/session-cache.json');
                    const cacheData = {
                        timestamp: new Date().toISOString(),
                        sessionId: context.sessionId || 'unknown',
                        project: projectContext.name,
                        memoriesLoaded: memories.length,  // Use actual loaded count after deduplication
                        recentCount: recentCount,
                        gitCommits: gitContext ? gitContext.commits.length : 0,
                        gitKeywords: gitContext ? gitContext.developmentKeywords.keywords.slice(0, 3) : [],
                        storageBackend: storageInfo ? storageInfo.backend : 'unknown'
                    };

                    await fs.writeFile(cachePath, JSON.stringify(cacheData, null, 2), 'utf8');
                } catch (error) {
                    // Silently fail - status line cache is optional
                    if (verbose && showMemoryDetails) {
                        console.warn(`[Memory Hook] Failed to write status line cache: ${error.message}`);
                    }
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
            // Just print the message - it already has its own formatting from context-formatter.js
            console.log(message);
        }
    };
    
    onSessionStart(mockContext)
        .then(() => {
            // Test completed quietly
            process.exit(0);
        })
        .catch(error => {
            console.error(`${CONSOLE_COLORS.RED}❌ Hook test failed:${CONSOLE_COLORS.RESET} ${error.message}`);
            process.exit(1);
        });
}