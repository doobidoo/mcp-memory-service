#!/usr/bin/env node
/**
 * ensure-server.js — SessionStart hook that ensures the HTTP memory server
 * is reachable, starting it in the background if necessary.
 *
 * Contract: NEVER block session start. All failure paths exit 0 with stderr.
 */
'use strict';

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

const HEALTH_TIMEOUT_MS = 500;
const POLL_INTERVAL_MS = 500;
const POLL_TIMEOUT_MS = 10_000;
const NO_SPAWN = process.env.ENSURE_SERVER_NO_SPAWN === '1';

function log(msg) {
    process.stderr.write(`[ensure-server] ${msg}\n`);
}

function resolveEndpoint() {
    if (process.env.MCP_MEMORY_ENDPOINT) {
        return process.env.MCP_MEMORY_ENDPOINT;
    }
    const configPath = path.join(os.homedir(), '.claude', 'hooks', 'config.json');
    try {
        const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
        const endpoint =
            cfg.memoryService?.http?.endpoint ||
            cfg.memoryService?.endpoint ||
            cfg.sessionHarvest?.endpoint;
        if (endpoint) return endpoint;
    } catch (_) {
        // fall through to default
    }
    return 'http://127.0.0.1:8000';
}

function checkHealth(endpoint) {
    return new Promise((resolve) => {
        let url;
        try {
            url = new URL('/api/health', endpoint);
        } catch (_) {
            return resolve(false);
        }
        const lib = url.protocol === 'https:' ? https : http;
        const req = lib.get(url, { timeout: HEALTH_TIMEOUT_MS }, (res) => {
            res.resume();
            resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.on('timeout', () => {
            req.destroy();
            resolve(false);
        });
    });
}

function resolveLogPath() {
    const preferred = path.join(os.homedir(), '.mcp-memory-service', 'http.log');
    try {
        fs.mkdirSync(path.dirname(preferred), { recursive: true });
        fs.accessSync(path.dirname(preferred), fs.constants.W_OK);
        return preferred;
    } catch (_) {
        return path.join(os.tmpdir(), 'mcp-memory-service-http.log');
    }
}

function spawnServer() {
    const logPath = resolveLogPath();
    let logFd;
    try {
        logFd = fs.openSync(logPath, 'a');
    } catch (err) {
        log(`could not open log file at ${logPath}: ${err.message}`);
        return false;
    }

    try {
        // Strategy 1: memory CLI entry point (installed by pip install mcp-memory-service)
        try {
            const child = spawn('memory', ['server', '--http'], {
                detached: true,
                stdio: ['ignore', logFd, logFd],
                env: process.env,
            });
            child.unref();
            log(`spawned HTTP server via 'memory server --http' (log: ${logPath})`);
            return true;
        } catch (err) {
            log(`'memory' CLI unavailable: ${err.message}`);
        }

        // Strategy 2 & 3: python-based fallbacks
        const pythonCandidates = [
            process.env.MCP_MEMORY_PYTHON,
            path.join(__dirname, '..', '..', '.venv', 'bin', 'python'),
            'python3',
            'python',
        ].filter(Boolean);

        const pythonInvocations = [
            ['-m', 'mcp_memory_service.cli.main', 'server', '--http'],
            [path.join(__dirname, '..', '..', 'scripts', 'server', 'run_http_server.py')],
        ];

        for (const python of pythonCandidates) {
            for (const args of pythonInvocations) {
                try {
                    const child = spawn(python, args, {
                        detached: true,
                        stdio: ['ignore', logFd, logFd],
                        env: process.env,
                    });
                    child.unref();
                    log(`spawned HTTP server via ${python} ${args.join(' ')} (log: ${logPath})`);
                    return true;
                } catch (err) {
                    log(`spawn with ${python} ${args.join(' ')} failed: ${err.message}`);
                }
            }
        }

        log('could not spawn HTTP server — install mcp-memory-service or set MCP_MEMORY_PYTHON');
        return false;
    } finally {
        if (typeof logFd === 'number') { try { fs.closeSync(logFd); } catch (_) {} }
    }
}

async function pollUntilHealthy(endpoint, deadlineMs) {
    while (Date.now() < deadlineMs) {
        if (await checkHealth(endpoint)) return true;
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    }
    return false;
}

async function main() {
    const endpoint = resolveEndpoint();
    if (await checkHealth(endpoint)) return;

    log(`HTTP server unreachable at ${endpoint}`);
    if (NO_SPAWN) {
        log('ENSURE_SERVER_NO_SPAWN=1 — skipping spawn (test mode)');
        return;
    }

    if (!spawnServer()) return;

    const deadline = Date.now() + POLL_TIMEOUT_MS;
    const ok = await pollUntilHealthy(endpoint, deadline);
    if (!ok) {
        log(`server did not become healthy within ${POLL_TIMEOUT_MS}ms`);
    } else {
        log('server is healthy');
    }
}

if (require.main === module) {
    main()
        .catch((err) => {
            log(`unexpected error: ${err.message}`);
        })
        .finally(() => {
            process.exit(0); // never block
        });
}
