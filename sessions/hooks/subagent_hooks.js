#!/usr/bin/env node

// ===== IMPORTS ===== //

/// ===== STDLIB ===== ///
const fs = require('fs');
const path = require('path');
///-///

/// ===== 3RD-PARTY ===== ///
///-///

/// ===== LOCAL ===== ///
const { editState, PROJECT_ROOT, loadState } = require('./shared_state.js');
///-///

//-//

// ===== FUNCTIONS ===== //

function findCurrentTranscript(transcriptPath, sessionId, staleThreshold = 30) {
    /**
     * Detect stale transcripts and find the current one by session ID.
     *
     * @param {string} transcriptPath - Path to the transcript file we received
     * @param {string} sessionId - Current session ID to match
     * @param {number} staleThreshold - Seconds threshold for considering transcript stale
     * @returns {string} Path to the current transcript (may be same as input if not stale)
     */
    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
        return transcriptPath;
    }

    try {
        // Read last line of transcript to get last message timestamp
        const lines = fs.readFileSync(transcriptPath, 'utf-8').split('\n').filter(line => line.trim());
        if (!lines.length) {
            return transcriptPath;
        }

        const lastLine = lines[lines.length - 1];
        const lastMsg = JSON.parse(lastLine);
        const lastTimestamp = lastMsg.timestamp;

        if (!lastTimestamp) {
            return transcriptPath;
        }

        // Parse ISO timestamp and compare to current time
        const lastTime = new Date(lastTimestamp);
        const currentTime = new Date();
        const ageSeconds = (currentTime - lastTime) / 1000;

        // If transcript is fresh, return it
        if (ageSeconds <= staleThreshold) {
            return transcriptPath;
        }

        // Transcript is stale - search for current one
        const transcriptDir = path.dirname(transcriptPath);
        const allFiles = fs.readdirSync(transcriptDir)
            .filter(f => f.endsWith('.jsonl'))
            .map(f => path.join(transcriptDir, f))
            .sort((a, b) => fs.statSync(b).mtime - fs.statSync(a).mtime)
            .slice(0, 5);  // Top 5 most recent

        // Check each transcript for matching session ID
        for (const candidate of allFiles) {
            try {
                const candidateLines = fs.readFileSync(candidate, 'utf-8').split('\n').filter(line => line.trim());
                if (!candidateLines.length) {
                    continue;
                }

                // Check last line for session ID
                const candidateLast = JSON.parse(candidateLines[candidateLines.length - 1]);
                const candidateSessionId = candidateLast.sessionId;

                if (candidateSessionId === sessionId) {
                    // Verify this transcript is fresh
                    const candidateTimestamp = candidateLast.timestamp;
                    if (candidateTimestamp) {
                        const candidateTime = new Date(candidateTimestamp);
                        const candidateAge = (currentTime - candidateTime) / 1000;

                        if (candidateAge <= staleThreshold) {
                            return candidate;
                        }
                    }
                }
            } catch {
                continue;
            }
        }

        // No fresh transcript found, return original
        return transcriptPath;
    } catch {
        // Any error, return original path
        return transcriptPath;
    }
}

//-//

// ===== GLOBALS ===== //

/// ===== CI DETECTION ===== ///
function isCIEnvironment() {
    // Check if running in a CI environment (GitHub Actions)
    const ciIndicators = [
        'GITHUB_ACTIONS',         // GitHub Actions
        'GITHUB_WORKFLOW',        // GitHub Actions workflow
        'CI',                     // Generic CI indicator (set by GitHub Actions)
        'CONTINUOUS_INTEGRATION', // Generic CI (alternative)
    ];
    return ciIndicators.some(indicator => process.env[indicator]);
}

// Skip subagent hooks in CI environments
if (isCIEnvironment()) {
    process.exit(0);
}
///-///

// Load input from stdin
let inputData = {};
try {
    const stdin = fs.readFileSync(0, 'utf-8');
    inputData = JSON.parse(stdin);
} catch (e) {
    console.error(`Error: Invalid JSON input: ${e.message}`);
    process.exit(1);
}

// Check if this is a Task tool call
const toolName = inputData.tool_name || "";
if (toolName !== "Task") {
    process.exit(0);
}

// Get the transcript path and session ID from the input data
let transcriptPath = inputData.transcript_path || "";
const sessionId = inputData.session_id || "";
if (!transcriptPath) {
    process.exit(0);
}

// Detect and recover from stale transcript
if (transcriptPath) {
    transcriptPath = findCurrentTranscript(transcriptPath, sessionId);
}

// Get the transcript into memory
let transcript = [];
try {
    const lines = fs.readFileSync(transcriptPath, 'utf-8').split('\n');
    for (const line of lines) {
        if (line.trim()) {
            transcript.push(JSON.parse(line));
        }
    }
} catch (e) {
    process.exit(0);
}
//-//

/*
╔═══════════════════════════════════════════════════════════════════╗
║ ██████╗██╗ ██╗█████╗  █████╗  █████╗██████╗██╗  ██╗██████╗██████╗ ║
║ ██╔═══╝██║ ██║██╔═██╗██╔══██╗██╔═══╝██╔═══╝███╗ ██║╚═██╔═╝██╔═══╝ ║
║ ██████╗██║ ██║█████╔╝███████║██║    █████╗ ████╗██║  ██║  ██████╗ ║
║ ╚═══██║██║ ██║██╔═██╗██╔══██║██║ ██╗██╔══╝ ██╔████║  ██║  ╚═══██║ ║
║ ██████║╚████╔╝█████╔╝██║  ██║╚█████║██████╗██║╚███║  ██║  ██████║ ║
║ ╚═════╝ ╚═══╝ ╚════╝ ╚═╝  ╚═╝ ╚════╝╚═════╝╚═╝ ╚══╝  ╚═╝  ╚═════╝ ║
╚═══════════════════════════════════════════════════════════════════╝
PreToolUse:Task:subagent_type hooks

This module handles PreToolUse processing for the Task tool:
    - Chunks the transcript for subagents based on token limits
    - Saves transcript chunks to designated directories
    - Sets flags to manage subagent context
*/

// ===== EXECUTION ===== //

//!> Set subagent flag
editState(s => {
    s.flags.subagent = true;
});
const STATE = loadState();
//!<

//!> Trunc + clean transcript
// Remove any pre-work transcript entries
let startFound = false;
let transcriptQueue = [...transcript];
while (!startFound && transcriptQueue.length > 0) {
    const entry = transcriptQueue.shift();
    const message = entry.message;
    if (message) {
        const content = message.content;
        if (Array.isArray(content)) {
            for (const block of content) {
                if (block.type === 'tool_use' && ['Edit', 'MultiEdit', 'Write'].includes(block.name)) {
                    startFound = true;
                    break;
                }
            }
        }
    }
}

// Clean the transcript
const cleanTranscript = [];
for (const entry of transcriptQueue) {
    const message = entry.message;
    const messageType = entry.type;

    if (message && ['user', 'assistant'].includes(messageType)) {
        const content = message.content;
        const role = message.role;
        cleanTranscript.push({ role: role, content: content });
    }
}
//!<

//!> Prepare subagent dir for transcript files
let subagentType = 'shared';
if (cleanTranscript.length === 0) {
    console.log("[Subagent] No relevant transcript entries found, skipping snapshot.");
    process.exit(0);
}

const taskCall = cleanTranscript[cleanTranscript.length - 1];
const content = taskCall.content;
if (Array.isArray(content)) {
    for (const block of content) {
        if (block.type === 'tool_use' && block.name === 'Task') {
            const taskInput = block.input || {};
            subagentType = taskInput.subagent_type || subagentType;
        }
    }
}

// Clear the current transcript directory
const BATCH_DIR = path.join(PROJECT_ROOT, 'sessions', 'transcripts', subagentType);
if (!fs.existsSync(BATCH_DIR)) {
    fs.mkdirSync(BATCH_DIR, { recursive: true });
}
// Remove existing files
const existingFiles = fs.readdirSync(BATCH_DIR);
for (const file of existingFiles) {
    const filePath = path.join(BATCH_DIR, file);
    if (fs.statSync(filePath).isFile()) {
        fs.unlinkSync(filePath);
    }
}
//!<

//!> Chunk and save transcript batches
const MAX_BYTES = 24000;
let usableContext = 160000;
if (STATE.model === "sonnet") {
    usableContext = 800000;
}

const cleanTranscriptText = JSON.stringify(cleanTranscript, null, 2);

const chunks = [];
let bufChars = [];
let bufBytes = 0;
let lastNewlineIdx = null;
let lastSpaceIdx = null;

// Convert string to byte array for accurate byte counting
const encoder = new TextEncoder();

for (let i = 0; i < cleanTranscriptText.length; i++) {
    const ch = cleanTranscriptText[i];
    const chBytes = encoder.encode(ch).length;

    // If overflowing, flush a chunk
    if (bufBytes + chBytes > MAX_BYTES) {
        let cutIdx = null;
        if (lastNewlineIdx !== null) {
            cutIdx = lastNewlineIdx;
        } else if (lastSpaceIdx !== null) {
            cutIdx = lastSpaceIdx;
        }

        if (cutIdx !== null && cutIdx > 0) {
            // Emit chunk up to the breakpoint
            chunks.push(bufChars.slice(0, cutIdx).join(''));
            const remainder = bufChars.slice(cutIdx);
            bufChars = remainder;
            bufBytes = encoder.encode(bufChars.join('')).length;
        } else {
            // No breakpoints, hard cut what we got
            if (bufChars.length > 0) {
                chunks.push(bufChars.join(''));
            }
            bufChars = [];
            bufBytes = 0;
        }

        lastNewlineIdx = null;
        lastSpaceIdx = null;
    }

    bufChars.push(ch);
    bufBytes += chBytes;

    if (ch === '\n') {
        lastNewlineIdx = bufChars.length;
        lastSpaceIdx = null;
    } else if (ch === ' ' && lastNewlineIdx === null) {
        lastSpaceIdx = bufChars.length;
    }
}

// Flush any remaining buffer
if (bufChars.length > 0) {
    chunks.push(bufChars.join(''));
}

// Verify all chunks meet byte limit
for (const chunk of chunks) {
    const byteLength = encoder.encode(chunk).length;
    if (byteLength > MAX_BYTES) {
        console.error("Chunking failed to enforce byte limit");
        process.exit(1);
    }
}

// Save chunks to files
chunks.forEach((chunk, idx) => {
    const partName = `current_transcript_${String(idx + 1).padStart(3, '0')}.txt`;
    const partPath = path.join(BATCH_DIR, partName);
    fs.writeFileSync(partPath, chunk, 'utf8');
});
//!<

//-//

// Allow the tool call to proceed
process.exit(0);