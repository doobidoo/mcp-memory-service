let eventSource = null;
let eventCount = 0;
let connectionStart = null;
let eventStats = {
    memory_stored: 0,
    memory_deleted: 0,
    search_completed: 0,
    heartbeat: 0
};

function updateStatus(connected) {
    const statusEl = document.getElementById('status');
    const connectBtn = document.getElementById('connectBtn');
    const disconnectBtn = document.getElementById('disconnectBtn');
    const statsStatus = document.getElementById('statsStatus');

    if (connected) {
        statusEl.className = 'status connected';
        statusEl.innerHTML = '✅ Connected to SSE stream';
        connectBtn.disabled = true;
        disconnectBtn.disabled = false;
        statsStatus.textContent = 'Connected';
        connectionStart = new Date();
    } else {
        statusEl.className = 'status disconnected';
        statusEl.innerHTML = '❌ Disconnected from SSE stream';
        connectBtn.disabled = false;
        disconnectBtn.disabled = true;
        statsStatus.textContent = 'Disconnected';
        connectionStart = null;
    }
}

function addLogEntry(message, type = 'info') {
    const log = document.getElementById('eventsLog');
    const entry = document.createElement('div');
    entry.className = `event-entry event-${type}`;
    entry.innerHTML = `<span style="color: #666;">${new Date().toLocaleTimeString()}</span> ${message}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
}

function updateStats() {
    document.getElementById('statsEvents').textContent = eventCount;
    document.getElementById('statMemoryStored').textContent = eventStats.memory_stored;
    document.getElementById('statMemoryDeleted').textContent = eventStats.memory_deleted;
    document.getElementById('statSearchCompleted').textContent = eventStats.search_completed;
    document.getElementById('statHeartbeat').textContent = eventStats.heartbeat;

    if (connectionStart) {
        const duration = Math.floor((new Date() - connectionStart) / 1000);
        document.getElementById('statsTime').textContent = `${duration}s`;
    }
}

function connectSSE() {
    if (eventSource) {
        eventSource.close();
    }

    addLogEntry('🔌 Connecting to SSE stream...', 'connection');
    eventSource = new EventSource('../api/events');

    eventSource.onopen = function() {
        updateStatus(true);
        addLogEntry('✅ Connected to SSE stream', 'connection');
    };

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            handleEvent(data, event.type || 'message');
        } catch (error) {
            addLogEntry(`❌ Invalid JSON: ${event.data}`, 'error');
        }
    };

    eventSource.addEventListener('memory_stored', function(event) {
        handleEvent(JSON.parse(event.data), 'memory_stored');
    });

    eventSource.addEventListener('memory_deleted', function(event) {
        handleEvent(JSON.parse(event.data), 'memory_deleted');
    });

    eventSource.addEventListener('search_completed', function(event) {
        handleEvent(JSON.parse(event.data), 'search_completed');
    });

    eventSource.addEventListener('heartbeat', function(event) {
        handleEvent(JSON.parse(event.data), 'heartbeat');
    });

    eventSource.addEventListener('connection_established', function(event) {
        handleEvent(JSON.parse(event.data), 'connection');
    });

    eventSource.onerror = function() {
        updateStatus(false);
        addLogEntry('❌ SSE connection error', 'error');
    };
}

function disconnectSSE() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    updateStatus(false);
    addLogEntry('🔌 Disconnected from SSE stream', 'connection');
}

function handleEvent(data, eventType) {
    eventCount++;
    document.getElementById('statsLastEvent').textContent = eventType;

    if (Object.prototype.hasOwnProperty.call(eventStats, eventType)) {
        eventStats[eventType]++;
    }

    let message = '';
    let logType = 'info';

    switch (eventType) {
        case 'memory_stored': {
            const hash = data.content_hash ? `${data.content_hash.substring(0, 12)}...` : 'unknown';
            const preview = data.content_preview || 'No preview';
            message = `💾 <strong>Memory Stored:</strong> ${hash}<br>&nbsp;&nbsp;&nbsp;&nbsp;${preview}`;
            logType = 'memory';
            break;
        }
        case 'memory_deleted': {
            const hash = data.content_hash ? `${data.content_hash.substring(0, 12)}...` : 'unknown';
            const success = data.success ? '✅' : '❌';
            message = `🗑️ <strong>Memory Deleted:</strong> ${success} ${hash}`;
            logType = 'memory';
            break;
        }
        case 'search_completed': {
            const query = data.query || 'Unknown query';
            const count = data.results_count || 0;
            const time = data.processing_time_ms || 0;
            message = `🔍 <strong>Search:</strong> "${query}" → ${count} results (${time.toFixed(1)}ms)`;
            logType = 'search';
            break;
        }
        case 'heartbeat': {
            const connections = data.active_connections || 0;
            message = `💓 <strong>Heartbeat:</strong> ${connections} active connections`;
            logType = 'heartbeat';
            break;
        }
        case 'connection_established':
            message = `🔌 <strong>Connected:</strong> ${data.message || 'Connection established'}`;
            logType = 'connection';
            break;
        default:
            message = `📨 <strong>${eventType}:</strong> ${JSON.stringify(data)}`;
    }

    addLogEntry(message, logType);
    updateStats();
}

async function testMemoryOperations() {
    addLogEntry('🚀 Starting test memory operations...', 'info');

    try {
        const testMemory = {
            content: `SSE test memory created at ${new Date().toLocaleString()}`,
            tags: ['sse-test', 'browser-test', 'demo'],
            memory_type: 'test',
            metadata: { source: 'sse_test_page' }
        };

        const storeResponse = await fetch('../api/memories', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testMemory)
        });

        if (storeResponse.ok) {
            const result = await storeResponse.json();
            addLogEntry(`✅ Test memory stored: ${result.content_hash?.substring(0, 12)}...`, 'info');

            setTimeout(async () => {
                const searchResponse = await fetch('../api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: 'SSE test memory', n_results: 3 })
                });

                if (searchResponse.ok) {
                    addLogEntry('✅ Test search completed', 'info');

                    setTimeout(async () => {
                        if (result.content_hash) {
                            const deleteResponse = await fetch(`../api/memories/${result.content_hash}`, {
                                method: 'DELETE'
                            });

                            if (deleteResponse.ok) {
                                addLogEntry('✅ Test memory deleted', 'info');
                            }
                        }
                    }, 2000);
                }
            }, 2000);
        }
    } catch (error) {
        addLogEntry(`❌ Test operation failed: ${error.message}`, 'error');
    }
}

function clearLog() {
    document.getElementById('eventsLog').innerHTML = '<div>🚀 SSE Event Log - Log cleared</div>';
    eventCount = 0;
    eventStats = { memory_stored: 0, memory_deleted: 0, search_completed: 0, heartbeat: 0 };
    updateStats();
}

document.querySelector('[data-action="connect"]').addEventListener('click', connectSSE);
document.querySelector('[data-action="disconnect"]').addEventListener('click', disconnectSSE);
document.querySelector('[data-action="test-memory-operations"]').addEventListener('click', testMemoryOperations);
document.querySelector('[data-action="clear-log"]').addEventListener('click', clearLog);
window.addEventListener('beforeunload', () => eventSource?.close());
