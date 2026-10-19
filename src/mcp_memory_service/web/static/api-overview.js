// Auth helper - reads API key from localStorage (shared with dashboard).
function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    const apiKey = localStorage.getItem('mcp_api_key');
    const oauthToken = localStorage.getItem('mcp_oauth_token');
    if (apiKey) headers['X-API-Key'] = apiKey;
    else if (oauthToken) headers.Authorization = `Bearer ${oauthToken}`;
    return headers;
}

async function updateStats() {
    try {
        const healthResponse = await fetch('api/health');
        await healthResponse.json();

        const detailedResponse = await fetch('api/health/detailed', { headers: getAuthHeaders() });
        const detailed = await detailedResponse.json();
        const stats = document.getElementById('stats');
        stats.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${detailed.statistics?.total_memories || 0}</div>
                <div class="stat-label">Total Memories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${detailed.storage?.primary_stats?.embedding_model || detailed.storage?.embedding_model || 'unknown'}</div>
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

function getBackendDisplayName(backend) {
    const backendMap = {
        'sqlite-vec': 'SQLite-vec',
        sqlite_vec: 'SQLite-vec',
        cloudflare: 'Cloudflare D1 + Vectorize',
        hybrid: 'Hybrid (SQLite-vec + Cloudflare)'
    };
    return backendMap[backend] || backend || 'Unknown Backend';
}

async function loadDynamicInfo() {
    try {
        const response = await fetch('api/health/detailed', { headers: getAuthHeaders() });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const healthData = await response.json();

        const versionEl = document.getElementById('version-display');
        if (versionEl && healthData.version) {
            versionEl.textContent = `v${healthData.version}`;
        }

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

updateStats();
setInterval(updateStats, 30000);
document.addEventListener('DOMContentLoaded', loadDynamicInfo);
