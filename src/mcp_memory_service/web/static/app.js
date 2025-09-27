/**
 * MCP Memory Service Dashboard - Main Application
 * Interactive frontend for memory management with real-time updates
 */

class MemoryDashboard {
    constructor() {
        this.apiBase = '/api';
        this.eventSource = null;
        this.memories = [];
        this.currentView = 'dashboard';
        this.searchResults = [];
        this.isLoading = false;

        // Bind methods
        this.handleSearch = this.handleSearch.bind(this);
        this.handleQuickSearch = this.handleQuickSearch.bind(this);
        this.handleNavigation = this.handleNavigation.bind(this);
        this.handleAddMemory = this.handleAddMemory.bind(this);
        this.handleMemoryClick = this.handleMemoryClick.bind(this);

        this.init();
    }

    /**
     * Initialize the application
     */
    async init() {
        this.setupEventListeners();
        this.setupSSE();
        await this.loadDashboardData();
        this.updateConnectionStatus('connected');

        console.log('MCP Memory Dashboard initialized');
    }

    /**
     * Set up event listeners for UI interactions
     */
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', this.handleNavigation);
        });

        // Search functionality
        const quickSearch = document.getElementById('quickSearch');
        const searchBtn = document.querySelector('.search-btn');

        quickSearch.addEventListener('input', this.debounce(this.handleQuickSearch, 300));
        quickSearch.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleSearch(e.target.value);
            }
        });
        searchBtn.addEventListener('click', () => {
            this.handleSearch(quickSearch.value);
        });

        // Add memory functionality
        document.getElementById('addMemoryBtn').addEventListener('click', this.handleAddMemory);
        document.querySelectorAll('[data-action="add-memory"]').forEach(btn => {
            btn.addEventListener('click', this.handleAddMemory);
        });

        // Modal close handlers
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.closeModal(e.target.closest('.modal-overlay'));
            });
        });

        // Modal overlay click to close
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) {
                    this.closeModal(overlay);
                }
            });
        });

        // Add memory form submission
        document.getElementById('saveMemoryBtn').addEventListener('click', this.handleSaveMemory.bind(this));
        document.getElementById('cancelAddBtn').addEventListener('click', () => {
            this.closeModal(document.getElementById('addMemoryModal'));
        });

        // Quick action handlers
        document.querySelectorAll('.action-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleQuickAction(action);
            });
        });

        // Filter handlers for search view
        document.getElementById('tagFilter')?.addEventListener('input', this.handleFilterChange.bind(this));
        document.getElementById('dateFilter')?.addEventListener('change', this.handleFilterChange.bind(this));
        document.getElementById('typeFilter')?.addEventListener('change', this.handleFilterChange.bind(this));

        // View option handlers
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.handleViewModeChange(e.target.dataset.view);
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                document.getElementById('quickSearch').focus();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'm') {
                e.preventDefault();
                this.handleAddMemory();
            }
        });
    }

    /**
     * Set up Server-Sent Events for real-time updates
     */
    setupSSE() {
        try {
            this.eventSource = new EventSource(`${this.apiBase}/events`);

            this.eventSource.onopen = () => {
                console.log('SSE connection established');
                this.updateConnectionStatus('connected');
            };

            this.eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleRealtimeUpdate(data);
                } catch (error) {
                    console.error('Error parsing SSE data:', error);
                }
            };

            this.eventSource.onerror = (error) => {
                console.error('SSE connection error:', error);
                this.updateConnectionStatus('disconnected');

                // Attempt to reconnect after 5 seconds
                setTimeout(() => {
                    if (this.eventSource.readyState === EventSource.CLOSED) {
                        this.setupSSE();
                    }
                }, 5000);
            };

        } catch (error) {
            console.error('Failed to establish SSE connection:', error);
            this.updateConnectionStatus('disconnected');
        }
    }

    /**
     * Handle real-time updates from SSE
     */
    handleRealtimeUpdate(data) {
        switch (data.type) {
            case 'memory_added':
                this.handleMemoryAdded(data.memory);
                this.showToast('Memory added successfully', 'success');
                break;
            case 'memory_deleted':
                this.handleMemoryDeleted(data.memory_id);
                this.showToast('Memory deleted', 'success');
                break;
            case 'memory_updated':
                this.handleMemoryUpdated(data.memory);
                this.showToast('Memory updated', 'success');
                break;
            case 'stats_updated':
                this.updateDashboardStats(data.stats);
                break;
            default:
                console.log('Unknown SSE event type:', data.type);
        }
    }

    /**
     * Load initial dashboard data
     */
    async loadDashboardData() {
        this.setLoading(true);

        try {
            // Load basic statistics
            const statsResponse = await this.apiCall('/health/detailed');
            if (statsResponse.storage_stats) {
                this.updateDashboardStats(statsResponse.storage_stats);
            }

            // Load recent memories
            const memoriesResponse = await this.apiCall('/memories?limit=5&offset=0');
            if (memoriesResponse.memories) {
                this.renderRecentMemories(memoriesResponse.memories);
            }

        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showToast('Failed to load dashboard data', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Handle navigation between views
     */
    handleNavigation(e) {
        const viewName = e.currentTarget.dataset.view;
        this.switchView(viewName);
    }

    /**
     * Switch between different views
     */
    switchView(viewName) {
        // Update navigation active state
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-view="${viewName}"]`).classList.add('active');

        // Hide all views
        document.querySelectorAll('.view-container').forEach(view => {
            view.classList.remove('active');
        });

        // Show target view
        const targetView = document.getElementById(`${viewName}View`);
        if (targetView) {
            targetView.classList.add('active');
            this.currentView = viewName;

            // Load view-specific data
            this.loadViewData(viewName);
        }
    }

    /**
     * Load data specific to the current view
     */
    async loadViewData(viewName) {
        switch (viewName) {
            case 'search':
                // Initialize search view with recent search or empty state
                break;
            case 'browse':
                // Load browseable content
                break;
            case 'manage':
                // Load management tools
                break;
            case 'analytics':
                // Load analytics data
                break;
            default:
                // Dashboard view is loaded in loadDashboardData
                break;
        }
    }

    /**
     * Handle quick search input
     */
    async handleQuickSearch(e) {
        const query = e.target.value.trim();
        if (query.length >= 2) {
            try {
                const results = await this.searchMemories(query);
                // Could show dropdown suggestions here
            } catch (error) {
                console.error('Quick search error:', error);
            }
        }
    }

    /**
     * Handle full search
     */
    async handleSearch(query) {
        if (!query.trim()) return;

        this.switchView('search');
        this.setLoading(true);

        try {
            const results = await this.searchMemories(query);
            this.searchResults = results;
            this.renderSearchResults(results);
            this.updateResultsCount(results.length);
        } catch (error) {
            console.error('Search error:', error);
            this.showToast('Search failed', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Search memories using the API
     */
    async searchMemories(query, filters = {}) {
        const payload = {
            query: query,
            limit: filters.limit || 20,
            threshold: filters.threshold || 0.7,
            ...filters
        };

        const response = await this.apiCall('/search', 'POST', payload);
        return response.results || [];
    }

    /**
     * Handle filter changes in search view
     */
    async handleFilterChange() {
        const tagFilter = document.getElementById('tagFilter')?.value;
        const dateFilter = document.getElementById('dateFilter')?.value;
        const typeFilter = document.getElementById('typeFilter')?.value;

        const filters = {};
        if (tagFilter) filters.tags = tagFilter.split(',').map(t => t.trim());
        if (dateFilter) filters.date_range = dateFilter;
        if (typeFilter) filters.type = typeFilter;

        const query = document.getElementById('quickSearch').value.trim();
        if (query) {
            try {
                const results = await this.searchMemories(query, filters);
                this.searchResults = results;
                this.renderSearchResults(results);
                this.updateResultsCount(results.length);
            } catch (error) {
                console.error('Filter search error:', error);
            }
        }
    }

    /**
     * Handle view mode changes (grid/list)
     */
    handleViewModeChange(mode) {
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-view="${mode}"]`).classList.add('active');

        const resultsContainer = document.getElementById('searchResultsList');
        resultsContainer.className = mode === 'grid' ? 'memory-grid' : 'memory-list';
    }

    /**
     * Handle quick actions
     */
    handleQuickAction(action) {
        switch (action) {
            case 'quick-search':
                this.switchView('search');
                document.getElementById('quickSearch').focus();
                break;
            case 'add-memory':
                this.handleAddMemory();
                break;
            case 'browse-tags':
                this.switchView('browse');
                break;
            case 'export-data':
                this.handleExportData();
                break;
        }
    }

    /**
     * Handle add memory action
     */
    handleAddMemory() {
        const modal = document.getElementById('addMemoryModal');

        // Reset modal for adding new memory
        this.resetAddMemoryModal();

        this.openModal(modal);
        document.getElementById('memoryContent').focus();
    }

    /**
     * Reset add memory modal to default state
     */
    resetAddMemoryModal() {
        const modal = document.getElementById('addMemoryModal');
        const title = modal.querySelector('.modal-header h3');
        const saveBtn = document.getElementById('saveMemoryBtn');

        // Reset modal title and button text
        title.textContent = 'Add New Memory';
        saveBtn.textContent = 'Save Memory';

        // Clear form
        document.getElementById('addMemoryForm').reset();

        // Clear editing state
        this.editingMemory = null;
    }

    /**
     * Handle save memory
     */
    async handleSaveMemory() {
        const content = document.getElementById('memoryContent').value.trim();
        const tags = document.getElementById('memoryTags').value.trim();
        const type = document.getElementById('memoryType').value;

        if (!content) {
            this.showToast('Please enter memory content', 'warning');
            return;
        }

        const payload = {
            content: content,
            tags: tags ? tags.split(',').map(t => t.trim()) : [],
            memory_type: type,
            metadata: {
                created_via: 'dashboard',
                user_agent: navigator.userAgent,
                updated_via: this.editingMemory ? 'dashboard_edit' : 'dashboard_create'
            }
        };

        try {
            let response;
            let successMessage;

            if (this.editingMemory) {
                // Update existing memory - delete old and create new (since content hash changes)
                await this.apiCall(`/memories/${this.editingMemory.content_hash}`, 'DELETE');
                response = await this.apiCall('/memories', 'POST', payload);
                successMessage = 'Memory updated successfully';
            } else {
                // Create new memory
                response = await this.apiCall('/memories', 'POST', payload);
                successMessage = 'Memory saved successfully';
            }

            this.closeModal(document.getElementById('addMemoryModal'));
            this.showToast(successMessage, 'success');

            // Reset editing state
            this.editingMemory = null;
            this.resetAddMemoryModal();

            // Refresh current view if needed
            if (this.currentView === 'dashboard') {
                this.loadDashboardData();
            } else if (this.currentView === 'search') {
                // Refresh search results
                const query = document.getElementById('quickSearch').value.trim();
                if (query) {
                    this.handleSearch(query);
                }
            }
        } catch (error) {
            console.error('Error saving memory:', error);
            this.showToast('Failed to save memory', 'error');
        }
    }

    /**
     * Handle memory click to show details
     */
    handleMemoryClick(memory) {
        this.showMemoryDetails(memory);
    }

    /**
     * Show memory details in modal
     */
    showMemoryDetails(memory) {
        const modal = document.getElementById('memoryModal');
        const title = document.getElementById('modalTitle');
        const content = document.getElementById('modalContent');

        title.textContent = 'Memory Details';
        content.innerHTML = this.renderMemoryDetails(memory);

        // Set up action buttons
        document.getElementById('editMemoryBtn').onclick = () => this.editMemory(memory);
        document.getElementById('deleteMemoryBtn').onclick = () => this.deleteMemory(memory);
        document.getElementById('shareMemoryBtn').onclick = () => this.shareMemory(memory);

        this.openModal(modal);
    }

    /**
     * Render memory details HTML
     */
    renderMemoryDetails(memory) {
        const createdDate = new Date(memory.created_at * 1000).toLocaleString();
        const updatedDate = memory.updated_at ? new Date(memory.updated_at * 1000).toLocaleString() : null;

        return `
            <div class="memory-detail">
                <div class="memory-meta">
                    <p><strong>Created:</strong> ${createdDate}</p>
                    ${updatedDate ? `<p><strong>Updated:</strong> ${updatedDate}</p>` : ''}
                    <p><strong>Type:</strong> ${memory.memory_type || 'note'}</p>
                    <p><strong>ID:</strong> ${memory.content_hash}</p>
                </div>

                <div class="memory-content">
                    <h4>Content</h4>
                    <div class="content-text">${this.escapeHtml(memory.content)}</div>
                </div>

                ${memory.tags && memory.tags.length > 0 ? `
                    <div class="memory-tags-section">
                        <h4>Tags</h4>
                        <div class="memory-tags">
                            ${memory.tags.map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}

                ${memory.metadata ? `
                    <div class="memory-metadata">
                        <h4>Metadata</h4>
                        <pre>${JSON.stringify(memory.metadata, null, 2)}</pre>
                    </div>
                ` : ''}
            </div>
        `;
    }

    /**
     * Delete memory
     */
    async deleteMemory(memory) {
        if (!confirm('Are you sure you want to delete this memory? This action cannot be undone.')) {
            return;
        }

        try {
            await this.apiCall(`/memories/${memory.content_hash}`, 'DELETE');
            this.closeModal(document.getElementById('memoryModal'));
            this.showToast('Memory deleted successfully', 'success');

            // Refresh current view
            if (this.currentView === 'dashboard') {
                this.loadDashboardData();
            } else if (this.currentView === 'search') {
                this.searchResults = this.searchResults.filter(m => m.memory.content_hash !== memory.content_hash);
                this.renderSearchResults(this.searchResults);
            }
        } catch (error) {
            console.error('Error deleting memory:', error);
            this.showToast('Failed to delete memory', 'error');
        }
    }

    /**
     * Edit memory
     */
    editMemory(memory) {
        // Close the memory details modal first
        this.closeModal(document.getElementById('memoryModal'));

        // Open the add memory modal with pre-filled data
        const modal = document.getElementById('addMemoryModal');
        const title = modal.querySelector('.modal-header h3');
        const saveBtn = document.getElementById('saveMemoryBtn');

        // Update modal for editing
        title.textContent = 'Edit Memory';
        saveBtn.textContent = 'Update Memory';

        // Pre-fill the form with existing data
        document.getElementById('memoryContent').value = memory.content;
        document.getElementById('memoryTags').value = memory.tags ? memory.tags.join(', ') : '';
        document.getElementById('memoryType').value = memory.memory_type || 'note';

        // Store the memory being edited
        this.editingMemory = memory;

        this.openModal(modal);
        document.getElementById('memoryContent').focus();
    }

    /**
     * Share memory
     */
    shareMemory(memory) {
        // Create shareable data
        const shareData = {
            content: memory.content,
            tags: memory.tags || [],
            type: memory.memory_type || 'note',
            created: new Date(memory.created_at * 1000).toISOString(),
            id: memory.content_hash
        };

        // Try to use Web Share API if available
        if (navigator.share) {
            navigator.share({
                title: 'Memory from MCP Memory Service',
                text: memory.content,
                url: window.location.href
            }).catch(err => {
                console.log('Error sharing:', err);
                this.fallbackShare(shareData);
            });
        } else {
            this.fallbackShare(shareData);
        }
    }

    /**
     * Fallback share method (copy to clipboard)
     */
    fallbackShare(shareData) {
        const shareText = `Memory Content:\n${shareData.content}\n\nTags: ${shareData.tags.join(', ')}\nType: ${shareData.type}\nCreated: ${shareData.created}`;

        navigator.clipboard.writeText(shareText).then(() => {
            this.showToast('Memory copied to clipboard', 'success');
        }).catch(err => {
            console.error('Could not copy text: ', err);
            this.showToast('Failed to copy to clipboard', 'error');
        });
    }

    /**
     * Handle data export
     */
    async handleExportData() {
        try {
            const response = await this.apiCall('/memories?limit=1000');
            const data = {
                export_date: new Date().toISOString(),
                total_memories: response.total,
                memories: response.memories
            };

            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `mcp-memories-export-${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showToast('Data exported successfully', 'success');
        } catch (error) {
            console.error('Export error:', error);
            this.showToast('Failed to export data', 'error');
        }
    }

    /**
     * Render recent memories
     */
    renderRecentMemories(memories) {
        const container = document.getElementById('recentMemoriesList');

        if (!memories || memories.length === 0) {
            container.innerHTML = '<p class="empty-state">No memories found. <a href="#" onclick="app.handleAddMemory()">Add your first memory</a></p>';
            return;
        }

        container.innerHTML = memories.map(memory => this.renderMemoryCard(memory)).join('');

        // Add click handlers
        container.querySelectorAll('.memory-card').forEach((card, index) => {
            card.addEventListener('click', () => this.handleMemoryClick(memories[index]));
        });
    }

    /**
     * Render search results
     */
    renderSearchResults(results) {
        const container = document.getElementById('searchResultsList');

        if (!results || results.length === 0) {
            container.innerHTML = '<p class="empty-state">No results found. Try a different search term.</p>';
            return;
        }

        container.innerHTML = results.map(result => this.renderMemoryCard(result.memory, result)).join('');

        // Add click handlers
        container.querySelectorAll('.memory-card').forEach((card, index) => {
            card.addEventListener('click', () => this.handleMemoryClick(results[index].memory));
        });
    }

    /**
     * Render a memory card
     */
    renderMemoryCard(memory, searchResult = null) {
        const createdDate = new Date(memory.created_at * 1000).toLocaleDateString();
        const relevanceScore = searchResult ? (searchResult.relevance_score * 100).toFixed(1) : null;

        return `
            <div class="memory-card" data-memory-id="${memory.content_hash}">
                <div class="memory-header">
                    <div class="memory-meta">
                        <span>${createdDate}</span>
                        ${memory.memory_type ? `<span> • ${memory.memory_type}</span>` : ''}
                        ${relevanceScore ? `<span> • ${relevanceScore}% match</span>` : ''}
                    </div>
                </div>

                <div class="memory-content">
                    ${this.escapeHtml(memory.content)}
                </div>

                ${memory.tags && memory.tags.length > 0 ? `
                    <div class="memory-tags">
                        ${memory.tags.map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
                    </div>
                ` : ''}
            </div>
        `;
    }

    /**
     * Update dashboard statistics
     */
    updateDashboardStats(stats) {
        document.getElementById('totalMemories').textContent = stats.total_memories || '0';
        document.getElementById('recentMemories').textContent = stats.recent_count || '0';
        document.getElementById('uniqueTags').textContent = stats.unique_tags || '0';
    }

    /**
     * Update search results count
     */
    updateResultsCount(count) {
        const element = document.getElementById('resultsCount');
        if (element) {
            element.textContent = `${count} result${count !== 1 ? 's' : ''}`;
        }
    }

    /**
     * Handle memory added via SSE
     */
    handleMemoryAdded(memory) {
        if (this.currentView === 'dashboard') {
            this.loadDashboardData();
        }
    }

    /**
     * Handle memory deleted via SSE
     */
    handleMemoryDeleted(memoryId) {
        // Remove from current view
        const cards = document.querySelectorAll(`[data-memory-id="${memoryId}"]`);
        cards.forEach(card => card.remove());

        // Update search results if in search view
        if (this.currentView === 'search') {
            this.searchResults = this.searchResults.filter(r => r.memory.content_hash !== memoryId);
            this.updateResultsCount(this.searchResults.length);
        }
    }

    /**
     * Handle memory updated via SSE
     */
    handleMemoryUpdated(memory) {
        // Refresh relevant views
        if (this.currentView === 'dashboard') {
            this.loadDashboardData();
        }
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(status) {
        const indicator = document.querySelector('.status-indicator');
        const text = document.querySelector('.status-text');

        indicator.className = `status-indicator ${status === 'connected' ? '' : status}`;

        switch (status) {
            case 'connected':
                text.textContent = 'Connected';
                break;
            case 'connecting':
                text.textContent = 'Connecting...';
                break;
            case 'disconnected':
                text.textContent = 'Disconnected';
                break;
        }
    }

    /**
     * Generic API call wrapper
     */
    async apiCall(endpoint, method = 'GET', data = null) {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${this.apiBase}${endpoint}`, options);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Modal management
     */
    openModal(modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';

        // Focus first input
        const firstInput = modal.querySelector('input, textarea');
        if (firstInput) {
            firstInput.focus();
        }
    }

    closeModal(modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }

    /**
     * Loading state management
     */
    setLoading(loading) {
        this.isLoading = loading;
        const overlay = document.getElementById('loadingOverlay');
        if (loading) {
            overlay.classList.add('active');
        } else {
            overlay.classList.remove('active');
        }
    }

    /**
     * Toast notification system
     */
    showToast(message, type = 'info', duration = 5000) {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;

        container.appendChild(toast);

        // Auto-remove after duration
        setTimeout(() => {
            toast.remove();
        }, duration);

        // Click to remove
        toast.addEventListener('click', () => {
            toast.remove();
        });
    }

    /**
     * Utility: Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Utility: Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Cleanup when page unloads
     */
    destroy() {
        if (this.eventSource) {
            this.eventSource.close();
        }
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MemoryDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.app) {
        window.app.destroy();
    }
});