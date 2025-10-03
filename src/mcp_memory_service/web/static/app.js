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
        await this.loadVersion();
        await this.loadDashboardData();
        this.updateConnectionStatus('connected');
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

        if (quickSearch) {
            quickSearch.addEventListener('input', this.debounce(this.handleQuickSearch, 300));
            quickSearch.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleSearch(e.target.value);
                }
            });
        }

        if (searchBtn && quickSearch) {
            searchBtn.addEventListener('click', () => {
                this.handleSearch(quickSearch.value);
            });
        }

        // Add memory functionality
        const addMemoryBtn = document.getElementById('addMemoryBtn');
        if (addMemoryBtn) {
            addMemoryBtn.addEventListener('click', this.handleAddMemory);
        }
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
        const saveMemoryBtn = document.getElementById('saveMemoryBtn');
        if (saveMemoryBtn) {
            saveMemoryBtn.addEventListener('click', this.handleSaveMemory.bind(this));
        }

        const cancelAddBtn = document.getElementById('cancelAddBtn');
        if (cancelAddBtn) {
            cancelAddBtn.addEventListener('click', () => {
                this.closeModal(document.getElementById('addMemoryModal'));
            });
        }

        // Quick action handlers
        document.querySelectorAll('.action-card').forEach(card => {
            card.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleQuickAction(action);
            });
        });

        // Filter handlers for search view
        const tagFilterInput = document.getElementById('tagFilter');
        tagFilterInput?.addEventListener('input', this.handleFilterChange.bind(this));
        tagFilterInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleFilterChange();
            }
        });
        document.getElementById('dateFilter')?.addEventListener('change', this.handleFilterChange.bind(this));
        document.getElementById('typeFilter')?.addEventListener('change', this.handleFilterChange.bind(this));

        // View option handlers
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.handleViewModeChange(e.target.dataset.view);
            });
        });

        // New filter action handlers
        document.getElementById('applyFiltersBtn')?.addEventListener('click', this.handleFilterChange.bind(this));
        document.getElementById('clearFiltersBtn')?.addEventListener('click', this.clearAllFilters.bind(this));

        // Settings button
        document.getElementById('settingsBtn')?.addEventListener('click', () => {
            this.showToast('Settings functionality not yet implemented.', 'info');
        });

        // Tag cloud event delegation
        document.getElementById('tagsCloudContainer')?.addEventListener('click', (e) => {
            if (e.target.classList.contains('tag-bubble') || e.target.closest('.tag-bubble')) {
                const tagButton = e.target.classList.contains('tag-bubble') ? e.target : e.target.closest('.tag-bubble');
                const tag = tagButton.dataset.tag;
                if (tag) {
                    this.filterByTag(tag);
                }
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                document.getElementById('searchInput').focus();
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
                // Unknown event type - ignore silently
        }
    }

    /**
     * Load application version from health endpoint
     */
    async loadVersion() {
        try {
            const healthResponse = await this.apiCall('/health');
            const versionBadge = document.getElementById('versionBadge');
            if (versionBadge && healthResponse.version) {
                versionBadge.textContent = `v${healthResponse.version}`;
            }
        } catch (error) {
            console.error('Error loading version:', error);
            const versionBadge = document.getElementById('versionBadge');
            if (versionBadge) {
                versionBadge.textContent = 'v?.?.?';
            }
        }
    }

    /**
     * Load initial dashboard data
     */
    async loadDashboardData() {
        this.setLoading(true);

        try {
            // Load recent memories for dashboard display
            const memoriesResponse = await this.apiCall('/memories?page=1&page_size=100');
            if (memoriesResponse.memories) {
                this.memories = memoriesResponse.memories;
                this.renderRecentMemories(memoriesResponse.memories);
            }

            // Load basic statistics
            const statsResponse = await this.apiCall('/health/detailed');
            if (statsResponse.storage) {
                this.updateDashboardStats(statsResponse.storage);
            }


        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showToast('Failed to load dashboard data', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Load browse view data (tags)
     */
    async loadBrowseData() {
        this.setLoading(true);
        try {
            // Load tags with counts from the dedicated endpoint
            const tagsResponse = await this.apiCall('/tags');
            if (tagsResponse.tags) {
                this.tags = tagsResponse.tags;
                this.renderTagsCloud();
            }
        } catch (error) {
            console.error('Error loading browse data:', error);
            this.showToast('Failed to load browse data', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Render tags cloud from API data
     */
    renderTagsCloud() {
        const container = document.getElementById('tagsCloudContainer');
        const taggedContainer = document.getElementById('taggedMemoriesContainer');

        // Hide the tagged memories view initially
        taggedContainer.style.display = 'none';

        if (!this.tags || this.tags.length === 0) {
            container.innerHTML = '<p class="text-neutral-600">No tags found. Start adding tags to your memories to see them here.</p>';
            return;
        }

        // Render tag bubbles (tags are already sorted by count from backend)
        container.innerHTML = this.tags.map(tagData => `
            <button class="tag-bubble" data-tag="${this.escapeHtml(tagData.tag)}">
                ${this.escapeHtml(tagData.tag)}
                <span class="count">${tagData.count}</span>
            </button>
        `).join('');
    }

    /**
     * Filter memories by selected tag
     */
    async filterByTag(tag) {
        const taggedContainer = document.getElementById('taggedMemoriesContainer');
        const tagNameSpan = document.getElementById('selectedTagName');
        const memoriesList = document.getElementById('taggedMemoriesList');

        try {
            // Fetch memories for this specific tag
            const memoriesResponse = await this.apiCall(`/memories?tag=${encodeURIComponent(tag)}&limit=100`);
            const filteredMemories = memoriesResponse.memories || [];

            // Show the tagged memories section
            tagNameSpan.textContent = tag;
            taggedContainer.style.display = 'block';

            // Render filtered memories
            this.renderMemoriesInContainer(filteredMemories, memoriesList);

            // Add event listener for clear filter button
            const clearBtn = document.getElementById('clearTagFilter');
            clearBtn.onclick = () => this.clearTagFilter();
        } catch (error) {
            console.error('Error filtering by tag:', error);
            this.showToast('Failed to load memories for tag', 'error');
        }
    }

    /**
     * Clear tag filter and show all tags
     */
    clearTagFilter() {
        const taggedContainer = document.getElementById('taggedMemoriesContainer');
        taggedContainer.style.display = 'none';
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Render memories in a specific container
     */
    renderMemoriesInContainer(memories, container) {
        if (!memories || memories.length === 0) {
            container.innerHTML = '<p class="empty-state">No memories found with this tag.</p>';
            return;
        }

        container.innerHTML = memories.map(memory => this.renderMemoryCard(memory)).join('');

        // Add click handlers
        container.querySelectorAll('.memory-card').forEach((card, index) => {
            card.addEventListener('click', () => this.handleMemoryClick(memories[index]));
        });
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
        // Update navigation active state (if navigation exists)
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        const navItem = document.querySelector(`[data-view="${viewName}"]`);
        if (navItem) {
            navItem.classList.add('active');
        }

        // Hide all views (if view containers exist)
        document.querySelectorAll('.view-container').forEach(view => {
            view.classList.remove('active');
        });

        // Show target view (if it exists)
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
                await this.loadBrowseData();
                break;
            case 'manage':
                // Load management tools
                break;
            case 'analytics':
                // Load analytics data
                break;
            case 'apiDocs':
                // API docs view - static content, no additional loading needed
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
        // Detect tag search patterns: #tag, tag:value, or "tag:value"
        const tagPattern = /^(#|tag:)(.+)$/i;
        const tagMatch = query.match(tagPattern);

        if (tagMatch) {
            // Use tag search endpoint
            const tagValue = tagMatch[2].trim();
            const payload = {
                tags: [tagValue],
                match_all: false // ANY match by default
            };

            const response = await this.apiCall('/search/by-tag', 'POST', payload);
            return response.results || [];
        } else {
            // Use semantic search endpoint
            const payload = {
                query: query,
                n_results: filters.limit || 20,
                similarity_threshold: filters.threshold || 0.7,
                ...filters
            };

            const response = await this.apiCall('/search', 'POST', payload);
            return response.results || [];
        }
    }

    /**
     * Handle filter changes in search view
     */
    async handleFilterChange() {
        const tagFilter = document.getElementById('tagFilter')?.value;
        const dateFilter = document.getElementById('dateFilter')?.value;
        const typeFilter = document.getElementById('typeFilter')?.value;
        const query = document.getElementById('quickSearch')?.value?.trim() || '';

        // Add loading state
        const applyBtn = document.getElementById('applyFiltersBtn');
        if (applyBtn) {
            applyBtn.classList.add('loading');
            applyBtn.disabled = true;
        }

        try {
            let results = [];

            if (tagFilter && tagFilter.trim()) {
                // If we have tags, use tag search (works with or without main query)
                const tags = tagFilter.split(',').map(t => t.trim()).filter(t => t);

                if (tags.length > 0) {
                    const payload = {
                        tags: tags,
                        match_all: false // ANY match by default
                    };

                    const response = await this.apiCall('/search/by-tag', 'POST', payload);
                    results = response.results || [];

                    // If we also have a main query, filter results further with semantic search
                    if (query) {
                        // TODO: Could implement hybrid search here
                        // For now, tag search takes precedence
                    }
                }
            } else if (dateFilter && dateFilter.trim()) {
                // Date filter only - use time-based search
                const payload = {
                    query: dateFilter,
                    n_results: 100
                };
                const response = await this.apiCall('/search/by-time', 'POST', payload);
                results = response.results || [];

                // Apply type filter if present
                if (typeFilter && typeFilter.trim()) {
                    results = results.filter(result => {
                        const memoryType = result.memory.memory_type || 'note';
                        return memoryType === typeFilter;
                    });
                }
            } else if (query) {
                // Semantic search with optional type filter
                const filters = {};
                if (typeFilter) filters.type = typeFilter;
                results = await this.searchMemories(query, filters);
            } else if (typeFilter && typeFilter.trim()) {
                // Only type filter - get all memories and filter by type
                const allMemoriesResponse = await this.apiCall('/memories?page=1&page_size=1000');
                if (allMemoriesResponse.memories) {
                    results = allMemoriesResponse.memories
                        .filter(memory => (memory.memory_type || 'note') === typeFilter)
                        .map(memory => ({ memory, similarity: 1.0 }));
                }
            } else {
                // No filters, clear results
                results = [];
            }

            this.searchResults = results;
            this.renderSearchResults(results);
            this.updateResultsCount(results.length);
            this.updateActiveFilters();

        } catch (error) {
            console.error('Filter search error:', error);
            this.showToast('Filter search failed', 'error');
        } finally {
            // Remove loading state
            const applyBtn = document.getElementById('applyFiltersBtn');
            if (applyBtn) {
                applyBtn.classList.remove('loading');
                applyBtn.disabled = false;
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
                const searchInput = document.getElementById('quickSearch');
                if (searchInput) {
                    searchInput.focus();
                }
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
                // Smart update: check if only metadata changed vs content changes
                const originalContentHash = this.editingMemory.content_hash;
                const contentChanged = this.editingMemory.content !== payload.content;


                if (!contentChanged) {
                    // Only metadata (tags, type, metadata) changed - use PUT endpoint
                    const updatePayload = {
                        tags: payload.tags,
                        memory_type: payload.memory_type,
                        metadata: payload.metadata
                    };

                    response = await this.apiCall(`/memories/${originalContentHash}`, 'PUT', updatePayload);
                    successMessage = 'Memory updated successfully';
                } else {
                    // Content changed - use create-delete approach (but with proper error handling)

                    try {
                        // Step 1: Create updated memory first
                        response = await this.apiCall('/memories', 'POST', payload);

                        // CRITICAL: Only proceed with deletion if creation actually succeeded
                        if (response.success) {
                            successMessage = 'Memory updated successfully';

                            try {
                                // Step 2: Delete original memory (only after successful creation)
                                const deleteResponse = await this.apiCall(`/memories/${originalContentHash}`, 'DELETE');
                            } catch (deleteError) {
                                console.error('Failed to delete original memory after creating new version:', deleteError);
                                this.showToast('Memory updated, but original version still exists. You may need to manually delete the duplicate.', 'warning');
                            }
                        } else {
                            // Creation failed - do NOT delete original memory
                            console.error('Creation failed:', response.message);
                            throw new Error(`Failed to create updated memory: ${response.message}`);
                        }
                    } catch (createError) {
                        // CREATE failed - original memory intact, no cleanup needed
                        console.error('Failed to create updated memory:', createError);
                        throw new Error(`Failed to update memory: ${createError.message}`);
                    }
                }
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
                const query = document.getElementById('searchInput').value.trim();
                if (query) {
                    this.handleSearch(query);
                }
            } else if (this.currentView === 'browse') {
                // Refresh browse view (tags cloud)
                this.loadBrowseData();
            }
        } catch (error) {
            console.error('Error saving memory:', error);
            this.showToast(error.message || 'Failed to save memory', 'error');
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
                        <h4 class="metadata-toggle" onclick="this.parentElement.classList.toggle('expanded')" style="cursor: pointer; user-select: none;">
                            <span class="toggle-icon">▶</span> Metadata
                        </h4>
                        <div class="metadata-content">
                            ${this.renderMetadata(memory.metadata)}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    /**
     * Render metadata in a prettier format
     */
    renderMetadata(metadata) {
        if (!metadata || typeof metadata !== 'object') {
            return '<p class="metadata-empty">No metadata available</p>';
        }

        let html = '<div class="metadata-items">';

        for (const [key, value] of Object.entries(metadata)) {
            let displayValue;

            if (typeof value === 'string') {
                displayValue = `<span class="metadata-string">"${this.escapeHtml(value)}"</span>`;
            } else if (typeof value === 'number') {
                displayValue = `<span class="metadata-number">${value}</span>`;
            } else if (typeof value === 'boolean') {
                displayValue = `<span class="metadata-boolean">${value}</span>`;
            } else if (Array.isArray(value)) {
                displayValue = `<span class="metadata-array">[${value.map(v =>
                    typeof v === 'string' ? `"${this.escapeHtml(v)}"` : v
                ).join(', ')}]</span>`;
            } else {
                displayValue = `<span class="metadata-object">${JSON.stringify(value)}</span>`;
            }

            html += `
                <div class="metadata-item">
                    <span class="metadata-key">${this.escapeHtml(key)}:</span>
                    <span class="metadata-value">${displayValue}</span>
                </div>
            `;
        }

        html += '</div>';
        return html;
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
            } else if (this.currentView === 'browse') {
                // Refresh browse view (tags cloud)
                this.loadBrowseData();
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
        document.getElementById('memoryContent').value = memory.content || '';

        // Handle tags - ensure they're displayed correctly
        const tagsValue = memory.tags && Array.isArray(memory.tags) ? memory.tags.join(', ') : '';
        document.getElementById('memoryTags').value = tagsValue;

        document.getElementById('memoryType').value = memory.memory_type || 'note';


        // Store the memory being edited
        this.editingMemory = memory;

        this.openModal(modal);

        // Use setTimeout to ensure modal is fully rendered before setting values
        setTimeout(() => {
            document.getElementById('memoryContent').focus();
        }, 100);
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
                // Share API failed, fall back to clipboard
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
            this.showToast('Preparing export...', 'info');

            // Fetch all memories using pagination
            const allMemories = [];
            const pageSize = 100; // Reasonable batch size
            let page = 1;
            let hasMore = true;
            let totalMemories = 0;

            while (hasMore) {
                const response = await this.apiCall(`/memories?page=${page}&page_size=${pageSize}`);

                if (page === 1) {
                    totalMemories = response.total;
                }

                if (response.memories && response.memories.length > 0) {
                    allMemories.push(...response.memories);
                    hasMore = response.has_more;
                    page++;

                    // Update progress
                    this.showToast(`Fetching memories... (${allMemories.length}/${totalMemories})`, 'info');
                } else {
                    hasMore = false;
                }
            }

            const data = {
                export_date: new Date().toISOString(),
                total_memories: totalMemories,
                exported_memories: allMemories.length,
                memories: allMemories
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

            this.showToast(`Successfully exported ${allMemories.length} memories`, 'success');
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

        if (!container) {
            console.error('recentMemoriesList container not found');
            return;
        }

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
     * Render tags in sidebar
     */
    renderTagsSidebar(tags) {
        const container = document.getElementById('tagsCloudContainer');

        if (!container) {
            console.warn('tagsCloudContainer element not found - skipping tags sidebar rendering');
            return;
        }

        if (!tags || tags.length === 0) {
            container.innerHTML = '<div class="no-tags">No tags found.</div>';
            return;
        }

        // Take top tags for sidebar display
        const topTags = tags.slice(0, 10);
        container.innerHTML = topTags.map(tagData => `
            <div class="tag-item" data-tag="${this.escapeHtml(tagData.tag)}">
                <span class="tag-name">${this.escapeHtml(tagData.tag)}</span>
                <span class="tag-count">${tagData.count}</span>
            </div>
        `).join('');

        // Add click handlers
        container.querySelectorAll('.tag-item').forEach(item => {
            item.addEventListener('click', () => {
                const tagName = item.dataset.tag;
                const searchInput = document.getElementById('searchInput');
                searchInput.value = `#${tagName}`;
                this.handleSearch(`#${tagName}`);
            });
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
        const relevanceScore = searchResult &&
            searchResult.similarity_score !== null &&
            searchResult.similarity_score !== undefined &&
            !isNaN(searchResult.similarity_score) &&
            searchResult.similarity_score > 0
            ? (searchResult.similarity_score * 100).toFixed(1)
            : null;

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
        const totalMemoriesEl = document.getElementById('totalMemories');
        if (totalMemoriesEl) {
            totalMemoriesEl.textContent = stats.total_memories || '0';
        }

        const recentMemoriesEl = document.getElementById('recentMemories');
        if (recentMemoriesEl) {
            recentMemoriesEl.textContent = stats.memories_this_week || '0';
        }

        const uniqueTagsEl = document.getElementById('uniqueTags');
        if (uniqueTagsEl) {
            uniqueTagsEl.textContent = stats.unique_tags || '0';
        }

        const storageBackendEl = document.getElementById('storageBackend');
        if (storageBackendEl) {
            storageBackendEl.textContent = stats.backend || 'unknown';
        }
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
     * Update active filters display
     */
    updateActiveFilters() {
        const activeFiltersContainer = document.getElementById('activeFilters');
        const filtersList = document.getElementById('activeFiltersList');

        if (!activeFiltersContainer || !filtersList) return;

        const tagFilter = document.getElementById('tagFilter')?.value?.trim();
        const dateFilter = document.getElementById('dateFilter')?.value;
        const typeFilter = document.getElementById('typeFilter')?.value;

        const filters = [];

        if (tagFilter) {
            const tags = tagFilter.split(',').map(t => t.trim()).filter(t => t);
            tags.forEach(tag => {
                filters.push({
                    type: 'tag',
                    value: tag,
                    label: `Tag: ${tag}`
                });
            });
        }

        if (dateFilter) {
            const dateLabels = {
                'today': 'Today',
                'week': 'This week',
                'month': 'This month',
                'year': 'This year'
            };
            filters.push({
                type: 'date',
                value: dateFilter,
                label: `Date: ${dateLabels[dateFilter] || dateFilter}`
            });
        }

        if (typeFilter) {
            const typeLabels = {
                'note': 'Notes',
                'code': 'Code',
                'reference': 'References',
                'idea': 'Ideas'
            };
            filters.push({
                type: 'type',
                value: typeFilter,
                label: `Type: ${typeLabels[typeFilter] || typeFilter}`
            });
        }

        if (filters.length === 0) {
            activeFiltersContainer.style.display = 'none';
            return;
        }

        activeFiltersContainer.style.display = 'block';
        filtersList.innerHTML = filters.map(filter => `
            <div class="filter-pill">
                ${this.escapeHtml(filter.label)}
                <button class="remove-filter" onclick="dashboard.removeFilter('${filter.type}', '${this.escapeHtml(filter.value)}')">
                    ×
                </button>
            </div>
        `).join('');
    }

    /**
     * Remove a specific filter
     */
    removeFilter(type, value) {
        switch (type) {
            case 'tag':
                const tagInput = document.getElementById('tagFilter');
                if (tagInput) {
                    const tags = tagInput.value.split(',').map(t => t.trim()).filter(t => t && t !== value);
                    tagInput.value = tags.join(', ');
                }
                break;
            case 'date':
                const dateSelect = document.getElementById('dateFilter');
                if (dateSelect) {
                    dateSelect.value = '';
                }
                break;
            case 'type':
                const typeSelect = document.getElementById('typeFilter');
                if (typeSelect) {
                    typeSelect.value = '';
                }
                break;
        }
        this.handleFilterChange();
    }

    /**
     * Clear all filters
     */
    clearAllFilters() {
        const tagFilter = document.getElementById('tagFilter');
        const dateFilter = document.getElementById('dateFilter');
        const typeFilter = document.getElementById('typeFilter');

        if (tagFilter) tagFilter.value = '';
        if (dateFilter) dateFilter.value = '';
        if (typeFilter) typeFilter.value = '';

        this.searchResults = [];
        this.renderSearchResults([]);
        this.updateResultsCount(0);
        this.updateActiveFilters();

        this.showToast('All filters cleared', 'info');
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
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            const indicator = statusElement.querySelector('.status-indicator');
            const text = statusElement.querySelector('.status-text');
            if (!indicator || !text) return;

            // Reset indicator classes
            indicator.className = 'status-indicator';

            switch (status) {
                case 'connected':
                    text.textContent = 'Connected';
                    // Connected uses default green color (no additional class needed)
                    break;
                case 'connecting':
                    text.textContent = 'Connecting...';
                    indicator.classList.add('connecting');
                    break;
                case 'disconnected':
                    text.textContent = 'Disconnected';
                    indicator.classList.add('disconnected');
                    break;
                default:
                    text.textContent = 'Unknown';
                    indicator.classList.add('disconnected');
            }
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
        const indicator = document.getElementById('loadingOverlay');
        if (indicator) {
            if (loading) {
                indicator.classList.remove('hidden');
            } else {
                indicator.classList.add('hidden');
            }
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