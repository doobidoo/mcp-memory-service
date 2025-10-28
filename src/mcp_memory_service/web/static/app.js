/**
 * MCP Memory Service Dashboard - Main Application
 * Interactive frontend for memory management with real-time updates
 */

class MemoryDashboard {
    // Delay between individual file uploads to avoid overwhelming the server (ms)
    static INDIVIDUAL_UPLOAD_DELAY = 500;

    // Static configuration for settings modal system information
    static SYSTEM_INFO_CONFIG = {
        settingsVersion: {
            sources: [{ path: 'version', api: 'health' }],
            formatter: (value) => value || 'N/A'
        },
        settingsBackend: {
            sources: [
                { path: 'storage.storage_backend', api: 'detailedHealth' },
                { path: 'storage.backend', api: 'detailedHealth' }
            ],
            formatter: (value) => value || 'N/A'
        },
        settingsPrimaryBackend: {
            sources: [
                { path: 'storage.primary_backend', api: 'detailedHealth' },
                { path: 'storage.backend', api: 'detailedHealth' }
            ],
            formatter: (value) => value || 'N/A'
        },
        settingsEmbeddingModel: {
            sources: [
                { path: 'storage.primary_stats.embedding_model', api: 'detailedHealth' },
                { path: 'storage.embedding_model', api: 'detailedHealth' }
            ],
            formatter: (value) => value || 'N/A'
        },
        settingsEmbeddingDim: {
            sources: [
                { path: 'storage.primary_stats.embedding_dimension', api: 'detailedHealth' },
                { path: 'storage.embedding_dimension', api: 'detailedHealth' }
            ],
            formatter: (value) => value || 'N/A'
        },
        settingsDbSize: {
            sources: [
                { path: 'storage.primary_stats.database_size_mb', api: 'detailedHealth' },
                { path: 'storage.database_size_mb', api: 'detailedHealth' }
            ],
            formatter: (value) => (value != null) ? `${value.toFixed(2)} MB` : 'N/A'
        },
        settingsTotalMemories: {
            sources: [{ path: 'storage.total_memories', api: 'detailedHealth' }],
            formatter: (value) => (value != null) ? value.toLocaleString() : 'N/A'
        },
        settingsUptime: {
            sources: [{ path: 'uptime_seconds', api: 'detailedHealth' }],
            formatter: (value) => (value != null) ? MemoryDashboard.formatUptime(value) : 'N/A'
        }
    };

    constructor() {
        this.apiBase = '/api';
        this.eventSource = null;
        this.memories = [];
        this.currentView = 'dashboard';
        this.searchResults = [];
        this.isLoading = false;
        this.liveSearchEnabled = true;
        this.debounceTimer = null;

        // Settings with defaults
        this.settings = {
            theme: 'light',
            viewDensity: 'comfortable',
            previewLines: 3
        };

        // Documents upload state
        this.selectedFiles = [];
        this.documentsListenersSetup = false;
        this.processingMode = 'batch'; // 'batch' or 'individual'

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
        this.loadSettings();
        this.applyTheme();
        this.setupEventListeners();
        this.setupSSE();
        await this.loadVersion();
        await this.loadDashboardData();
        this.updateConnectionStatus('connected');

        // Initialize sync status monitoring for hybrid mode
        await this.checkSyncStatus();
        this.startSyncStatusMonitoring();
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

        // Live search toggle handler
        const liveSearchToggle = document.getElementById('liveSearchToggle');
        liveSearchToggle?.addEventListener('change', this.handleLiveSearchToggle.bind(this));

        // Filter handlers for search view
        const tagFilterInput = document.getElementById('tagFilter');
        tagFilterInput?.addEventListener('input', this.handleDebouncedFilterChange.bind(this));
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

        // Theme toggle button
        document.getElementById('themeToggleBtn')?.addEventListener('click', () => {
            this.toggleTheme();
        });

        // Settings button
        document.getElementById('settingsBtn')?.addEventListener('click', () => {
            this.openSettingsModal();
        });

        // Settings modal handlers
        document.getElementById('saveSettingsBtn')?.addEventListener('click', () => {
            this.saveSettings();
        });

        document.getElementById('cancelSettingsBtn')?.addEventListener('click', () => {
            this.closeModal(document.getElementById('settingsModal'));
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

        // Manage tab event listeners
        document.getElementById('deleteByTagBtn')?.addEventListener('click', this.handleBulkDeleteByTag.bind(this));
        document.getElementById('cleanupDuplicatesBtn')?.addEventListener('click', this.handleCleanupDuplicates.bind(this));
        document.getElementById('deleteByDateBtn')?.addEventListener('click', this.handleBulkDeleteByDate.bind(this));
        document.getElementById('optimizeDbBtn')?.addEventListener('click', this.handleOptimizeDatabase.bind(this));
        document.getElementById('rebuildIndexBtn')?.addEventListener('click', this.handleRebuildIndex.bind(this));

        // Analytics tab event listeners
        document.getElementById('growthPeriodSelect')?.addEventListener('change', this.handleGrowthPeriodChange.bind(this));

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
     * Load documents view data
     */
    async loadDocumentsData() {
        this.setLoading(true);
        try {
            // Load upload history
            await this.loadUploadHistory();
            // Setup document upload event listeners
            this.setupDocumentsEventListeners();
        } catch (error) {
            console.error('Error loading documents data:', error);
            this.showToast('Failed to load documents data', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Load upload history from API
     */
    async loadUploadHistory() {
        console.log('Loading upload history...');
        try {
            const historyResponse = await this.apiCall('/documents/history');
            console.log('Upload history response:', historyResponse);
            if (historyResponse.uploads) {
                this.renderUploadHistory(historyResponse.uploads);
            } else {
                console.warn('No uploads property in response');
                this.renderUploadHistory([]);
            }
        } catch (error) {
            console.error('Error loading upload history:', error);
            // Show a message in the history container instead of just logging
            const historyContainer = document.getElementById('uploadHistory');
            if (historyContainer) {
                historyContainer.innerHTML = '<p style="text-align: center; color: var(--error);">Failed to load upload history. Please check the console for details.</p>';
            }
        }
    }

    /**
     * Render upload history
     */
    renderUploadHistory(uploads) {
        const historyContainer = document.getElementById('uploadHistory');
        if (!historyContainer) return;

        if (uploads.length === 0) {
            historyContainer.innerHTML = '<p style="text-align: center; color: var(--neutral-500);">No uploads yet. Start by uploading some documents!</p>';
            return;
        }

        const historyHtml = uploads.map(upload => {
        const statusClass = upload.status.toLowerCase();
        const statusText = upload.status.charAt(0).toUpperCase() + upload.status.slice(1);
        const progressPercent = upload.progress || 0;
        const hasMemories = upload.chunks_stored > 0;

        return `
        <div class="upload-item ${statusClass}" data-upload-id="${upload.upload_id}" data-filename="${this.escapeHtml(upload.filename)}">
        <div class="upload-info">
        <div class="upload-filename">${this.escapeHtml(upload.filename)}</div>
                        <div class="upload-meta">
                            ${upload.chunks_stored || 0} chunks stored •
                            ${(upload.file_size / 1024).toFixed(1)} KB •
                            ${new Date(upload.created_at).toLocaleString()}
                        </div>
                        ${upload.status === 'processing' ? `
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${progressPercent}%"></div>
                            </div>
                        ` : ''}
                    </div>
                    <div class="upload-actions-container">
                        <div class="upload-status ${statusClass}">
                            <span>${statusText}</span>
                            ${upload.errors && upload.errors.length > 0 ? `
                                <span title="${this.escapeHtml(upload.errors.join('; '))}">⚠️</span>
                            ` : ''}
                        </div>
                        ${upload.status === 'completed' && hasMemories ? `
                            <div class="upload-actions">
                                <button class="btn-icon btn-view-memory"
                                title="View memory chunks">
                                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                                    </svg>
                                    <span>View</span>
                                </button>
                                <button class="btn-icon btn-remove"
                                title="Remove document and memories">
                                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                                        <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                                    </svg>
                                    <span>Remove</span>
                                </button>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');

        historyContainer.innerHTML = historyHtml;
    }

    /**
     * Setup event listeners for documents view
     */
    setupDocumentsEventListeners() {
        // Prevent duplicate event listener setup
        if (this.documentsListenersSetup) {
            console.log('Document listeners already set up, skipping...');
            return;
        }

        // File selection buttons
        const fileSelectBtn = document.getElementById('fileSelectBtn');
        const fileInput = document.getElementById('fileInput');

        if (fileSelectBtn && fileInput) {
            fileSelectBtn.addEventListener('click', () => {
                fileInput.click();
            });

            fileInput.addEventListener('change', (e) => {
                this.handleFileSelection(e.target.files);
            });
        }

        // Drag and drop
        const dropZone = document.getElementById('dropZone');
        if (dropZone) {
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });

            dropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                const files = e.dataTransfer.files;
                this.handleFileSelection(files);
            });
        }

        // Configuration controls
        const chunkSizeInput = document.getElementById('chunkSize');
        const chunkOverlapInput = document.getElementById('chunkOverlap');
        const chunkSizeValue = document.getElementById('chunkSizeValue');
        const chunkOverlapValue = document.getElementById('chunkOverlapValue');

        if (chunkSizeInput && chunkSizeValue) {
            chunkSizeInput.addEventListener('input', (e) => {
                chunkSizeValue.textContent = e.target.value;
                this.updateUploadButton();
            });
        }

        // Chunking help info icon
        const infoIcon = document.querySelector('.info-icon');
        if (infoIcon) {
            infoIcon.addEventListener('click', () => {
                this.toggleChunkingHelp();
            });
        }

        // Overlap help info icon
        const overlapInfoIcon = document.querySelector('.info-icon-overlap');
        if (overlapInfoIcon) {
            overlapInfoIcon.addEventListener('click', () => {
                this.toggleOverlapHelp();
            });
        }

        // Processing mode help info icon
        const processingModeInfoIcon = document.querySelector('.info-icon-processing');
        if (processingModeInfoIcon) {
            processingModeInfoIcon.addEventListener('click', () => {
                this.toggleProcessingModeHelp();
            });
        }

        if (chunkOverlapInput && chunkOverlapValue) {
            chunkOverlapInput.addEventListener('input', (e) => {
                chunkOverlapValue.textContent = e.target.value;
                this.updateUploadButton();
            });
        }

        // Processing mode toggle buttons
        const batchModeBtn = document.getElementById('batchModeBtn');
        const individualModeBtn = document.getElementById('individualModeBtn');

        if (batchModeBtn) {
            batchModeBtn.addEventListener('click', () => {
                this.setProcessingMode('batch');
            });
        }

        if (individualModeBtn) {
            individualModeBtn.addEventListener('click', () => {
                this.setProcessingMode('individual');
            });
        }

        // Upload button
        const uploadBtn = document.getElementById('uploadBtn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => {
                this.handleDocumentUpload();
            });
        }

        // Add event listeners for buttons with data-action attribute
        document.querySelectorAll('[data-action]').forEach(button => {
            button.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                if (this[action] && typeof this[action] === 'function') {
                    this[action]();
                }
            });
        });

        // Force sync button
        const forceSyncButton = document.getElementById('forceSyncButton');
        if (forceSyncButton) {
            forceSyncButton.addEventListener('click', () => {
                this.forceSync();
            });
        }

        // Document search button
        const docSearchBtn = document.getElementById('docSearchBtn');
        const docSearchInput = document.getElementById('docSearchInput');
        if (docSearchBtn && docSearchInput) {
            docSearchBtn.addEventListener('click', () => {
                const query = docSearchInput.value.trim();
                if (query) {
                    this.searchDocumentContent(query);
                } else {
                    this.showToast('Please enter a search query', 'warning');
                }
            });

            // Enter key to search
            docSearchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    const query = docSearchInput.value.trim();
                    if (query) {
                        this.searchDocumentContent(query);
                    }
                }
            });
        }

        // Upload history action buttons (event delegation)
        const uploadHistory = document.getElementById('uploadHistory');
        if (uploadHistory) {
            uploadHistory.addEventListener('click', (e) => {
                const button = e.target.closest('.btn-view-memory, .btn-remove');
                if (!button) return;

                const uploadItem = button.closest('.upload-item');
                const uploadId = uploadItem?.dataset.uploadId;
                const filename = uploadItem?.dataset.filename;

                if (!uploadId) return;

                if (button.classList.contains('btn-view-memory')) {
                    this.viewDocumentMemory(uploadId);
                } else if (button.classList.contains('btn-remove')) {
                    this.removeDocument(uploadId, filename);
                }
            });
        }

        // Close modal when clicking outside
        const memoryViewerModal = document.getElementById('memoryViewerModal');
        if (memoryViewerModal) {
            memoryViewerModal.addEventListener('click', (e) => {
                if (e.target === memoryViewerModal) {
                    this.closeMemoryViewer();
                }
            });
        }

        // Mark listeners as set up to prevent duplicates
        this.documentsListenersSetup = true;
        console.log('Document listeners setup complete');
    }

    /**
     * Handle file selection from input or drag-drop
     */
    handleFileSelection(files) {
        if (!files || files.length === 0) return;

        this.selectedFiles = Array.from(files);
        this.updateUploadButton();

        // Show file preview in drop zone
        const dropZone = document.getElementById('dropZone');
        if (dropZone) {
            const fileNames = this.selectedFiles.map(f => this.escapeHtml(f.name)).join(', ');
            const content = dropZone.querySelector('.drop-zone-content');
            if (content) {
                content.innerHTML = `
                    <svg width="48" height="48" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
                    </svg>
                    <h3>${files.length} file${files.length > 1 ? 's' : ''} selected</h3>
                    <p>${fileNames}</p>
                    <input type="file" id="fileInput" multiple accept=".pdf,.docx,.pptx,.txt,.md,.json" style="display: none;">
                `;
            }
        }
    }

    /**
     * Update upload button state based on selections
     */
    updateUploadButton() {
        const uploadBtn = document.getElementById('uploadBtn');
        const hasFiles = this.selectedFiles && this.selectedFiles.length > 0;

        if (uploadBtn) {
            uploadBtn.disabled = !hasFiles;
            uploadBtn.textContent = hasFiles ?
                `Upload & Ingest ${this.selectedFiles.length} file${this.selectedFiles.length > 1 ? 's' : ''}` :
                'Upload & Ingest';
        }

        // Show/hide processing mode section based on file count
        const processingModeSection = document.getElementById('processingModeSection');
        if (processingModeSection) {
            processingModeSection.style.display = (hasFiles && this.selectedFiles.length > 1) ? 'block' : 'none';
        }
    }

    /**
     * Set processing mode (batch or individual)
     */
    setProcessingMode(mode) {
        this.processingMode = mode;

        // Update button states
        const batchBtn = document.getElementById('batchModeBtn');
        const individualBtn = document.getElementById('individualModeBtn');
        const modeDescription = document.getElementById('modeDescription');

        if (batchBtn) {
            batchBtn.classList.toggle('active', mode === 'batch');
        }
        if (individualBtn) {
            individualBtn.classList.toggle('active', mode === 'individual');
        }
        if (modeDescription) {
            modeDescription.innerHTML = mode === 'batch'
                ? '<small>All selected files will be processed together with the same tags.</small>'
                : '<small>Each file will be processed individually with the same tags.</small>';
        }

        console.log(`Processing mode set to: ${mode}`);
    }

    /**
     * Handle document upload
     */
    async handleDocumentUpload() {
        if (!this.selectedFiles || this.selectedFiles.length === 0) {
            this.showToast('No files selected', 'error');
            return;
        }

        const tags = document.getElementById('docTags')?.value || '';
        const chunkSize = document.getElementById('chunkSize')?.value || 1000;
        const chunkOverlap = document.getElementById('chunkOverlap')?.value || 200;
        const memoryType = document.getElementById('memoryType')?.value || 'document';

        try {
            this.setLoading(true);

            if (this.selectedFiles.length === 1 || this.processingMode === 'individual') {
                // Individual file processing (single file or individual mode for multiple files)
                for (let i = 0; i < this.selectedFiles.length; i++) {
                    const file = this.selectedFiles[i];
                    try {
                        await this.uploadSingleDocument(file, {
                            tags,
                            chunk_size: parseInt(chunkSize),
                            chunk_overlap: parseInt(chunkOverlap),
                            memory_type: memoryType
                        });

                        // Small delay between individual uploads to avoid overwhelming the server
                        if (i < this.selectedFiles.length - 1) {
                            await new Promise(resolve => setTimeout(resolve, this.constructor.INDIVIDUAL_UPLOAD_DELAY));
                        }
                    } catch (error) {
                        console.error(`Failed to upload ${file.name}:`, error);
                        this.showToast(`Failed to upload ${file.name}: ${error.message}`, 'error');
                        // Continue with remaining files
                    }
                }
            } else {
                // Batch upload
                await this.uploadBatchDocuments(this.selectedFiles, {
                    tags,
                    chunk_size: parseInt(chunkSize),
                    chunk_overlap: parseInt(chunkOverlap),
                    memory_type: memoryType
                });
            }

            // Clear selection and reload history
            this.selectedFiles = [];
            this.updateUploadButton();
            await this.loadUploadHistory();

            // Reset drop zone
            const dropZone = document.getElementById('dropZone');
            if (dropZone) {
                const content = dropZone.querySelector('.drop-zone-content');
                if (content) {
                    content.innerHTML = `
                        <svg width="48" height="48" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/>
                        </svg>
                        <h3>Drag & drop files here</h3>
                        <p>or <button id="fileSelectBtn" class="link-button">browse to select files</button></p>
                        <p class="supported-formats">Supported formats: PDF, DOCX, PPTX, TXT, MD, JSON</p>
                        <input type="file" id="fileInput" multiple accept=".pdf,.docx,.pptx,.txt,.md,.json" style="display: none;">
                    `;
                }
                // Re-setup event listeners for the new elements
                this.setupDocumentsEventListeners();
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showToast('Upload failed: ' + error.message, 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Upload single document
     */
    async uploadSingleDocument(file, config) {
        console.log(`Uploading file: ${file.name}, size: ${file.size} bytes`);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('tags', config.tags);
        formData.append('chunk_size', config.chunk_size.toString());
        formData.append('chunk_overlap', config.chunk_overlap.toString());
        formData.append('memory_type', config.memory_type);

        const response = await fetch(`${this.apiBase}/documents/upload`, {
            method: 'POST',
            body: formData
        });

        console.log(`Upload response status: ${response.status}`);

        if (!response.ok) {
            let errorMessage = `Upload failed with status ${response.status}`;
            try {
                const error = await response.json();
                console.error('Upload error details:', error);
                errorMessage = error.detail || error.message || errorMessage;
            } catch (e) {
                console.error('Could not parse error response:', e);
                try {
                    const errorText = await response.text();
                    console.error('Error response text:', errorText);
                    errorMessage = errorText || errorMessage;
                } catch (e2) {
                    console.error('Could not read error response:', e2);
                }
            }
            throw new Error(errorMessage);
        }

        const result = await response.json();
        console.log('Upload result:', result);
        this.showToast(`Upload started for ${file.name}`, 'success');

        // Monitor progress if we have an upload ID
        if (result.upload_id) {
            this.monitorUploadProgress(result.upload_id);
        }

        return result;
    }

    /**
     * Upload batch documents
     */
    async uploadBatchDocuments(files, config) {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file);
        });
        formData.append('tags', config.tags);
        formData.append('chunk_size', config.chunk_size.toString());
        formData.append('chunk_overlap', config.chunk_overlap.toString());
        formData.append('memory_type', config.memory_type);

        const response = await fetch(`${this.apiBase}/documents/batch-upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Batch upload failed');
        }

        const result = await response.json();
        this.showToast(`Batch upload started for ${files.length} files`, 'success');

        // Monitor progress if we have an upload ID
        if (result.upload_id) {
            this.monitorUploadProgress(result.upload_id);
        }

        return result;
    }

    /**
     * Monitor upload progress by polling status endpoint
     */
    monitorUploadProgress(uploadId) {
        const pollStatus = async () => {
            try {
                const statusResponse = await this.apiCall(`/documents/status/${uploadId}`);
                this.updateUploadProgress(uploadId, statusResponse);

                if (statusResponse.progress >= 100 || statusResponse.status === 'completed' || statusResponse.status === 'failed') {
                    // Upload completed, refresh history
                    this.loadUploadHistory();
                } else {
                    // Continue polling
                    setTimeout(pollStatus, 2000); // Poll every 2 seconds
                }
            } catch (error) {
                // If polling fails, try again with longer interval
                setTimeout(pollStatus, 5000);
            }
        };

        // Start polling after a short delay
        setTimeout(pollStatus, 1000);
    }

    /**
     * Update upload progress display
     */
    updateUploadProgress(uploadId, statusData) {
        // Find the upload item in history and update it
        const historyContainer = document.getElementById('uploadHistory');
        if (!historyContainer) return;

        const uploadItems = historyContainer.querySelectorAll('.upload-item');
        uploadItems.forEach(item => {
            const filename = item.querySelector('.upload-filename');
            if (filename && filename.textContent.includes(uploadId)) {
                // This is a simplified update - in practice you'd match by upload ID
                this.loadUploadHistory(); // For now, just refresh the entire history
            }
        });
    }

    /**
     * Check hybrid backend sync status
     */
    async checkSyncStatus() {
        try {
            const syncStatus = await this.apiCall('/sync/status');

            // Only show sync bar for hybrid mode
            const syncBar = document.getElementById('syncStatusBar');
            if (!syncBar) {
                console.warn('Sync status bar element not found');
                return;
            }

            console.log('Sync status:', syncStatus);

            if (!syncStatus.is_hybrid) {
                console.log('Not hybrid mode, hiding sync bar');
                syncBar.classList.remove('visible');
                return;
            }

            // Show sync bar for hybrid mode
            console.log('Hybrid mode detected, showing sync bar');
            syncBar.classList.add('visible');

            // Update sync status UI
            const statusIcon = document.getElementById('syncStatusIcon');
            const statusText = document.getElementById('syncStatusText');
            const statusDetails = document.getElementById('syncStatusDetails');
            const syncButton = document.getElementById('forceSyncButton');

            // Determine status and update UI
            if (syncStatus.status === 'syncing') {
                statusIcon.textContent = '🔄';
                statusText.textContent = 'Syncing...';
                statusDetails.textContent = `${syncStatus.operations_pending} operations pending`;
                syncBar.className = 'sync-status-bar visible syncing';
                syncButton.disabled = true;
            } else if (syncStatus.status === 'pending') {
                statusIcon.textContent = '⏱️';
                statusText.textContent = 'Sync Pending';
                const nextSync = Math.ceil(syncStatus.next_sync_eta_seconds);
                statusDetails.textContent = `${syncStatus.operations_pending} operations • Next sync in ${nextSync}s`;
                syncBar.className = 'sync-status-bar visible pending';
                syncButton.disabled = false;
            } else if (syncStatus.status === 'error') {
                statusIcon.textContent = '⚠️';
                statusText.textContent = 'Sync Error';
                statusDetails.textContent = `${syncStatus.operations_failed} failed operations`;
                syncBar.className = 'sync-status-bar visible error';
                syncButton.disabled = false;
            } else {
                // synced status
                statusIcon.textContent = '✅';
                statusText.textContent = 'Synced';
                const lastSync = Math.floor(syncStatus.time_since_last_sync_seconds);
                statusDetails.textContent = lastSync > 0 ? `Last sync ${lastSync}s ago` : 'Just now';
                syncBar.className = 'sync-status-bar visible synced';
                syncButton.disabled = false;
            }

        } catch (error) {
            console.error('Error checking sync status:', error);
            // Hide sync bar on error (likely not hybrid mode)
            const syncBar = document.getElementById('syncStatusBar');
            if (syncBar) syncBar.style.display = 'none';
        }
    }

    /**
     * Start periodic sync status monitoring
     */
    startSyncStatusMonitoring() {
        // Check sync status every 10 seconds
        setInterval(() => {
            this.checkSyncStatus();
        }, 10000);
    }

    /**
     * Manually force sync to Cloudflare
     */
    async forceSync() {
        const syncButton = document.getElementById('forceSyncButton');
        const originalText = syncButton.innerHTML;

        try {
            // Disable button and show loading state
            syncButton.disabled = true;
            syncButton.innerHTML = '<span class="sync-button-icon">⏳</span><span class="sync-button-text">Syncing...</span>';

            const result = await this.apiCall('/sync/force', 'POST');

            if (result.success) {
                this.showToast(`Synced ${result.operations_synced} operations in ${result.time_taken_seconds}s`, 'success');

                // Refresh dashboard data to show newly synced memories
                if (this.currentView === 'dashboard') {
                    await this.loadDashboardData();
                }
            } else {
                this.showToast('Sync failed: ' + result.message, 'error');
            }

        } catch (error) {
            console.error('Error forcing sync:', error);
            this.showToast('Failed to force sync: ' + error.message, 'error');
        } finally {
            // Re-enable button
            syncButton.disabled = false;
            syncButton.innerHTML = originalText;

            // Refresh sync status immediately
            await this.checkSyncStatus();
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

            // Smooth scroll to results section for better UX
            taggedContainer.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });

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
     * Toggle chunking help section visibility
     */
    toggleChunkingHelp() {
        const helpSection = document.getElementById('chunkingHelpSection');
        if (helpSection) {
            if (helpSection.style.display === 'none') {
                helpSection.style.display = 'block';
            } else {
                helpSection.style.display = 'none';
            }
        }
    }

    /**
     * Hide chunking help section
     */
    hideChunkingHelp() {
        const helpSection = document.getElementById('chunkingHelpSection');
        if (helpSection) {
            helpSection.style.display = 'none';
        }
    }

    /**
     * Toggle overlap help section visibility
     */
    toggleOverlapHelp() {
        const helpSection = document.getElementById('overlapHelpSection');
        if (helpSection) {
            if (helpSection.style.display === 'none') {
                helpSection.style.display = 'block';
            } else {
                helpSection.style.display = 'none';
            }
        }
    }

    /**
     * Hide overlap help section
     */
    hideOverlapHelp() {
        const helpSection = document.getElementById('overlapHelpSection');
        if (helpSection) {
            helpSection.style.display = 'none';
        }
    }

    /**
     * Toggle processing mode help section visibility
     */
    toggleProcessingModeHelp() {
        const helpSection = document.getElementById('processingModeHelpSection');
        if (helpSection) {
            helpSection.style.display = helpSection.style.display === 'block' ? 'none' : 'block';
        }
    }

    /**
     * Hide processing mode help section
     */
    hideProcessingModeHelp() {
        const helpSection = document.getElementById('processingModeHelpSection');
        if (helpSection) {
            helpSection.style.display = 'none';
        }
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
            case 'documents':
                await this.loadDocumentsData();
                break;
            case 'manage':
                await this.loadManageData();
                break;
            case 'analytics':
                await this.loadAnalyticsData();
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
                ...filters
            };

            // Only add similarity_threshold if explicitly set in filters
            if (filters.threshold !== undefined) {
                payload.similarity_threshold = filters.threshold;
            }

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

            // Priority 1: If we have a semantic query, start with semantic search
            if (query) {
                const filters = {};
                if (typeFilter) filters.type = typeFilter;
                results = await this.searchMemories(query, filters);

                // Apply tag filtering to semantic search results if tags are specified
                if (tagFilter && tagFilter.trim()) {
                    const tags = tagFilter.split(',').map(t => t.trim()).filter(t => t);
                    if (tags.length > 0) {
                        results = results.filter(result => {
                            const memoryTags = result.memory.tags || [];
                            // Check if any of the specified tags match memory tags (case-insensitive)
                            return tags.some(filterTag =>
                                memoryTags.some(memoryTag =>
                                    memoryTag.toLowerCase().includes(filterTag.toLowerCase())
                                )
                            );
                        });
                    }
                }
            }
            // Priority 2: Tag-only search (when no semantic query)
            else if (tagFilter && tagFilter.trim()) {
                const tags = tagFilter.split(',').map(t => t.trim()).filter(t => t);

                if (tags.length > 0) {
                    const payload = {
                        tags: tags,
                        match_all: false // ANY match by default
                    };

                    const response = await this.apiCall('/search/by-tag', 'POST', payload);
                    results = response.results || [];

                    // Apply type filter if present
                    if (typeFilter && typeFilter.trim()) {
                        results = results.filter(result => {
                            const memoryType = result.memory.memory_type || 'note';
                            return memoryType === typeFilter;
                        });
                    }
                }
            }
            // Priority 3: Date-based search
            else if (dateFilter && dateFilter.trim()) {
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
            }
            // Priority 4: Type-only filter
            else if (typeFilter && typeFilter.trim()) {
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

    // Group document chunks by upload_id
        const groupedMemories = this.groupMemoriesByUpload(memories);

    container.innerHTML = groupedMemories.map(group => {
    if (group.type === 'document') {
            return this.renderDocumentGroup(group);
            } else {
                return this.renderMemoryCard(group.memory);
            }
        }).join('');

        // Add click handlers for individual memories
        container.querySelectorAll('.memory-card').forEach((card, index) => {
            const group = groupedMemories[index];
            if (group.type === 'single') {
                card.addEventListener('click', () => this.handleMemoryClick(group.memory));
            }
        });

        // Add click handlers for document groups
        container.querySelectorAll('.document-group').forEach((groupEl, index) => {
            const group = groupedMemories.filter(g => g.type === 'document')[index];
            if (group) {
                groupEl.addEventListener('click', (e) => {
                    // Don't trigger if clicking on action buttons
                    if (e.target.closest('.document-actions')) return;
                    this.showDocumentChunks(group);
                });
            }
        });

        // Add click handlers for document action buttons
        container.querySelectorAll('.document-group').forEach((groupEl, index) => {
            const group = groupedMemories.filter(g => g.type === 'document')[index];
            if (group) {
                // View chunks button
                const viewBtn = groupEl.querySelector('.btn-view-chunks');
                if (viewBtn) {
                    viewBtn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        this.showDocumentChunks(group);
                    });
                }

                // Remove button
                const removeBtn = groupEl.querySelector('.btn-remove');
                if (removeBtn) {
                    removeBtn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        await this.removeDocument(group.upload_id, group.source_file);
                        // removeDocument() already handles view refresh
                    });
                }
            }
        });
    }

    /**
     * Group memories by upload_id for document chunks
     */
    groupMemoriesByUpload(memories) {
        const groups = [];
        const documentGroups = new Map();
        const processedHashes = new Set();

        for (const memory of memories) {
            // Check if this is a document chunk
            const isDocumentChunk = memory.metadata && memory.metadata.upload_id;

            if (isDocumentChunk && !processedHashes.has(memory.content_hash)) {
                const uploadId = memory.metadata.upload_id;
                const sourceFile = memory.metadata.source_file || 'Unknown file';

                if (!documentGroups.has(uploadId)) {
                    documentGroups.set(uploadId, {
                        upload_id: uploadId,
                        source_file: sourceFile,
                        memories: [],
                        created_at: memory.created_at,
                        tags: new Set()
                    });
                }

                const group = documentGroups.get(uploadId);
                group.memories.push(memory);
                group.tags.add(...(memory.tags || []));
                processedHashes.add(memory.content_hash);
            } else if (!processedHashes.has(memory.content_hash)) {
                // Regular memory
                groups.push({
                    type: 'single',
                    memory: memory
                });
                processedHashes.add(memory.content_hash);
            }
        }

        // Convert document groups to array format
        for (const group of documentGroups.values()) {
            groups.push({
                type: 'document',
                upload_id: group.upload_id,
                source_file: group.source_file,
                memories: group.memories,
                created_at: group.created_at,
                tags: Array.from(group.tags)
            });
        }

        // Sort by creation time (most recent first)
        groups.sort((a, b) => {
            const timeA = a.type === 'document' ? a.created_at : a.memory.created_at;
            const timeB = b.type === 'document' ? b.created_at : b.memory.created_at;
            return timeB - timeA;
        });

        return groups;
    }

    /**
     * Render a document group card
     */
    renderDocumentGroup(group) {
        const createdDate = new Date(group.created_at * 1000).toLocaleDateString();
        const fileName = this.escapeHtml(group.source_file);
        const chunkCount = group.memories.length;
        // Filter out metadata tags AND tags that are too long (likely corrupted/malformed)
        const uniqueTags = [...new Set(group.tags.filter(tag =>
            !tag.startsWith('upload_id:') &&
            !tag.startsWith('source_file:') &&
            !tag.startsWith('file_type:') &&
            tag.length < 100  // Reject tags longer than 100 chars (likely corrupted metadata)
        ))];

        return `
            <div class="document-group" data-upload-id="${this.escapeHtml(group.upload_id)}">
                <div class="document-header">
                    <div class="document-icon">📄</div>
                    <div class="document-info">
                        <div class="document-title">${fileName}</div>
                        <div class="document-meta">
                            ${chunkCount} chunks • ${createdDate}
                        </div>
                    </div>
                </div>
                <div class="document-preview">
                    ${group.memories[0] ? this.escapeHtml(group.memories[0].content.substring(0, 150)) + (group.memories[0].content.length > 150 ? '...' : '') : 'No content preview available'}
                </div>
                ${uniqueTags.length > 0 ? `
                    <div class="document-tags">
                        ${uniqueTags.slice(0, 3).map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
                        ${uniqueTags.length > 3 ? `<span class="tag more">+${uniqueTags.length - 3} more</span>` : ''}
                    </div>
                ` : ''}
                <div class="document-actions">
                    <button class="btn-icon btn-view-chunks" title="View all chunks">
                        <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                        </svg>
                        <span>View Chunks</span>
                    </button>
                    <button class="btn-icon btn-remove" title="Remove document">
                        <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                        </svg>
                        <span>Remove</span>
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Show document chunks in a modal
     */
    showDocumentChunks(group) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal-content large-modal">
                <div class="modal-header">
                    <h3>📄 ${this.escapeHtml(group.source_file)}</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="document-chunks">
                        ${group.memories.map((memory, index) => `
                            <div class="chunk-item">
                                <div class="chunk-header">
                                    <span class="chunk-number">Chunk ${index + 1}</span>
                                    <div class="chunk-meta">
                                        ${memory.metadata && memory.metadata.page ? `Page ${memory.metadata.page} • ` : ''}
                                        ${memory.content.length} chars
                                    </div>
                                </div>
                                <div class="chunk-content">
                                    ${this.escapeHtml(memory.content)}
                                </div>
                                ${memory.tags && memory.tags.length > 0 ? `
                                    <div class="chunk-tags">
                                        ${memory.tags.map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Add active class to show modal (required for display: flex)
        setTimeout(() => modal.classList.add('active'), 10);

        // Add close handlers
        const closeModal = () => {
            modal.classList.remove('active');
            setTimeout(() => document.body.removeChild(modal), 300);
        };
        modal.querySelector('.modal-close').addEventListener('click', closeModal);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
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

    // Truncate content to 150 characters for preview
    const truncatedContent = memory.content.length > 150
    ? memory.content.substring(0, 150) + '...'
    : memory.content;

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
    ${this.escapeHtml(truncatedContent)}
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
                <button class="remove-filter" data-filter-type="${this.escapeHtml(filter.type)}" data-filter-value="${this.escapeHtml(filter.value)}">
                    ×
                </button>
            </div>
        `).join('');

        // Add event listeners for filter removal
        filtersList.addEventListener('click', (e) => {
            const button = e.target.closest('.remove-filter');
            if (!button) return;

            const type = button.dataset.filterType;
            const value = button.dataset.filterValue;
            this.removeFilter(type, value);
        });
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
     * Handle live search toggle
     */
    handleLiveSearchToggle(event) {
        this.liveSearchEnabled = event.target.checked;
        const modeText = document.getElementById('searchModeText');
        if (modeText) {
            modeText.textContent = this.liveSearchEnabled ? 'Live Search' : 'Manual Search';
        }

        // Show a toast to indicate the mode change
        this.showToast(
            `Search mode: ${this.liveSearchEnabled ? 'Live (searches as you type)' : 'Manual (click Search button)'}`,
            'info'
        );
    }

    /**
     * Handle debounced filter changes for live search
     */
    handleDebouncedFilterChange() {
        // Clear any existing timer
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }

        // Only trigger search if live search is enabled
        if (this.liveSearchEnabled) {
            this.debounceTimer = setTimeout(() => {
                this.handleFilterChange();
            }, 300); // 300ms debounce
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
     * View document memory chunks
     */
    async viewDocumentMemory(uploadId) {
        try {
            this.setLoading(true);
            const response = await this.apiCall(`/documents/search-content/${uploadId}`);

            if (response.status === 'success' || response.status === 'partial') {
                // Show modal
                const modal = document.getElementById('memoryViewerModal');
                const filename = document.getElementById('memoryViewerFilename');
                const stats = document.getElementById('memoryViewerStats');
                const chunksList = document.getElementById('memoryChunksList');

                filename.textContent = response.filename || 'Document';
                stats.textContent = `${response.total_found} chunk${response.total_found !== 1 ? 's' : ''} found`;

                // Render chunks
                if (response.memories && response.memories.length > 0) {
                    const chunksHtml = response.memories.map((memory, index) => {
                        const chunkIndex = memory.chunk_index !== undefined ? memory.chunk_index : index;
                        const page = memory.page ? ` • Page ${memory.page}` : '';
                        const contentPreview = memory.content.length > 300
                            ? memory.content.substring(0, 300) + '...'
                            : memory.content;

                        return `
                            <div class="memory-chunk-item">
                                <div class="chunk-header">
                                    <span class="chunk-number">Chunk ${chunkIndex + 1}${page}</span>
                                    <span class="chunk-hash" title="${memory.content_hash}">${memory.content_hash.substring(0, 12)}...</span>
                                </div>
                                <div class="chunk-content">${this.escapeHtml(contentPreview)}</div>
                                <div class="chunk-tags">
                                    ${memory.tags.map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
                                </div>
                            </div>
                        `;
                    }).join('');

                    chunksList.innerHTML = chunksHtml;
                } else {
                    chunksList.innerHTML = '<p class="text-muted">No memory chunks found for this document.</p>';
                }

                modal.style.display = 'flex';

                if (response.status === 'partial') {
                    this.showToast(`Found ${response.total_found} chunks (partial results)`, 'warning');
                }
            } else {
                this.showToast('Failed to load document memories', 'error');
            }
        } catch (error) {
            console.error('Error viewing document memory:', error);
            this.showToast('Error loading document memories', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Close memory viewer modal
     */
    closeMemoryViewer() {
        const modal = document.getElementById('memoryViewerModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    /**
     * Remove document and its memories
     */
    async removeDocument(uploadId, filename) {
        console.log('removeDocument called with:', { uploadId, filename, currentView: this.currentView });

        if (!confirm(`Remove "${filename}" and all its memory chunks?\n\nThis action cannot be undone.`)) {
            console.log('User cancelled removal');
            return;
        }

        try {
            this.setLoading(true);
            console.log('Making DELETE request to:', `/documents/remove/${uploadId}`);

            const response = await this.apiCall(`/documents/remove/${uploadId}`, 'DELETE');

            console.log('Delete response:', response);

            if (response.status === 'success') {
                this.showToast(`Removed "${filename}" (${response.memories_deleted} memories deleted)`, 'success');
                // Refresh the current view (Dashboard or Documents tab)
                console.log('Refreshing view:', this.currentView);
                if (this.currentView === 'dashboard') {
                    await this.loadDashboardData();
                } else if (this.currentView === 'documents') {
                    await this.loadUploadHistory();
                }
            } else {
                console.error('Removal failed with response:', response);
                this.showToast('Failed to remove document', 'error');
            }
        } catch (error) {
            console.error('Error removing document:', error);
            console.error('Error stack:', error.stack);
            this.showToast('Error removing document', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Search document content
     */
    async searchDocumentContent(query) {
        try {
            this.setLoading(true);
            const resultsContainer = document.getElementById('docSearchResults');
            const resultsList = document.getElementById('docSearchResultsList');
            const resultsCount = document.getElementById('docSearchCount');

            // Use the regular search endpoint but filter for document memories
            // Higher n_results to ensure we get enough document results after filtering
            const response = await this.apiCall('/search', 'POST', {
                query: query,
                n_results: 100
            });

            if (response.results) {
                // Filter results to only show document-type memories
                const documentResults = response.results.filter(r =>
                    r.memory?.memory_type === 'document' || (r.memory?.tags && r.memory.tags.some(tag => tag.startsWith('upload_id:')))
                );

                // Limit display to top 20 most relevant document results
                const displayResults = documentResults.slice(0, 20);

                resultsCount.textContent = `${documentResults.length} result${documentResults.length !== 1 ? 's' : ''}${documentResults.length > 20 ? ' (showing top 20)' : ''}`;

                if (displayResults.length > 0) {
                    const resultsHtml = displayResults.map(result => {
                        const mem = result.memory;
                        const uploadIdTag = mem.tags?.find(tag => tag.startsWith('upload_id:'));
                        const sourceFile = mem.metadata?.source_file || 'Unknown file';
                        const contentPreview = mem.content.length > 200
                            ? mem.content.substring(0, 200) + '...'
                            : mem.content;

                        return `
                            <div class="search-result-item">
                                <div class="result-header">
                                    <strong>${this.escapeHtml(sourceFile)}</strong>
                                    <span class="similarity-score">${Math.round((result.similarity_score || 0) * 100)}% match</span>
                                </div>
                                <div class="result-content">${this.escapeHtml(contentPreview)}</div>
                                <div class="result-tags">
                                    ${(mem.tags || []).slice(0, 5).map(tag => `<span class="tag">${this.escapeHtml(tag)}</span>`).join('')}
                                </div>
                            </div>
                        `;
                    }).join('');

                    resultsList.innerHTML = resultsHtml;
                } else {
                    resultsList.innerHTML = '<p class="text-muted">No matching document content found. Try different search terms.</p>';
                }

                resultsContainer.style.display = 'block';
            } else {
                this.showToast('Search failed', 'error');
            }
        } catch (error) {
            console.error('Error searching documents:', error);
            this.showToast('Error performing search', 'error');
        } finally {
            this.setLoading(false);
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
     * Load settings from localStorage
     */
    loadSettings() {
        try {
            const saved = localStorage.getItem('memoryDashboardSettings');
            if (saved) {
                this.settings = { ...this.settings, ...JSON.parse(saved) };
            }
        } catch (error) {
            console.warn('Failed to load settings:', error);
        }
    }

    /**
     * Save settings to localStorage
     */
    saveSettingsToStorage() {
        try {
            localStorage.setItem('memoryDashboardSettings', JSON.stringify(this.settings));
        } catch (error) {
            console.error('Failed to save settings:', error);
            this.showToast('Failed to save settings. Your preferences will not be persisted.', 'error');
        }
    }

    /**
     * Apply theme to the page
     */
    applyTheme(theme = this.settings.theme) {
        const isDark = theme === 'dark';
        document.body.classList.toggle('dark-mode', isDark);

        // Toggle icon visibility using CSS classes
        const sunIcon = document.getElementById('sunIcon');
        const moonIcon = document.getElementById('moonIcon');
        if (sunIcon && moonIcon) {
            sunIcon.classList.toggle('hidden', isDark);
            moonIcon.classList.toggle('hidden', !isDark);
        }
    }

    /**
     * Toggle between light and dark theme
     */
    toggleTheme() {
        const newTheme = this.settings.theme === 'dark' ? 'light' : 'dark';
        this.settings.theme = newTheme;
        this.applyTheme(newTheme);
        this.saveSettingsToStorage();
        this.showToast(`Switched to ${newTheme} mode`, 'success');
    }

    /**
     * Open settings modal
     */
    async openSettingsModal() {
        const modal = document.getElementById('settingsModal');

        // Populate form with current settings
        document.getElementById('themeSelect').value = this.settings.theme;
        document.getElementById('viewDensity').value = this.settings.viewDensity;
        document.getElementById('previewLines').value = this.settings.previewLines;

        // Reset system info to loading state
        this.resetSystemInfoLoadingState();

        // Load system information
        await this.loadSystemInfo();

        this.openModal(modal);
    }

    /**
     * Reset system info fields to loading state
     */
    resetSystemInfoLoadingState() {
        Object.keys(MemoryDashboard.SYSTEM_INFO_CONFIG).forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = 'Loading...';
            }
        });
    }

    /**
     * Load system information for settings modal
     */
    async loadSystemInfo() {
        try {
            // Use Promise.allSettled for robust error handling
            const [healthResult, detailedHealthResult] = await Promise.allSettled([
                this.apiCall('/health'),
                this.apiCall('/health/detailed')
            ]);

            const apiData = {
                health: healthResult.status === 'fulfilled' ? healthResult.value : null,
                detailedHealth: detailedHealthResult.status === 'fulfilled' ? detailedHealthResult.value : null
            };

            // Update fields using configuration
            Object.entries(MemoryDashboard.SYSTEM_INFO_CONFIG).forEach(([fieldId, config]) => {
                const element = document.getElementById(fieldId);
                if (!element) return;

                let value = null;
                for (const source of config.sources) {
                    const apiResponse = apiData[source.api];
                    if (apiResponse) {
                        value = this.getNestedValue(apiResponse, source.path);
                        if (value !== undefined && value !== null) break;
                    }
                }

                element.textContent = config.formatter(value);
            });

            // Log warnings for failed API calls
            if (healthResult.status === 'rejected') {
                console.warn('Failed to load health endpoint:', healthResult.reason);
            }
            if (detailedHealthResult.status === 'rejected') {
                console.warn('Failed to load detailed health endpoint:', detailedHealthResult.reason);
            }
        } catch (error) {
            console.error('Unexpected error loading system info:', error);
            // Set all system info fields that are still in loading state to error
            Object.keys(MemoryDashboard.SYSTEM_INFO_CONFIG).forEach(id => {
                const element = document.getElementById(id);
                if (element && element.textContent === 'Loading...') {
                    element.textContent = 'Error';
                }
            });
        }
    }

    /**
     * Get nested object value by path string
     * @param {Object} obj - Object to traverse
     * @param {string} path - Dot-separated path (e.g., 'storage.primary_stats.embedding_model')
     * @returns {*} Value at path or undefined
     */
    getNestedValue(obj, path) {
        return path.split('.').reduce((current, key) => current?.[key], obj);
    }

    /**
     * Format uptime seconds into human readable string
     * @param {number} seconds - Uptime in seconds
     * @returns {string} Formatted uptime string
     */
    static formatUptime(seconds) {
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);

        const parts = [];
        if (days > 0) parts.push(`${days}d`);
        if (hours > 0) parts.push(`${hours}h`);
        if (minutes > 0) parts.push(`${minutes}m`);

        return parts.length > 0 ? parts.join(' ') : '< 1m';
    }

    /**
     * Save settings from modal
     */
    saveSettings() {
        // Get values from form
        const theme = document.getElementById('themeSelect').value;
        const viewDensity = document.getElementById('viewDensity').value;
        const previewLines = parseInt(document.getElementById('previewLines').value, 10);

        // Update settings
        this.settings.theme = theme;
        this.settings.viewDensity = viewDensity;
        this.settings.previewLines = previewLines;

        // Apply changes
        this.applyTheme(theme);
        this.saveSettingsToStorage();

        // Close modal and show confirmation
        this.closeModal(document.getElementById('settingsModal'));
        this.showToast('Settings saved successfully', 'success');
    }

    // ===== MANAGE TAB METHODS =====

    /**
     * Load manage tab data
    */
    async loadManageData() {
    try {
            // Load tag statistics for bulk operations
            await this.loadTagSelectOptions();
            await this.loadTagManagementStats();
        } catch (error) {
            console.error('Failed to load manage data:', error);
            this.showToast('Failed to load management data', 'error');
        }
    }

    /**
     * Load tag options for bulk delete select
     */
    async loadTagSelectOptions() {
        try {
            const response = await fetch(`${this.apiBase}/manage/tags/stats`);
            if (!response.ok) throw new Error('Failed to load tags');

            const data = await response.json();
            const select = document.getElementById('deleteTagSelect');
            if (!select) return;

            // Clear existing options except the first
            while (select.children.length > 1) {
                select.removeChild(select.lastChild);
            }

            // Add tag options
            data.tags.forEach(tagStat => {
                const option = document.createElement('option');
                option.value = tagStat.tag;
                option.textContent = `${tagStat.tag} (${tagStat.count} memories)`;
                option.dataset.count = tagStat.count;  // Store count in data attribute
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load tag options:', error);
        }
    }

    /**
     * Load tag management statistics
     */
    async loadTagManagementStats() {
        const container = document.getElementById('tagManagementContainer');
        if (!container) return;

        try {
            const response = await fetch(`${this.apiBase}/manage/tags/stats`);
            if (!response.ok) throw new Error('Failed to load tag stats');

            const data = await response.json();
            this.renderTagManagementTable(data);
        } catch (error) {
            console.error('Failed to load tag management stats:', error);
            container.innerHTML = '<p class="error">Failed to load tag statistics</p>';
        }
    }

    /**
     * Render tag management table
     */
    renderTagManagementTable(data) {
        const container = document.getElementById('tagManagementContainer');
        if (!container) return;

        let html = '<table class="tag-stats-table">';
        html += '<thead><tr>';
        html += '<th>Tag</th>';
        html += '<th>Count</th>';
        html += '<th>Actions</th>';
        html += '</tr></thead><tbody>';

        data.tags.forEach(tagStat => {
        html += '<tr>';
        html += `<td class="tag-name">${tagStat.tag}</td>`;
        html += `<td class="tag-count">${tagStat.count}</td>`;
        html += '<td class="tag-actions">';
        html += `<button class="tag-action-btn" data-action="rename-tag" data-tag="${this.escapeHtml(tagStat.tag)}">Rename</button>`;
        html += `<button class="tag-action-btn danger" data-action="delete-tag" data-tag="${this.escapeHtml(tagStat.tag)}" data-count="${tagStat.count}">Delete</button>`;
        html += '</td></tr>';
        });

        html += '</tbody></table>';
        container.innerHTML = html;

        // Add event listeners for tag actions
        container.addEventListener('click', (e) => {
            const button = e.target.closest('[data-action]');
            if (!button) return;

            const action = button.dataset.action;
            const tag = button.dataset.tag;

            if (action === 'rename-tag') {
                this.renameTag(tag);
            } else if (action === 'delete-tag') {
                const count = parseInt(button.dataset.count, 10);
                this.deleteTag(tag, count);
            }
        });
    }

    /**
     * Handle bulk delete by tag
     */
    async handleBulkDeleteByTag() {
        const select = document.getElementById('deleteTagSelect');
        const tag = select.value;

        if (!tag) {
            this.showToast('Please select a tag to delete', 'warning');
            return;
        }

        // Extract count from data attribute
        const option = select.querySelector(`option[value="${tag}"]`);
        const count = parseInt(option.dataset.count, 10) || 0;

        if (!await this.confirmBulkOperation(`Delete ${count} memories with tag "${tag}"?`)) {
            return;
        }

        this.setLoading(true);
        try {
            const response = await fetch(`${this.apiBase}/manage/bulk-delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    tag: tag,
                    confirm_count: count
                })
            });

            const result = await response.json();
            if (result.success) {
                this.showToast(result.message, 'success');
                await this.loadManageData(); // Refresh data
                await this.loadDashboardData(); // Refresh dashboard stats
            } else {
                this.showToast(result.message, 'error');
            }
        } catch (error) {
            console.error('Bulk delete failed:', error);
            this.showToast('Bulk delete operation failed', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Handle cleanup duplicates
     */
    async handleCleanupDuplicates() {
        if (!await this.confirmBulkOperation('Remove all duplicate memories?')) {
            return;
        }

        this.setLoading(true);
        try {
            const response = await fetch(`${this.apiBase}/manage/cleanup-duplicates`, {
                method: 'POST'
            });

            const result = await response.json();
            if (result.success) {
                this.showToast(result.message, 'success');
                await this.loadManageData();
                await this.loadDashboardData();
            } else {
                this.showToast(result.message, 'error');
            }
        } catch (error) {
            console.error('Cleanup duplicates failed:', error);
            this.showToast('Cleanup operation failed', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Handle bulk delete by date
     */
    async handleBulkDeleteByDate() {
        const dateInput = document.getElementById('deleteDateInput');
        const date = dateInput.value;

        if (!date) {
            this.showToast('Please select a date', 'warning');
            return;
        }

        if (!await this.confirmBulkOperation(`Delete all memories before ${date}?`)) {
            return;
        }

        this.setLoading(true);
        try {
            const response = await fetch(`${this.apiBase}/manage/bulk-delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    before_date: date
                })
            });

            const result = await response.json();
            if (result.success) {
                this.showToast(result.message, 'success');
                await this.loadManageData();
                await this.loadDashboardData();
            } else {
                this.showToast(result.message, 'error');
            }
        } catch (error) {
            console.error('Bulk delete by date failed:', error);
            this.showToast('Bulk delete operation failed', 'error');
        } finally {
            this.setLoading(false);
        }
    }

    /**
     * Handle database optimization
     */
    async handleOptimizeDatabase() {
        this.showToast('Database optimization not yet implemented', 'warning');
    }

    /**
     * Handle index rebuild
     */
    async handleRebuildIndex() {
        this.showToast('Index rebuild not yet implemented', 'warning');
    }

    /**
     * Rename a tag
     */
    async renameTag(oldTag) {
        const newTag = prompt(`Rename tag "${oldTag}" to:`, oldTag);
        if (!newTag || newTag === oldTag) return;

        this.showToast('Tag renaming not yet implemented', 'warning');
    }

    /**
     * Delete a tag
     */
    async deleteTag(tag, count) {
        if (!await this.confirmBulkOperation(`Delete tag "${tag}" from ${count} memories?`)) {
            return;
        }

        this.showToast('Tag deletion not yet implemented', 'warning');
    }

    // ===== ANALYTICS TAB METHODS =====

    /**
     * Load analytics tab data
     */
    async loadAnalyticsData() {
        try {
            await Promise.all([
                this.loadAnalyticsOverview(),
                this.loadMemoryGrowthChart(),
                this.loadTagUsageChart(),
                this.loadMemoryTypesChart(),
                this.loadTopTagsReport(),
                this.loadRecentActivityReport()
            ]);
        } catch (error) {
            console.error('Failed to load analytics data:', error);
            this.showToast('Failed to load analytics data', 'error');
        }
    }

    /**
     * Load analytics overview metrics
     */
    async loadAnalyticsOverview() {
        try {
            const response = await fetch(`${this.apiBase}/analytics/overview`);
            if (!response.ok) throw new Error('Failed to load overview');

            const data = await response.json();

            // Update metric cards
            this.updateElementText('analyticsTotalMemories', data.total_memories || 0);
            this.updateElementText('analyticsThisWeek', data.memories_this_week || 0);
            this.updateElementText('analyticsUniqueTags', data.unique_tags || 0);
            this.updateElementText('analyticsDbSize', data.database_size_mb ?
                `${data.database_size_mb.toFixed(1)} MB` : 'N/A');
        } catch (error) {
            console.error('Failed to load analytics overview:', error);
        }
    }

    /**
     * Load memory growth chart
     */
    async loadMemoryGrowthChart() {
        const container = document.getElementById('memoryGrowthChart');
        const period = document.getElementById('growthPeriodSelect').value;

        if (!container) return;

        try {
            const response = await fetch(`${this.apiBase}/analytics/memory-growth?period=${period}`);
            if (!response.ok) throw new Error('Failed to load growth data');

            const data = await response.json();
            this.renderMemoryGrowthChart(container, data);
        } catch (error) {
            console.error('Failed to load memory growth:', error);
            container.innerHTML = '<p class="error">Failed to load growth chart</p>';
        }
    }

    /**
     * Render memory growth chart
     */
    renderMemoryGrowthChart(container, data) {
        if (!data.data_points || data.data_points.length === 0) {
            container.innerHTML = '<p>No growth data available</p>';
            return;
        }

        // Simple text-based chart for now (could be enhanced with a charting library)
        let html = '<div class="simple-chart">';
        html += '<div class="chart-header">Daily Memory Growth</div>';

        data.data_points.slice(-10).forEach(point => { // Show last 10 days
            const barWidth = Math.max(point.count * 5, 1); // Scale bars
            html += `<div class="chart-row">
                <span class="chart-label">${point.date}</span>
                <div class="chart-bar" style="width: ${barWidth}px"></div>
                <span class="chart-value">${point.count}</span>
            </div>`;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Load tag usage chart
     */
    async loadTagUsageChart() {
        const container = document.getElementById('tagUsageChart');
        if (!container) return;

        try {
            const response = await fetch(`${this.apiBase}/analytics/tag-usage`);
            if (!response.ok) throw new Error('Failed to load tag usage');

            const data = await response.json();
            this.renderTagUsageChart(container, data);
        } catch (error) {
            console.error('Failed to load tag usage:', error);
            container.innerHTML = '<p class="error">Failed to load tag usage chart</p>';
        }
    }

    /**
     * Render tag usage chart
     */
    renderTagUsageChart(container, data) {
        if (!data.tags || data.tags.length === 0) {
            container.innerHTML = '<p>No tags found</p>';
            return;
        }

        let html = '<div class="simple-chart">';
        html += '<div class="chart-header">Tag Usage Distribution</div>';

        data.tags.slice(0, 10).forEach(tag => { // Top 10 tags
            const barWidth = Math.max(tag.percentage * 3, 1); // Scale bars
            html += `<div class="chart-row">
                <span class="chart-label">${tag.tag}</span>
                <div class="chart-bar" style="width: ${barWidth}px"></div>
                <span class="chart-value">${tag.count} (${tag.percentage}%)</span>
            </div>`;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Load memory types chart
     */
    async loadMemoryTypesChart() {
        const container = document.getElementById('memoryTypesChart');
        if (!container) return;

        try {
            const response = await fetch(`${this.apiBase}/analytics/memory-types`);
            if (!response.ok) throw new Error('Failed to load memory types');

            const data = await response.json();
            this.renderMemoryTypesChart(container, data);
        } catch (error) {
            console.error('Failed to load memory types:', error);
            container.innerHTML = '<p class="error">Failed to load memory types chart</p>';
        }
    }

    /**
     * Render memory types chart
     */
    renderMemoryTypesChart(container, data) {
        if (!data.types || data.types.length === 0) {
            container.innerHTML = '<p>No memory types found</p>';
            return;
        }

        let html = '<div class="simple-chart">';
        html += '<div class="chart-header">Memory Types</div>';

        data.types.forEach(type => {
            const barWidth = Math.max(type.percentage * 3, 1); // Scale bars
            html += `<div class="chart-row">
                <span class="chart-label">${type.memory_type || 'untyped'}</span>
                <div class="chart-bar" style="width: ${barWidth}px"></div>
                <span class="chart-value">${type.count} (${type.percentage}%)</span>
            </div>`;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    /**
     * Load top tags report
     */
    async loadTopTagsReport() {
        const container = document.getElementById('topTagsList');
        if (!container) return;

        try {
            const response = await fetch(`${this.apiBase}/analytics/tag-usage`);
            if (!response.ok) throw new Error('Failed to load top tags');

            const data = await response.json();
            this.renderTopTagsReport(container, data);
        } catch (error) {
            console.error('Failed to load top tags:', error);
            container.innerHTML = '<p class="error">Failed to load top tags</p>';
        }
    }

    /**
     * Render top tags report
     */
    renderTopTagsReport(container, data) {
        if (!data.tags || data.tags.length === 0) {
            container.innerHTML = '<p>No tags found</p>';
            return;
        }

        let html = '<ul class="tags-list">';
        data.tags.slice(0, 10).forEach(tag => {
            html += `<li><strong>${tag.tag}</strong>: ${tag.count} memories (${tag.percentage}%)</li>`;
        });
        html += '</ul>';

        container.innerHTML = html;
    }

    /**
     * Load recent activity report
     */
    async loadRecentActivityReport() {
        const container = document.getElementById('recentActivityList');
        if (!container) return;

        try {
            // Get recent memories
            const response = await fetch(`${this.apiBase}/memories?page=1&page_size=5`);
            if (!response.ok) throw new Error('Failed to load recent activity');

            const data = await response.json();
            this.renderRecentActivityReport(container, data.memories);
        } catch (error) {
            console.error('Failed to load recent activity:', error);
            container.innerHTML = '<p class="error">Failed to load recent activity</p>';
        }
    }

    /**
     * Render recent activity report
     */
    renderRecentActivityReport(container, memories) {
        if (!memories || memories.length === 0) {
            container.innerHTML = '<p>No recent activity</p>';
            return;
        }

        let html = '<ul class="activity-list">';
        memories.forEach(memory => {
            const date = memory.created_at_iso ?
                new Date(memory.created_at_iso).toLocaleDateString() : 'Unknown';
            const preview = memory.content.substring(0, 50) + (memory.content.length > 50 ? '...' : '');
            html += `<li><strong>${date}</strong>: ${preview}</li>`;
        });
        html += '</ul>';

        container.innerHTML = html;
    }

    /**
     * Handle growth period change
     */
    async handleGrowthPeriodChange() {
        await this.loadMemoryGrowthChart();
    }

    // ===== UTILITY METHODS =====

    /**
     * Show confirmation dialog for bulk operations
     */
    async confirmBulkOperation(message) {
        return confirm(`⚠️ WARNING: ${message}

This action cannot be undone. Are you sure?`);
    }

    /**
     * Update element text content
     */
    updateElementText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        }
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