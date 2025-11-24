# Issue #240 Phase 2: Low-Hanging Complexity Reductions

## Executive Summary

**Current State:**
- Overall Health: 63/100 (Grade C)
- Cyclomatic Complexity Score: 40/100
- Average complexity: 9.5
- High-risk functions (>7): 28 functions
- Maximum complexity: 62 (install.py::main)

**Phase 2 Goals:**
- Target functions: 5 main targets (complexity 10-15) + 5 quick wins
- Target complexity improvement: +10-15 points (40 → 50-55)
- Expected overall health improvement: +3 points (63 → 66-68)
- Strategy: Extract methods, guard clauses, dict lookups (no architectural changes)

**Total Estimated Effort:** 12-15 hours

**Functions Analyzed:** 5 target functions + 5 quick wins

---

## Target Function 1: install.py::configure_paths() (Complexity: 15)

### Current Implementation
**Purpose:** Configure storage paths for memory service based on platform and backend type.

**Location:** Lines 1287-1445 (158 lines)

### Complexity Breakdown
```
Lines 1287-1306: +3 complexity (platform-specific path detection)
Lines 1306-1347: +5 complexity (storage backend conditional setup)
Lines 1349-1358: +2 complexity (backup directory test with error handling)
Lines 1359-1443: +5 complexity (Claude Desktop config update nested logic)
Total Base: 15
```

**Primary Contributors:**
1. Platform detection branching (macOS/Windows/Linux) - 3 branches
2. Storage backend type branching (sqlite_vec/hybrid/cloudflare/chromadb) - 4 branches
3. Nested Claude config file discovery and JSON manipulation
4. Error handling for directory creation and file operations

### Refactoring Proposal #1: Extract Platform Path Detection
**Risk:** Low | **Impact:** -3 complexity | **Time:** 1 hour

**Before:**
```python
def configure_paths(args):
    print_step("4", "Configuring paths")
    system_info = detect_system()
    home_dir = Path.home()

    # Determine base directory based on platform
    if platform.system() == 'Darwin':  # macOS
        base_dir = home_dir / 'Library' / 'Application Support' / 'mcp-memory'
    elif platform.system() == 'Windows':  # Windows
        base_dir = Path(os.environ.get('LOCALAPPDATA', '')) / 'mcp-memory'
    else:  # Linux and others
        base_dir = home_dir / '.local' / 'share' / 'mcp-memory'

    storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'chromadb')
    ...
```

**After:**
```python
def get_platform_base_dir() -> Path:
    """Get platform-specific base directory for MCP Memory storage.

    Returns:
        Path: Platform-appropriate base directory
    """
    home_dir = Path.home()

    PLATFORM_PATHS = {
        'Darwin': home_dir / 'Library' / 'Application Support' / 'mcp-memory',
        'Windows': Path(os.environ.get('LOCALAPPDATA', '')) / 'mcp-memory',
    }

    system = platform.system()
    return PLATFORM_PATHS.get(system, home_dir / '.local' / 'share' / 'mcp-memory')

def configure_paths(args):
    print_step("4", "Configuring paths")
    system_info = detect_system()
    base_dir = get_platform_base_dir()
    storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'chromadb')
    ...
```

**Complexity Impact:** 15 → 12 (-3)
- Removes platform branching from main function
- Uses dict lookup instead of if/elif/else chain

### Refactoring Proposal #2: Extract Storage Path Setup
**Risk:** Low | **Impact:** -4 complexity | **Time:** 1.5 hours

**Before:**
```python
def configure_paths(args):
    ...
    if storage_backend in ['sqlite_vec', 'hybrid', 'cloudflare']:
        storage_path = args.chroma_path or (base_dir / 'sqlite_vec.db')
        storage_dir = storage_path.parent if storage_path.name.endswith('.db') else storage_path
        backups_path = args.backups_path or (base_dir / 'backups')

        try:
            os.makedirs(storage_dir, exist_ok=True)
            os.makedirs(backups_path, exist_ok=True)
            print_info(f"SQLite-vec database: {storage_path}")
            print_info(f"Backups path: {backups_path}")

            # Test if directory is writable
            test_file = os.path.join(storage_dir, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            print_error(f"Failed to configure SQLite-vec paths: {e}")
            return False
    else:
        chroma_path = args.chroma_path or (base_dir / 'chroma_db')
        backups_path = args.backups_path or (base_dir / 'backups')
        storage_path = chroma_path
        ...
```

**After:**
```python
def setup_storage_directories(backend: str, base_dir: Path, args) -> Tuple[Path, Path, bool]:
    """Setup storage and backup directories for the specified backend.

    Args:
        backend: Storage backend type
        base_dir: Base directory for storage
        args: Command line arguments

    Returns:
        Tuple of (storage_path, backups_path, success)
    """
    if backend in ['sqlite_vec', 'hybrid', 'cloudflare']:
        storage_path = args.chroma_path or (base_dir / 'sqlite_vec.db')
        storage_dir = storage_path.parent if storage_path.name.endswith('.db') else storage_path
    else:  # chromadb
        storage_path = args.chroma_path or (base_dir / 'chroma_db')
        storage_dir = storage_path

    backups_path = args.backups_path or (base_dir / 'backups')

    try:
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(backups_path, exist_ok=True)

        # Test writability
        test_file = storage_dir / '.write_test'
        test_file.write_text('test')
        test_file.unlink()

        print_info(f"Storage path: {storage_path}")
        print_info(f"Backups path: {backups_path}")
        return storage_path, backups_path, True

    except Exception as e:
        print_error(f"Failed to configure storage paths: {e}")
        return storage_path, backups_path, False

def configure_paths(args):
    print_step("4", "Configuring paths")
    system_info = detect_system()
    base_dir = get_platform_base_dir()
    storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'chromadb')

    storage_path, backups_path, success = setup_storage_directories(
        storage_backend, base_dir, args
    )
    if not success:
        print_warning("Continuing with Claude Desktop configuration despite storage setup failure")
    ...
```

**Complexity Impact:** 12 → 8 (-4)
- Removes nested storage backend setup logic
- Early return pattern for error handling

### Refactoring Proposal #3: Extract Claude Config Update
**Risk:** Medium | **Impact:** -3 complexity | **Time:** 1.5 hours

**Before:**
```python
def configure_paths(args):
    ...
    # Configure Claude Desktop if available
    import json
    claude_config_paths = [...]

    for config_path in claude_config_paths:
        if config_path.exists():
            print_info(f"Found Claude Desktop config at {config_path}")
            try:
                config_text = config_path.read_text()
                config = json.loads(config_text)

                # Validate config structure
                if not isinstance(config, dict):
                    print_warning(f"Invalid config format...")
                    continue

                # Update or add MCP Memory configuration
                if 'mcpServers' not in config:
                    config['mcpServers'] = {}

                # Create environment configuration based on storage backend
                env_config = {...}

                if storage_backend in ['sqlite_vec', 'hybrid']:
                    env_config["MCP_MEMORY_SQLITE_PATH"] = str(storage_path)
                    ...
```

**After:**
```python
def build_mcp_env_config(storage_backend: str, storage_path: Path,
                        backups_path: Path) -> Dict[str, str]:
    """Build MCP environment configuration for Claude Desktop.

    Args:
        storage_backend: Type of storage backend
        storage_path: Path to storage directory/file
        backups_path: Path to backups directory

    Returns:
        Dict of environment variables for MCP configuration
    """
    env_config = {
        "MCP_MEMORY_BACKUPS_PATH": str(backups_path),
        "MCP_MEMORY_STORAGE_BACKEND": storage_backend
    }

    if storage_backend in ['sqlite_vec', 'hybrid']:
        env_config["MCP_MEMORY_SQLITE_PATH"] = str(storage_path)
        env_config["MCP_MEMORY_SQLITE_PRAGMAS"] = "busy_timeout=15000,cache_size=20000"

    if storage_backend in ['hybrid', 'cloudflare']:
        cloudflare_vars = [
            'CLOUDFLARE_API_TOKEN',
            'CLOUDFLARE_ACCOUNT_ID',
            'CLOUDFLARE_D1_DATABASE_ID',
            'CLOUDFLARE_VECTORIZE_INDEX'
        ]
        for var in cloudflare_vars:
            value = os.environ.get(var)
            if value:
                env_config[var] = value

    if storage_backend == 'chromadb':
        env_config["MCP_MEMORY_CHROMA_PATH"] = str(storage_path)

    return env_config

def update_claude_config_file(config_path: Path, env_config: Dict[str, str],
                              project_root: Path, is_windows: bool) -> bool:
    """Update Claude Desktop configuration file with MCP Memory settings.

    Args:
        config_path: Path to Claude config file
        env_config: Environment configuration dictionary
        project_root: Root directory of the project
        is_windows: Whether running on Windows

    Returns:
        bool: True if update succeeded
    """
    try:
        config_text = config_path.read_text()
        config = json.loads(config_text)

        if not isinstance(config, dict):
            print_warning(f"Invalid config format in {config_path}")
            return False

        if 'mcpServers' not in config:
            config['mcpServers'] = {}

        # Create server configuration
        if is_windows:
            script_path = str((project_root / "memory_wrapper.py").resolve())
            config['mcpServers']['memory'] = {
                "command": "python",
                "args": [script_path],
                "env": env_config
            }
        else:
            config['mcpServers']['memory'] = {
                "command": "uv",
                "args": ["--directory", str(project_root.resolve()), "run", "memory"],
                "env": env_config
            }

        config_path.write_text(json.dumps(config, indent=2))
        print_success("Updated Claude Desktop configuration")
        return True

    except (OSError, PermissionError, json.JSONDecodeError) as e:
        print_warning(f"Failed to update Claude Desktop configuration: {e}")
        return False

def configure_paths(args):
    print_step("4", "Configuring paths")
    system_info = detect_system()
    base_dir = get_platform_base_dir()
    storage_backend = os.environ.get('MCP_MEMORY_STORAGE_BACKEND', 'chromadb')

    storage_path, backups_path, success = setup_storage_directories(
        storage_backend, base_dir, args
    )
    if not success:
        print_warning("Continuing with Claude Desktop configuration")

    # Configure Claude Desktop
    env_config = build_mcp_env_config(storage_backend, storage_path, backups_path)
    project_root = Path(__file__).parent.parent.parent

    claude_config_paths = [
        Path.home() / 'Library' / 'Application Support' / 'Claude' / 'claude_desktop_config.json',
        Path.home() / '.config' / 'Claude' / 'claude_desktop_config.json',
        Path('claude_config') / 'claude_desktop_config.json'
    ]

    for config_path in claude_config_paths:
        if config_path.exists():
            print_info(f"Found Claude Desktop config at {config_path}")
            if update_claude_config_file(config_path, env_config, project_root,
                                        system_info["is_windows"]):
                break

    return True
```

**Complexity Impact:** 8 → 5 (-3)
- Removes nested config update logic
- Separates env config building from file I/O
- Early return pattern in update function

### Implementation Plan
1. **Extract platform detection** (1 hour, low risk) - Simple dict lookup
2. **Extract storage setup** (1.5 hours, low risk) - Straightforward extraction
3. **Extract Claude config** (1.5 hours, medium risk) - Requires careful testing

**Total Complexity Reduction:** 15 → 5 (-10 points)
**Total Time:** 4 hours

---

## Target Function 2: cloudflare.py::_execute_batch() (Complexity: 14)

### Current Implementation
**Purpose:** Execute batched D1 SQL queries with retry logic.

**Note:** After examining the cloudflare.py file, I found that `_execute_batch()` does not exist. The complexity report may be outdated or the function was refactored. Instead, I'll analyze `_search_by_tags_internal()` which shows similar complexity patterns (lines 583-667, complexity ~13).

### Complexity Breakdown (\_search_by_tags_internal)
```
Lines 590-610: +4 complexity (tag normalization and operation validation)
Lines 612-636: +5 complexity (SQL query construction with time filtering)
Lines 638-667: +4 complexity (result processing with error handling)
Total: 13
```

### Refactoring Proposal #1: Extract Tag Normalization
**Risk:** Low | **Impact:** -2 complexity | **Time:** 45 minutes

**Before:**
```python
async def _search_by_tags_internal(self, tags, operation=None, time_start=None, time_end=None):
    try:
        if not tags:
            return []

        # Normalize tags (deduplicate, drop empty strings)
        deduped_tags = list(dict.fromkeys([tag for tag in tags if tag]))
        if not deduped_tags:
            return []

        if isinstance(operation, str):
            normalized_operation = operation.strip().upper() or "AND"
        else:
            normalized_operation = "AND"

        if normalized_operation not in {"AND", "OR"}:
            logger.warning("Unsupported tag search operation '%s'; defaulting to AND", operation)
            normalized_operation = "AND"
```

**After:**
```python
def normalize_tags_for_search(tags: List[str]) -> List[str]:
    """Deduplicate and filter empty tag strings.

    Args:
        tags: List of tag strings (may contain duplicates or empty strings)

    Returns:
        Deduplicated list of non-empty tags
    """
    return list(dict.fromkeys([tag for tag in tags if tag]))

def normalize_operation(operation: Optional[str]) -> str:
    """Normalize tag search operation to AND or OR.

    Args:
        operation: Raw operation string (case-insensitive)

    Returns:
        Normalized operation: "AND" or "OR"
    """
    if isinstance(operation, str):
        normalized = operation.strip().upper() or "AND"
    else:
        normalized = "AND"

    if normalized not in {"AND", "OR"}:
        logger.warning(f"Unsupported operation '{operation}'; defaulting to AND")
        normalized = "AND"

    return normalized

async def _search_by_tags_internal(self, tags, operation=None, time_start=None, time_end=None):
    try:
        if not tags:
            return []

        deduped_tags = normalize_tags_for_search(tags)
        if not deduped_tags:
            return []

        normalized_operation = normalize_operation(operation)
```

**Complexity Impact:** 13 → 11 (-2)

### Refactoring Proposal #2: Extract SQL Query Builder
**Risk:** Low | **Impact:** -3 complexity | **Time:** 1 hour

**Before:**
```python
async def _search_by_tags_internal(self, tags, operation=None, time_start=None, time_end=None):
    ...
    placeholders = ",".join(["?"] * len(deduped_tags))
    params: List[Any] = list(deduped_tags)

    sql = (
        "SELECT m.* FROM memories m "
        "JOIN memory_tags mt ON m.id = mt.memory_id "
        "JOIN tags t ON mt.tag_id = t.id "
        f"WHERE t.name IN ({placeholders})"
    )

    if time_start is not None:
        sql += " AND m.created_at >= ?"
        params.append(time_start)
    if time_end is not None:
        sql += " AND m.created_at <= ?"
        params.append(time_end)

    sql += " GROUP BY m.id"

    if normalized_operation == "AND":
        sql += " HAVING COUNT(DISTINCT t.name) = ?"
        params.append(len(deduped_tags))

    sql += " ORDER BY m.created_at DESC"
```

**After:**
```python
def build_tag_search_query(tags: List[str], operation: str,
                          time_start: Optional[float] = None,
                          time_end: Optional[float] = None) -> Tuple[str, List[Any]]:
    """Build SQL query for tag-based search with time filtering.

    Args:
        tags: List of deduplicated tags
        operation: Search operation ("AND" or "OR")
        time_start: Optional start timestamp filter
        time_end: Optional end timestamp filter

    Returns:
        Tuple of (sql_query, parameters_list)
    """
    placeholders = ",".join(["?"] * len(tags))
    params: List[Any] = list(tags)

    sql = (
        "SELECT m.* FROM memories m "
        "JOIN memory_tags mt ON m.id = mt.memory_id "
        "JOIN tags t ON mt.tag_id = t.id "
        f"WHERE t.name IN ({placeholders})"
    )

    if time_start is not None:
        sql += " AND m.created_at >= ?"
        params.append(time_start)

    if time_end is not None:
        sql += " AND m.created_at <= ?"
        params.append(time_end)

    sql += " GROUP BY m.id"

    if operation == "AND":
        sql += " HAVING COUNT(DISTINCT t.name) = ?"
        params.append(len(tags))

    sql += " ORDER BY m.created_at DESC"

    return sql, params

async def _search_by_tags_internal(self, tags, operation=None, time_start=None, time_end=None):
    try:
        if not tags:
            return []

        deduped_tags = normalize_tags_for_search(tags)
        if not deduped_tags:
            return []

        normalized_operation = normalize_operation(operation)
        sql, params = build_tag_search_query(deduped_tags, normalized_operation,
                                            time_start, time_end)
```

**Complexity Impact:** 11 → 8 (-3)

### Implementation Plan
1. **Extract tag normalization** (45 min, low risk) - Pure functions, easy to test
2. **Extract SQL builder** (1 hour, low risk) - Testable without database

**Total Complexity Reduction:** 13 → 8 (-5 points)
**Total Time:** 1.75 hours

---

## Target Function 3: consolidator.py::_compress_redundant_memories() (Complexity: 13)

### Current Implementation
**Purpose:** Identify and compress semantically similar memory clusters.

**Note:** After examining consolidator.py (556 lines), I found that `_compress_redundant_memories()` does not exist in the current codebase. The function was likely refactored into the consolidation pipeline. The most complex function in this file is `consolidate()` at lines 80-210 (complexity ~12).

### Complexity Breakdown (consolidate method)
```
Lines 99-110: +2 complexity (hybrid backend sync pause logic)
Lines 112-120: +2 complexity (memory retrieval and validation)
Lines 125-150: +4 complexity (association discovery conditional logic)
Lines 155-181: +4 complexity (compression and forgetting conditional logic)
Total: 12
```

### Refactoring Proposal #1: Extract Hybrid Sync Management
**Risk:** Low | **Impact:** -2 complexity | **Time:** 45 minutes

**Before:**
```python
async def consolidate(self, time_horizon: str, **kwargs) -> ConsolidationReport:
    ...
    # Check if hybrid backend and pause sync during consolidation
    sync_was_paused = False
    is_hybrid = hasattr(self.storage, 'pause_sync') and hasattr(self.storage, 'resume_sync')

    try:
        self.logger.info(f"Starting {time_horizon} consolidation...")

        # Pause hybrid sync to eliminate bottleneck during metadata updates
        if is_hybrid:
            self.logger.info("Pausing hybrid backend sync during consolidation")
            await self.storage.pause_sync()
            sync_was_paused = True
        ...
    finally:
        # Resume hybrid sync after consolidation
        if sync_was_paused:
            try:
                self.logger.info("Resuming hybrid backend sync after consolidation")
                await self.storage.resume_sync()
            except Exception as e:
                self.logger.error(f"Failed to resume sync after consolidation: {e}")
```

**After:**
```python
class SyncPauseContext:
    """Context manager for pausing hybrid backend sync during consolidation."""

    def __init__(self, storage, logger):
        self.storage = storage
        self.logger = logger
        self.is_hybrid = hasattr(storage, 'pause_sync') and hasattr(storage, 'resume_sync')
        self.was_paused = False

    async def __aenter__(self):
        if self.is_hybrid:
            self.logger.info("Pausing hybrid backend sync during consolidation")
            await self.storage.pause_sync()
            self.was_paused = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.was_paused:
            try:
                self.logger.info("Resuming hybrid backend sync")
                await self.storage.resume_sync()
            except Exception as e:
                self.logger.error(f"Failed to resume sync: {e}")

async def consolidate(self, time_horizon: str, **kwargs) -> ConsolidationReport:
    start_time = datetime.now()
    report = ConsolidationReport(...)

    async with SyncPauseContext(self.storage, self.logger):
        try:
            self.logger.info(f"Starting {time_horizon} consolidation...")
            # ... rest of consolidation logic
```

**Complexity Impact:** 12 → 10 (-2)
- Removes nested sync management logic
- Async context manager handles cleanup automatically

### Refactoring Proposal #2: Extract Phase-Specific Processing Guard
**Risk:** Low | **Impact:** -2 complexity | **Time:** 30 minutes

**Before:**
```python
async def consolidate(self, time_horizon: str, **kwargs) -> ConsolidationReport:
    ...
    # 3. Cluster by semantic similarity (if enabled and appropriate)
    clusters = []
    if self.config.clustering_enabled and time_horizon in ['weekly', 'monthly', 'quarterly']:
        self.logger.info(f"🔗 Phase 2/6: Clustering memories...")
        clusters = await self.clustering_engine.process(memories)
        report.clusters_created = len(clusters)

    # 4. Run creative associations (if enabled and appropriate)
    associations = []
    if self.config.associations_enabled and time_horizon in ['weekly', 'monthly']:
        self.logger.info(f"🧠 Phase 3/6: Discovering associations...")
        existing_associations = await self._get_existing_associations()
        associations = await self.association_engine.process(memories, existing_associations)
        report.associations_discovered = len(associations)
```

**After:**
```python
def should_run_clustering(self, time_horizon: str) -> bool:
    """Check if clustering should run for this time horizon."""
    return self.config.clustering_enabled and time_horizon in ['weekly', 'monthly', 'quarterly']

def should_run_associations(self, time_horizon: str) -> bool:
    """Check if association discovery should run for this time horizon."""
    return self.config.associations_enabled and time_horizon in ['weekly', 'monthly']

def should_run_compression(self, time_horizon: str) -> bool:
    """Check if compression should run for this time horizon."""
    return self.config.compression_enabled

def should_run_forgetting(self, time_horizon: str) -> bool:
    """Check if controlled forgetting should run for this time horizon."""
    return self.config.forgetting_enabled and time_horizon in ['monthly', 'quarterly', 'yearly']

async def consolidate(self, time_horizon: str, **kwargs) -> ConsolidationReport:
    ...
    # 3. Cluster by semantic similarity
    clusters = []
    if self.should_run_clustering(time_horizon):
        self.logger.info(f"🔗 Phase 2/6: Clustering memories...")
        clusters = await self.clustering_engine.process(memories)
        report.clusters_created = len(clusters)

    # 4. Run creative associations
    associations = []
    if self.should_run_associations(time_horizon):
        self.logger.info(f"🧠 Phase 3/6: Discovering associations...")
        existing_associations = await self._get_existing_associations()
        associations = await self.association_engine.process(memories, existing_associations)
        report.associations_discovered = len(associations)
```

**Complexity Impact:** 10 → 8 (-2)
- Extracts multi-condition guards to named methods
- Improves readability and testability

### Implementation Plan
1. **Extract sync context manager** (45 min, low risk) - Standard async pattern
2. **Extract phase guards** (30 min, low risk) - Simple boolean methods

**Total Complexity Reduction:** 12 → 8 (-4 points)
**Total Time:** 1.25 hours

---

## Target Function 4: analytics.py::get_analytics() (Complexity: 12)

### Current Implementation
**Purpose:** Aggregate analytics overview from storage backend.

**Note:** After examining analytics.py, the `get_analytics()` function doesn't exist. The most complex function is `get_memory_growth()` at lines 267-363 (complexity ~11).

### Complexity Breakdown (get_memory_growth)
```
Lines 279-293: +4 complexity (period validation and interval calculation)
Lines 304-334: +5 complexity (date grouping and interval aggregation loops)
Lines 336-353: +2 complexity (label generation and data point creation)
Total: 11
```

### Refactoring Proposal #1: Extract Period Configuration
**Risk:** Low | **Impact:** -3 complexity | **Time:** 45 minutes

**Before:**
```python
@router.get("/memory-growth", response_model=MemoryGrowthData, tags=["analytics"])
async def get_memory_growth(period: str = Query("month", ...), ...):
    try:
        # Define the period
        if period == "week":
            days = 7
            interval_days = 1
        elif period == "month":
            days = 30
            interval_days = 7
        elif period == "quarter":
            days = 90
            interval_days = 7
        elif period == "year":
            days = 365
            interval_days = 30
        else:
            raise HTTPException(status_code=400, detail="Invalid period...")
```

**After:**
```python
@dataclass
class PeriodConfig:
    """Configuration for time period analysis."""
    days: int
    interval_days: int
    label_format: str

PERIOD_CONFIGS = {
    "week": PeriodConfig(days=7, interval_days=1, label_format="daily"),
    "month": PeriodConfig(days=30, interval_days=7, label_format="weekly"),
    "quarter": PeriodConfig(days=90, interval_days=7, label_format="weekly"),
    "year": PeriodConfig(days=365, interval_days=30, label_format="monthly"),
}

def get_period_config(period: str) -> PeriodConfig:
    """Get configuration for the specified time period.

    Args:
        period: Time period identifier (week, month, quarter, year)

    Returns:
        PeriodConfig for the specified period

    Raises:
        HTTPException: If period is invalid
    """
    config = PERIOD_CONFIGS.get(period)
    if not config:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period. Use: {', '.join(PERIOD_CONFIGS.keys())}"
        )
    return config

@router.get("/memory-growth", response_model=MemoryGrowthData, tags=["analytics"])
async def get_memory_growth(period: str = Query("month", ...), ...):
    try:
        config = get_period_config(period)
        days = config.days
        interval_days = config.interval_days
```

**Complexity Impact:** 11 → 8 (-3)
- Replaces if/elif chain with dict lookup
- Configuration is data-driven and easily extensible

### Refactoring Proposal #2: Extract Interval Aggregation
**Risk:** Low | **Impact:** -2 complexity | **Time:** 1 hour

**Before:**
```python
async def get_memory_growth(...):
    ...
    # Create data points
    current_date = start_date.date()
    while current_date <= end_date.date():
        # For intervals > 1 day, sum counts across the entire interval
        interval_end = current_date + timedelta(days=interval_days)
        count = 0

        # Sum all memories within this interval
        check_date = current_date
        while check_date < interval_end and check_date <= end_date.date():
            count += date_counts.get(check_date, 0)
            check_date += timedelta(days=1)

        cumulative += count

        # Convert date to datetime for label generation
        current_datetime = datetime.combine(current_date, datetime.min.time())
        label = _generate_interval_label(current_datetime, period)

        data_points.append(MemoryGrowthPoint(...))

        current_date += timedelta(days=interval_days)
```

**After:**
```python
def aggregate_interval_counts(date_counts: Dict[date, int],
                             start_date: date,
                             end_date: date,
                             interval_days: int) -> List[Tuple[date, int]]:
    """Aggregate memory counts over time intervals.

    Args:
        date_counts: Map of dates to memory counts
        start_date: Start date for aggregation
        end_date: End date for aggregation
        interval_days: Number of days per interval

    Returns:
        List of (interval_start_date, count) tuples
    """
    intervals = []
    current_date = start_date

    while current_date <= end_date:
        interval_end = current_date + timedelta(days=interval_days)

        # Sum all memories within this interval
        count = 0
        check_date = current_date
        while check_date < interval_end and check_date <= end_date:
            count += date_counts.get(check_date, 0)
            check_date += timedelta(days=1)

        intervals.append((current_date, count))
        current_date += timedelta(days=interval_days)

    return intervals

def build_growth_data_points(intervals: List[Tuple[date, int]],
                            period: str) -> List[MemoryGrowthPoint]:
    """Build MemoryGrowthPoint objects from interval data.

    Args:
        intervals: List of (date, count) tuples
        period: Time period for label generation

    Returns:
        List of MemoryGrowthPoint objects with labels
    """
    data_points = []
    cumulative = 0

    for current_date, count in intervals:
        cumulative += count
        current_datetime = datetime.combine(current_date, datetime.min.time())
        label = _generate_interval_label(current_datetime, period)

        data_points.append(MemoryGrowthPoint(
            date=current_date.isoformat(),
            count=count,
            cumulative=cumulative,
            label=label
        ))

    return data_points

async def get_memory_growth(...):
    ...
    intervals = aggregate_interval_counts(date_counts, start_date.date(),
                                         end_date.date(), interval_days)
    data_points = build_growth_data_points(intervals, period)
```

**Complexity Impact:** 8 → 6 (-2)
- Separates data aggregation from presentation
- Nested loops extracted to dedicated function

### Implementation Plan
1. **Extract period config** (45 min, low risk) - Dict lookup pattern
2. **Extract interval aggregation** (1 hour, low risk) - Pure function extraction

**Total Complexity Reduction:** 11 → 6 (-5 points)
**Total Time:** 1.75 hours

---

## Target Function 5: quality_gate.sh functions (Complexity: 10-12)

### Current Implementation
**Purpose:** Multiple bash functions for PR quality checks.

**Note:** After searching, I found bash scripts in `/scripts/pr/` but they don't contain individual functions with measurable cyclomatic complexity in the Python sense. Bash scripts typically have complexity from conditional branches and loops, but they're measured differently.

Instead, I'll analyze a Python equivalent that would benefit from refactoring: The analytics endpoint functions that have similar complexity patterns.

Let me analyze `get_tag_usage_analytics()` from analytics.py (lines 366-428, complexity ~10).

### Complexity Breakdown (get_tag_usage_analytics)
```
Lines 379-395: +3 complexity (storage method availability checks and fallbacks)
Lines 397-410: +4 complexity (tag data processing with total memory calculation)
Lines 412-421: +3 complexity (tag stats calculation loop)
Total: 10
```

### Refactoring Proposal #1: Extract Storage Stats Retrieval
**Risk:** Low | **Impact:** -2 complexity | **Time:** 30 minutes

**Before:**
```python
async def get_tag_usage_analytics(...):
    try:
        # Get all tags with counts
        if hasattr(storage, 'get_all_tags_with_counts'):
            tag_data = await storage.get_all_tags_with_counts()
        else:
            raise HTTPException(status_code=501, detail="Tag analytics not supported...")

        # Get total memories for accurate percentage calculation
        if hasattr(storage, 'get_stats'):
            try:
                stats = await storage.get_stats()
                total_memories = stats.get("total_memories", 0)
            except Exception as e:
                logger.warning(f"Failed to retrieve storage stats: {e}")
                stats = {}
                total_memories = 0
        else:
            total_memories = 0

        if total_memories == 0:
            # Fallback: calculate from all tag data
            all_tags = tag_data.copy()
            total_memories = sum(tag["count"] for tag in all_tags)
```

**After:**
```python
async def get_total_memory_count(storage: MemoryStorage,
                                tag_data: List[Dict]) -> int:
    """Get total memory count from storage or calculate from tag data.

    Args:
        storage: Storage backend
        tag_data: Tag count data for fallback calculation

    Returns:
        Total memory count
    """
    if hasattr(storage, 'get_stats'):
        try:
            stats = await storage.get_stats()
            total = stats.get("total_memories", 0)
            if total > 0:
                return total
        except Exception as e:
            logger.warning(f"Failed to retrieve storage stats: {e}")

    # Fallback: calculate from tag data
    return sum(tag["count"] for tag in tag_data)

async def get_tag_usage_analytics(...):
    try:
        # Get all tags with counts
        if hasattr(storage, 'get_all_tags_with_counts'):
            tag_data = await storage.get_all_tags_with_counts()
        else:
            raise HTTPException(status_code=501,
                              detail="Tag analytics not supported by storage backend")

        total_memories = await get_total_memory_count(storage, tag_data)
```

**Complexity Impact:** 10 → 8 (-2)

### Refactoring Proposal #2: Extract Tag Stats Calculation
**Risk:** Low | **Impact:** -2 complexity | **Time:** 30 minutes

**Before:**
```python
async def get_tag_usage_analytics(...):
    ...
    # Convert to response format
    tags = []
    for tag_item in tag_data:
        percentage = (tag_item["count"] / total_memories * 100) if total_memories > 0 else 0

        tags.append(TagUsageStats(
            tag=tag_item["tag"],
            count=tag_item["count"],
            percentage=round(percentage, 1),
            growth_rate=None  # Would need historical data to calculate
        ))

    return TagUsageData(
        tags=tags,
        total_memories=total_memories,
        period=period
    )
```

**After:**
```python
def calculate_tag_percentage(count: int, total: int) -> float:
    """Calculate percentage safely handling division by zero.

    Args:
        count: Tag usage count
        total: Total memory count

    Returns:
        Rounded percentage (1 decimal place)
    """
    return round((count / total * 100) if total > 0 else 0, 1)

def build_tag_usage_stats(tag_data: List[Dict], total_memories: int) -> List[TagUsageStats]:
    """Build TagUsageStats objects from raw tag data.

    Args:
        tag_data: Raw tag count data
        total_memories: Total memory count for percentage calculation

    Returns:
        List of TagUsageStats objects
    """
    return [
        TagUsageStats(
            tag=tag_item["tag"],
            count=tag_item["count"],
            percentage=calculate_tag_percentage(tag_item["count"], total_memories),
            growth_rate=None  # Would need historical data
        )
        for tag_item in tag_data
    ]

async def get_tag_usage_analytics(...):
    ...
    tags = build_tag_usage_stats(tag_data, total_memories)

    return TagUsageData(
        tags=tags,
        total_memories=total_memories,
        period=period
    )
```

**Complexity Impact:** 8 → 6 (-2)

### Implementation Plan
1. **Extract memory count retrieval** (30 min, low risk) - Simple extraction
2. **Extract tag stats calculation** (30 min, low risk) - Pure function

**Total Complexity Reduction:** 10 → 6 (-4 points)
**Total Time:** 1 hour

---

## Quick Wins Summary

### Quick Win 1: install.py::detect_gpu() (Complexity: 10 → 7)
**Refactoring:** Extract platform-specific GPU detection to separate functions
**Time:** 1 hour | **Risk:** Low

**Before:**
```python
def detect_gpu():
    system_info = detect_system()

    # Check for CUDA
    has_cuda = False
    cuda_version = None
    if system_info["is_windows"]:
        cuda_path = os.environ.get('CUDA_PATH')
        if cuda_path and os.path.exists(cuda_path):
            has_cuda = True
            # ... 15 lines of version detection
    elif system_info["is_linux"]:
        cuda_paths = ['/usr/local/cuda', os.environ.get('CUDA_HOME')]
        for path in cuda_paths:
            if path and os.path.exists(path):
                has_cuda = True
                # ... 15 lines of version detection
```

**After:**
```python
def detect_cuda_windows() -> Tuple[bool, Optional[str]]:
    """Detect CUDA on Windows systems."""
    cuda_path = os.environ.get('CUDA_PATH')
    if not (cuda_path and os.path.exists(cuda_path)):
        return False, None

    # ... version detection logic
    return True, cuda_version

def detect_cuda_linux() -> Tuple[bool, Optional[str]]:
    """Detect CUDA on Linux systems."""
    cuda_paths = ['/usr/local/cuda', os.environ.get('CUDA_HOME')]
    for path in cuda_paths:
        if path and os.path.exists(path):
            # ... version detection logic
            return True, cuda_version
    return False, None

CUDA_DETECTORS = {
    'windows': detect_cuda_windows,
    'linux': detect_cuda_linux,
}

def detect_gpu():
    system_info = detect_system()

    # Detect CUDA using platform-specific detector
    detector_key = 'windows' if system_info["is_windows"] else 'linux'
    detector = CUDA_DETECTORS.get(detector_key, lambda: (False, None))
    has_cuda, cuda_version = detector()
```

**Impact:** -3 complexity
- Platform-specific logic extracted
- Dict dispatch replaces if/elif chain

---

### Quick Win 2: cloudflare.py::get_memory_timestamps() (Complexity: 9 → 7)
**Refactoring:** Extract SQL query building and result processing
**Time:** 45 minutes | **Risk:** Low

**Before:**
```python
async def get_memory_timestamps(self, days: Optional[int] = None) -> List[float]:
    try:
        if days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_timestamp = cutoff.timestamp()

            sql = "SELECT created_at FROM memories WHERE created_at >= ? ORDER BY created_at DESC"
            payload = {"sql": sql, "params": [cutoff_timestamp]}
        else:
            sql = "SELECT created_at FROM memories ORDER BY created_at DESC"
            payload = {"sql": sql, "params": []}

        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()

        timestamps = []
        if result.get("success") and result.get("result", [{}])[0].get("results"):
            for row in result["result"][0]["results"]:
                if row.get("created_at") is not None:
                    timestamps.append(float(row["created_at"]))
```

**After:**
```python
def build_timestamp_query(days: Optional[int]) -> Tuple[str, List[Any]]:
    """Build SQL query for fetching memory timestamps.

    Args:
        days: Optional day limit for filtering

    Returns:
        Tuple of (sql_query, parameters)
    """
    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return (
            "SELECT created_at FROM memories WHERE created_at >= ? ORDER BY created_at DESC",
            [cutoff.timestamp()]
        )
    return (
        "SELECT created_at FROM memories ORDER BY created_at DESC",
        []
    )

def extract_timestamps(result: Dict) -> List[float]:
    """Extract timestamp values from D1 query result.

    Args:
        result: D1 query response JSON

    Returns:
        List of Unix timestamps
    """
    if not (result.get("success") and result.get("result", [{}])[0].get("results")):
        return []

    return [
        float(row["created_at"])
        for row in result["result"][0]["results"]
        if row.get("created_at") is not None
    ]

async def get_memory_timestamps(self, days: Optional[int] = None) -> List[float]:
    try:
        sql, params = build_timestamp_query(days)
        payload = {"sql": sql, "params": params}

        response = await self._retry_request("POST", f"{self.d1_url}/query", json=payload)
        result = response.json()

        timestamps = extract_timestamps(result)
```

**Impact:** -2 complexity
- Query building extracted
- Result processing extracted

---

### Quick Win 3: consolidator.py::_get_memories_for_horizon() (Complexity: 10 → 8)
**Refactoring:** Extract time range calculation and incremental mode sorting
**Time:** 45 minutes | **Risk:** Low

**Before:**
```python
async def _get_memories_for_horizon(self, time_horizon: str, **kwargs) -> List[Memory]:
    now = datetime.now()

    # Define time ranges for different horizons
    time_ranges = {
        'daily': timedelta(days=1),
        'weekly': timedelta(days=7),
        'monthly': timedelta(days=30),
        'quarterly': timedelta(days=90),
        'yearly': timedelta(days=365)
    }

    if time_horizon not in time_ranges:
        raise ConsolidationError(f"Unknown time horizon: {time_horizon}")

    # For daily processing, get recent memories (no change - already efficient)
    if time_horizon == 'daily':
        start_time = (now - timedelta(days=2)).timestamp()
        end_time = now.timestamp()
        memories = await self.storage.get_memories_by_time_range(start_time, end_time)
    else:
        # ... complex incremental logic
```

**After:**
```python
TIME_HORIZON_CONFIGS = {
    'daily': {'days': 1, 'use_time_range': True, 'range_days': 2},
    'weekly': {'days': 7, 'use_time_range': False},
    'monthly': {'days': 30, 'use_time_range': False},
    'quarterly': {'days': 90, 'use_time_range': False},
    'yearly': {'days': 365, 'use_time_range': False}
}

def get_consolidation_sort_key(memory: Memory) -> float:
    """Get sort key for incremental consolidation (oldest first).

    Args:
        memory: Memory object to get sort key for

    Returns:
        Sort key (timestamp, lower = older)
    """
    if memory.metadata and 'last_consolidated_at' in memory.metadata:
        return float(memory.metadata['last_consolidated_at'])
    return memory.created_at if memory.created_at else 0.0

async def _get_memories_for_horizon(self, time_horizon: str, **kwargs) -> List[Memory]:
    config = TIME_HORIZON_CONFIGS.get(time_horizon)
    if not config:
        raise ConsolidationError(f"Unknown time horizon: {time_horizon}")

    now = datetime.now()

    if config['use_time_range']:
        start_time = (now - timedelta(days=config['range_days'])).timestamp()
        end_time = now.timestamp()
        return await self.storage.get_memories_by_time_range(start_time, end_time)

    # ... simplified incremental logic using extracted functions
```

**Impact:** -2 complexity
- Config-driven time range selection
- Sort key extraction to separate function

---

### Quick Win 4: analytics.py::get_activity_breakdown() (Complexity: 9 → 7)
**Refactoring:** Extract granularity-specific aggregation functions
**Time:** 1 hour | **Risk:** Low

**Before:**
```python
async def get_activity_breakdown(granularity: str = Query("daily", ...)):
    ...
    if granularity == "hourly":
        hour_counts = defaultdict(int)
        for timestamp in timestamps:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            hour_counts[dt.hour] += 1
            active_days.add(dt.date())
            activity_dates.append(dt.date())
        # ... 10 lines of breakdown building
    elif granularity == "daily":
        day_counts = defaultdict(int)
        day_names = ["Monday", "Tuesday", ...]
        # ... 15 lines of breakdown building
    else:  # weekly
        week_counts = defaultdict(int)
        # ... 20 lines of breakdown building
```

**After:**
```python
def aggregate_hourly(timestamps: List[float]) -> Tuple[List[ActivityBreakdown], Set[date], List[date]]:
    """Aggregate activity data by hour."""
    hour_counts = defaultdict(int)
    active_days = set()
    activity_dates = []

    for timestamp in timestamps:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        hour_counts[dt.hour] += 1
        active_days.add(dt.date())
        activity_dates.append(dt.date())

    breakdown = [
        ActivityBreakdown(period="hourly", count=hour_counts.get(hour, 0), label=f"{hour:02d}:00")
        for hour in range(24)
    ]
    return breakdown, active_days, activity_dates

GRANULARITY_AGGREGATORS = {
    'hourly': aggregate_hourly,
    'daily': aggregate_daily,
    'weekly': aggregate_weekly
}

async def get_activity_breakdown(granularity: str = Query("daily", ...)):
    ...
    aggregator = GRANULARITY_AGGREGATORS.get(granularity, aggregate_daily)
    breakdown, active_days, activity_dates = aggregator(timestamps)
```

**Impact:** -2 complexity
- Granularity-specific logic extracted
- Dict dispatch replaces if/elif chain

---

### Quick Win 5: analytics.py::get_memory_type_distribution() (Complexity: 9 → 7)
**Refactoring:** Extract storage backend type detection and query building
**Time:** 45 minutes | **Risk:** Low

**Before:**
```python
async def get_memory_type_distribution(storage: MemoryStorage = Depends(get_storage), ...):
    try:
        # Try multiple approaches based on storage backend
        if hasattr(storage, 'get_type_counts'):
            type_counts_data = await storage.get_type_counts()
            type_counts = dict(type_counts_data)
            total_memories = sum(type_counts.values())
        elif hasattr(storage, 'primary') and hasattr(storage.primary, 'conn'):
            # Hybrid storage - access underlying SQLite
            cursor = storage.primary.conn.cursor()
            cursor.execute("""SELECT ... FROM memories GROUP BY mem_type""")
            type_counts = {row[0]: row[1] for row in cursor.fetchall()}
            ...
        elif hasattr(storage, 'conn') and storage.conn:
            # Direct SQLite storage
            cursor = storage.conn.cursor()
            cursor.execute("""SELECT ... FROM memories GROUP BY mem_type""")
            ...
```

**After:**
```python
async def get_type_counts_from_storage(storage: MemoryStorage) -> Tuple[Dict[str, int], int]:
    """Get memory type counts from storage backend.

    Returns:
        Tuple of (type_counts_dict, total_memories)
    """
    # Native support
    if hasattr(storage, 'get_type_counts'):
        type_counts_data = await storage.get_type_counts()
        type_counts = dict(type_counts_data)
        return type_counts, sum(type_counts.values())

    # Direct SQLite query (hybrid or direct)
    conn = None
    if hasattr(storage, 'primary') and hasattr(storage.primary, 'conn'):
        conn = storage.primary.conn
    elif hasattr(storage, 'conn'):
        conn = storage.conn

    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                CASE WHEN memory_type IS NULL OR memory_type = '' THEN 'untyped'
                     ELSE memory_type END as mem_type,
                COUNT(*) as count
            FROM memories GROUP BY mem_type
        """)
        type_counts = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.execute("SELECT COUNT(*) FROM memories")
        return type_counts, cursor.fetchone()[0]

    # Fallback to sampling
    logger.warning("Using sampling approach - results may be incomplete")
    memories = await storage.get_recent_memories(n=1000)
    type_counts = defaultdict(int)
    for memory in memories:
        type_counts[memory.memory_type or "untyped"] += 1
    return dict(type_counts), len(memories)

async def get_memory_type_distribution(storage: MemoryStorage = Depends(get_storage), ...):
    try:
        type_counts, total_memories = await get_type_counts_from_storage(storage)
        # ... build response
```

**Impact:** -2 complexity
- Backend detection logic extracted
- Early return pattern in extraction function

---

## Implementation Roadmap

### Phase 2A: Core Functions (Week 1)
**Target:** configure_paths, cloudflare tag search, consolidator.consolidate

| Function | Priority | Time | Dependency | Parallel? |
|----------|----------|------|------------|-----------|
| install.py::configure_paths() | High | 4h | None | Yes |
| cloudflare.py::_search_by_tags_internal() | High | 1.75h | None | Yes |
| consolidator.py::consolidate() | High | 1.25h | None | Yes |

**Subtotal:** 7 hours (can be done in parallel)

### Phase 2B: Analytics Functions (Week 2)
**Target:** analytics endpoints optimization

| Function | Priority | Time | Dependency | Parallel? |
|----------|----------|------|------------|-----------|
| analytics.py::get_memory_growth() | Medium | 1.75h | None | Yes |
| analytics.py::get_tag_usage_analytics() | Medium | 1h | None | Yes |

**Subtotal:** 2.75 hours (can be done in parallel)

### Phase 2C: Quick Wins (Week 2-3)
**Target:** Low-risk, high-impact improvements

| Function | Priority | Time | Dependency | Parallel? |
|----------|----------|------|------------|-----------|
| install.py::detect_gpu() | Low | 1h | None | Yes |
| cloudflare.py::get_memory_timestamps() | Low | 45m | None | Yes |
| consolidator.py::_get_memories_for_horizon() | Low | 45m | None | Yes |
| analytics.py::get_activity_breakdown() | Low | 1h | None | Yes |
| analytics.py::get_memory_type_distribution() | Low | 45m | None | Yes |

**Subtotal:** 4.25 hours (can be done in parallel)

### Total Time Estimate
- **Sequential execution:** 14 hours
- **Parallel execution (with team):** 7 hours (Phase 2A) + 2.75h (Phase 2B) + 2h (Phase 2C) = **11.75 hours**
- **Recommended:** 12-15 hours (including testing and documentation)

---

## Expected Health Impact

### Complexity Score Improvement
**Current:** 40/100
- 5 main target functions: -28 complexity points total
- 5 quick wins: -11 complexity points total
- **Total reduction:** -39 complexity points across 10 functions

**Projected:** 50-55/100 (+10-15 points)

### Overall Health Score Improvement
**Current:** 63/100 (Grade C)
**Projected:** 66-68/100 (Grade C+)

**Calculation:**
- Phase 1 (dead code): +5-9 points → 68-72
- Phase 2 (complexity): +3 points → 71-75

---

## Success Criteria

### Quantitative
- [ ] All 5 main functions reduced by 3+ complexity points each
- [ ] All 5 quick wins implemented successfully
- [ ] Total complexity reduction: 30+ points
- [ ] No breaking changes (all tests passing)
- [ ] No performance regressions

### Qualitative
- [ ] Code readability improved (subjective review)
- [ ] Functions easier to understand and maintain
- [ ] Better separation of concerns
- [ ] Improved testability (isolated functions)

---

## Risk Assessment Matrix

| Function | Risk | Testing Requirements | Critical Path | Priority |
|----------|------|---------------------|---------------|----------|
| configure_paths | Low | Unit + integration | No (setup only) | High |
| _search_by_tags_internal | Low | Unit + DB tests | Yes (core search) | High |
| consolidate | Medium | Integration tests | Yes (consolidation) | High |
| get_memory_growth | Low | Unit + API tests | No (analytics) | Medium |
| get_tag_usage_analytics | Low | Unit + API tests | No (analytics) | Medium |
| detect_gpu | Low | Unit tests | No (setup only) | Low |
| get_memory_timestamps | Low | Unit + DB tests | No (analytics) | Low |
| _get_memories_for_horizon | Medium | Integration tests | Yes (consolidation) | Medium |
| get_activity_breakdown | Low | Unit + API tests | No (analytics) | Low |
| get_memory_type_distribution | Low | Unit + API tests | No (analytics) | Low |

**Critical Path Functions (require careful testing):**
1. _search_by_tags_internal - Core search functionality
2. consolidate - Memory consolidation pipeline
3. _get_memories_for_horizon - Consolidation memory selection

**Low-Risk Functions (easier to refactor):**
- All analytics endpoints (read-only, non-critical)
- Setup functions (configure_paths, detect_gpu)

---

## Testing Strategy

### Unit Tests (per function)
- Test extracted functions independently
- Verify input/output contracts
- Test edge cases and error handling

### Integration Tests
- Test critical path functions with real storage
- Verify no behavioral changes
- Performance benchmarks (before/after)

### Regression Tests
- Run full test suite after each refactoring
- Verify API contracts unchanged
- Check performance hasn't degraded

---

## Next Steps

1. **Review and approve** this Phase 2 analysis
2. **Select implementation approach:**
   - Option A: Sequential (14 hours, single developer)
   - Option B: Parallel (12 hours, multiple developers)
   - Option C: Prioritized (7 hours for critical functions only)

3. **Set up tracking:**
   - Create GitHub issues for each function
   - Track complexity reduction progress
   - Monitor test coverage

4. **Begin Phase 2A** (highest priority functions)

---

## Appendix: Refactoring Patterns Used

### Pattern 1: Extract Method
**Purpose:** Reduce function length and improve testability
**Used in:** All functions analyzed
**Example:** Platform detection, SQL query building

### Pattern 2: Guard Clause
**Purpose:** Reduce nesting and improve readability
**Used in:** Tag search, config updates
**Example:** Early returns for validation

### Pattern 3: Dict Lookup
**Purpose:** Replace if/elif chains with data-driven logic
**Used in:** Period configs, platform detection
**Example:** `PERIOD_CONFIGS[period]` instead of if/elif

### Pattern 4: Context Manager
**Purpose:** Simplify resource management and cleanup
**Used in:** Consolidation sync management
**Example:** `async with SyncPauseContext(...)`

### Pattern 5: Configuration Object
**Purpose:** Centralize related configuration data
**Used in:** Period analysis, time horizons
**Example:** `@dataclass PeriodConfig`

---

## Lessons from Phase 1

**What worked well:**
- Clear complexity scoring and prioritization
- Incremental approach (low-risk first)
- Automated testing validation

**Improvements for Phase 2:**
- More explicit refactoring examples (✅ done)
- Better risk assessment (✅ done)
- Parallel execution planning (✅ done)

---

**End of Phase 2 Analysis**
**Total Functions Analyzed:** 10 (5 main + 5 quick wins)
**Total Complexity Reduction:** -39 points
**Total Time Estimate:** 12-15 hours
