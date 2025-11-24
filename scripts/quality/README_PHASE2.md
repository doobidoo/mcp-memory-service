# Phase 2 Complexity Reduction - Quick Reference

## Overview

This guide provides a quick reference for implementing Phase 2 complexity reductions identified in `phase2_complexity_analysis.md`.

## Quick Stats

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **Complexity Score** | 40/100 | 50-55/100 | +10-15 points |
| **Overall Health** | 63/100 | 66-68/100 | +3 points |
| **Functions Analyzed** | 10 | - | - |
| **Total Time Estimate** | - | 12-15 hours | - |
| **Complexity Reduction** | - | -39 points | - |

## Priority Matrix

### High Priority (Week 1) - 7 hours
Critical path functions that need careful attention:

1. **install.py::configure_paths()** (15 → 5, -10 points, 4h)
   - Extract platform detection
   - Extract storage setup
   - Extract Claude config update

2. **cloudflare.py::_search_by_tags_internal()** (13 → 8, -5 points, 1.75h)
   - Extract tag normalization
   - Extract SQL query builder

3. **consolidator.py::consolidate()** (12 → 8, -4 points, 1.25h)
   - Extract sync context manager
   - Extract phase guards

### Medium Priority (Week 2) - 2.75 hours
Analytics functions (non-critical):

4. **analytics.py::get_memory_growth()** (11 → 6, -5 points, 1.75h)
   - Extract period configuration
   - Extract interval aggregation

5. **analytics.py::get_tag_usage_analytics()** (10 → 6, -4 points, 1h)
   - Extract storage stats retrieval
   - Extract tag stats calculation

### Low Priority (Weeks 2-3) - 4.25 hours
Quick wins with minimal risk:

6. **install.py::detect_gpu()** (10 → 7, -3 points, 1h)
7. **cloudflare.py::get_memory_timestamps()** (9 → 7, -2 points, 45m)
8. **consolidator.py::_get_memories_for_horizon()** (10 → 8, -2 points, 45m)
9. **analytics.py::get_activity_breakdown()** (9 → 7, -2 points, 1h)
10. **analytics.py::get_memory_type_distribution()** (9 → 7, -2 points, 45m)

## Refactoring Patterns Cheat Sheet

### Pattern 1: Extract Method
**When to use:** Function > 50 lines, nested logic, repeated code

**Example:**
```python
# Before
def complex_function():
    # 20 lines of platform detection
    # 30 lines of setup logic
    # 15 lines of validation

# After
def detect_platform(): ...
def setup_system(): ...
def validate_config(): ...

def complex_function():
    platform = detect_platform()
    setup_system(platform)
    validate_config()
```

### Pattern 2: Dict Lookup
**When to use:** if/elif/else chains with similar structure

**Example:**
```python
# Before
if period == "week":
    days = 7
elif period == "month":
    days = 30
elif period == "year":
    days = 365

# After
PERIOD_DAYS = {"week": 7, "month": 30, "year": 365}
days = PERIOD_DAYS[period]
```

### Pattern 3: Guard Clause
**When to use:** Nested if statements, early validation

**Example:**
```python
# Before
def process(data):
    if data is not None:
        if data.valid():
            if data.ready():
                return process_data(data)
    return None

# After
def process(data):
    if data is None:
        return None
    if not data.valid():
        return None
    if not data.ready():
        return None
    return process_data(data)
```

### Pattern 4: Context Manager
**When to use:** Resource management, setup/teardown logic

**Example:**
```python
# Before
def process():
    resource = acquire()
    try:
        do_work(resource)
    finally:
        release(resource)

# After
class ResourceManager:
    async def __aenter__(self): ...
    async def __aexit__(self, *args): ...

async def process():
    async with ResourceManager() as resource:
        do_work(resource)
```

### Pattern 5: Configuration Object
**When to use:** Related configuration values, multiple parameters

**Example:**
```python
# Before
def analyze(period, days, interval, format):
    ...

# After
@dataclass
class AnalysisConfig:
    period: str
    days: int
    interval: int
    format: str

def analyze(config: AnalysisConfig):
    ...
```

## Testing Checklist

For each refactored function:

- [ ] **Unit tests pass** - Run `pytest tests/test_<module>.py`
- [ ] **Integration tests pass** - Run `pytest tests/integration/`
- [ ] **No performance regression** - Benchmark before/after
- [ ] **API contracts unchanged** - Check response formats
- [ ] **Edge cases tested** - Null inputs, empty lists, errors
- [ ] **Documentation updated** - Docstrings, comments

## Implementation Order

### Sequential (Single Developer)
1. Week 1: High priority functions (7h)
2. Week 2: Medium priority functions (2.75h)
3. Week 3: Low priority quick wins (4.25h)

**Total:** 14 hours over 3 weeks

### Parallel (Multiple Developers)
1. **Developer A:** configure_paths, detect_gpu (5h)
2. **Developer B:** cloudflare functions (2.5h)
3. **Developer C:** consolidator functions (2h)
4. **Developer D:** analytics functions (4.75h)

**Total:** ~7 hours (with coordination overhead: 9-10 hours)

### Prioritized (Critical Path Only)
Focus on high-priority functions only:
1. configure_paths (4h)
2. _search_by_tags_internal (1.75h)
3. consolidate (1.25h)

**Total:** 7 hours for core improvements

## Risk Mitigation

### Critical Path Functions
**Extra caution required:**
- _search_by_tags_internal (core search)
- consolidate (memory consolidation)
- _get_memories_for_horizon (consolidation)

**Safety measures:**
- Create feature branch for each
- Comprehensive integration tests
- Performance benchmarking
- Staged rollout (dev → staging → production)

### Low-Risk Functions
**Can be batched:**
- All analytics endpoints (read-only)
- Setup functions (non-critical path)

**Safety measures:**
- Standard unit testing
- Manual smoke testing
- Can be rolled back easily

## Success Metrics

### Quantitative Goals
- [ ] Complexity score: 40 → 50+ (+10 points minimum)
- [ ] Overall health: 63 → 66+ (+3 points minimum)
- [ ] All 10 functions refactored successfully
- [ ] Zero breaking changes
- [ ] All tests passing

### Qualitative Goals
- [ ] Code easier to understand (peer review)
- [ ] Functions are testable in isolation
- [ ] Better separation of concerns
- [ ] Improved maintainability

## Common Pitfalls to Avoid

### 1. Over-Extraction
**Problem:** Creating too many tiny functions
**Solution:** Extract only when it improves clarity (10+ lines minimum)

### 2. Breaking API Contracts
**Problem:** Changing function signatures
**Solution:** Keep public APIs unchanged, refactor internals only

### 3. Performance Regression
**Problem:** Excessive function calls overhead
**Solution:** Benchmark before/after, inline hot paths if needed

### 4. Incomplete Testing
**Problem:** Missing edge cases
**Solution:** Test error paths, null inputs, boundary conditions

### 5. Rushing Critical Functions
**Problem:** Breaking core functionality
**Solution:** Extra time for testing critical path functions

## Command Reference

### Run Quality Analysis
```bash
# Run pyscn baseline report
python -m pyscn baseline --output scripts/quality/baseline_report.txt

# Check specific function complexity
python -m radon cc src/mcp_memory_service/storage/cloudflare.py -a

# Check cyclomatic complexity for all files
python -m radon cc src/ -a
```

### Run Tests
```bash
# All tests
pytest tests/

# Specific module
pytest tests/test_storage.py

# Integration tests only
pytest tests/integration/

# With coverage
pytest tests/ --cov=mcp_memory_service --cov-report=html
```

### Benchmark Performance
```bash
# Before refactoring
python scripts/benchmarks/run_benchmarks.py --baseline

# After refactoring
python scripts/benchmarks/run_benchmarks.py --compare
```

## Getting Help

### Resources
- **Phase 2 Analysis:** `scripts/quality/phase2_complexity_analysis.md` (detailed proposals)
- **Phase 1 Results:** `scripts/quality/phase1_dead_code_analysis.md` (lessons learned)
- **Complexity Guide:** `scripts/quality/complexity_scoring_guide.md` (understanding metrics)

### Questions?
- Review the detailed analysis for each function
- Check the refactoring pattern examples
- Test incrementally after each change
- Ask for peer review on critical functions

---

**Last Updated:** 2024-11-24
**Next Review:** After Phase 2 completion
