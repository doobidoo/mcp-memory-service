# Regression Tests

This document provides structured test scenarios for validating critical functionality and preventing regressions. Each test includes setup instructions, expected results, evidence collection, and pass/fail criteria.

## Purpose

Regression tests ensure that:
- Critical bugs don't reappear after being fixed
- Performance optimizations don't degrade over time
- Platform-specific issues are caught before release
- Integration points (MCP, HTTP API, storage backends) work correctly

## Test Categories

1. [Database Locking & Concurrency](#database-locking--concurrency)
2. [Storage Backend Integrity](#storage-backend-integrity)
3. [Dashboard Performance](#dashboard-performance)
4. [Tag Filtering Correctness](#tag-filtering-correctness)
5. [MCP Protocol Compliance](#mcp-protocol-compliance)

---

## Database Locking & Concurrency

### Test 1: Concurrent MCP Server Startup

**Context:** v8.9.0+ fixed "database is locked" errors by setting SQLite pragmas before connection

**Setup:**
1. Close all Claude Desktop instances
2. Ensure SQLite database exists at `~/Library/Application Support/mcp-memory/sqlite_vec.db` (macOS)
3. Verify `.env` contains `MCP_MEMORY_SQLITE_PRAGMAS=busy_timeout=15000,journal_mode=WAL`

**Execution:**
1. Open 3 Claude Desktop instances simultaneously (within 5 seconds)
2. In each instance, trigger memory service initialization:
   ```
   /mcp
   # Wait for MCP servers to connect
   # Try storing a memory in each instance
   ```
3. Monitor logs in `~/Library/Logs/Claude/mcp-server-memory.log`

**Expected Results:**
- ✅ All 3 instances connect successfully
- ✅ Zero "database is locked" errors in logs
- ✅ All instances show healthy status via `/api/health`
- ✅ Memory operations work in all instances

**Evidence Collection:**
```bash
# Check for lock errors
grep -i "database is locked" ~/Library/Logs/Claude/mcp-server-memory.log

# Verify pragma settings
sqlite3 ~/Library/Application\ Support/mcp-memory/sqlite_vec.db "PRAGMA busy_timeout;"
# Expected output: 15000

# Check journal mode
sqlite3 ~/Library/Application\ Support/mcp-memory/sqlite_vec.db "PRAGMA journal_mode;"
# Expected output: wal
```

**Pass Criteria:**
- ✅ Zero lock errors
- ✅ All servers initialize within 10 seconds
- ✅ Concurrent memory operations succeed
- ❌ FAIL if any server shows "database is locked"

---

### Test 2: Concurrent Memory Operations

**Context:** Test simultaneous read/write operations from multiple clients

**Setup:**
1. Start HTTP server: `memory launch`
2. Verify server is healthy: `curl http://127.0.0.1:8000/api/health`

**Execution:**
1. Run concurrent memory stores from multiple terminals:
   ```bash
   # Terminal 1
   for i in {1..50}; do
     curl -X POST http://127.0.0.1:8000/api/memories \
       -H "Content-Type: application/json" \
       -d "{\"content\":\"Test memory $i from terminal 1\",\"tags\":[\"test\",\"concurrent\"]}"
   done

   # Terminal 2 (run simultaneously)
   for i in {1..50}; do
     curl -X POST http://127.0.0.1:8000/api/memories \
       -H "Content-Type: application/json" \
       -d "{\"content\":\"Test memory $i from terminal 2\",\"tags\":[\"test\",\"concurrent\"]}"
   done
   ```

2. While stores are running, perform searches:
   ```bash
   # Terminal 3
   for i in {1..20}; do
     curl -s "http://127.0.0.1:8000/api/search" \
       -H "Content-Type: application/json" \
       -d '{"query":"test memory","limit":10}'
   done
   ```

**Expected Results:**
- ✅ All 100 memory stores complete successfully
- ✅ Zero HTTP 500 errors
- ✅ Search operations return results during writes
- ✅ No database lock errors in server logs

**Evidence Collection:**
```bash
# Count successful stores
curl -s "http://127.0.0.1:8000/api/search/by-tag" \
  -H "Content-Type: application/json" \
  -d '{"tags":["concurrent"],"limit":1000}' | jq '.memories | length'
# Expected: 100

# Check server logs for errors
tail -100 ~/Library/Logs/Claude/mcp-server-memory.log | grep -i error
```

**Pass Criteria:**
- ✅ 100 memories stored successfully
- ✅ Zero database lock errors
- ✅ Zero HTTP 500 responses
- ❌ FAIL if any operation times out or errors

---

## Storage Backend Integrity

### Test 3: Hybrid Backend Synchronization

**Context:** Verify hybrid backend syncs SQLite → Cloudflare without data loss

**Setup:**
1. Configure hybrid backend in `.env`:
   ```bash
   MCP_MEMORY_STORAGE_BACKEND=hybrid
   MCP_HYBRID_SYNC_INTERVAL=10  # Frequent sync for testing
   CLOUDFLARE_API_TOKEN=your-token
   CLOUDFLARE_ACCOUNT_ID=your-account
   CLOUDFLARE_D1_DATABASE_ID=your-db-id
   CLOUDFLARE_VECTORIZE_INDEX=mcp-memory-index
   ```
2. Clear Cloudflare backend: `python scripts/database/clear_cloudflare.py --confirm`
3. Start server: `memory launch`

**Execution:**
1. Store 10 test memories via API:
   ```bash
   for i in {1..10}; do
     curl -X POST http://127.0.0.1:8000/api/memories \
       -H "Content-Type: application/json" \
       -d "{\"content\":\"Hybrid test memory $i\",\"tags\":[\"hybrid-test\"]}"
   done
   ```

2. Wait 30 seconds (3x sync interval) for background sync

3. Query Cloudflare backend directly:
   ```bash
   python scripts/sync/check_cloudflare_sync.py --tag hybrid-test
   ```

**Expected Results:**
- ✅ All 10 memories present in SQLite (immediate)
- ✅ All 10 memories synced to Cloudflare (within 30s)
- ✅ Content hashes match between backends
- ✅ No sync errors in server logs

**Evidence Collection:**
```bash
# Check SQLite count
curl -s "http://127.0.0.1:8000/api/search/by-tag" \
  -H "Content-Type: application/json" \
  -d '{"tags":["hybrid-test"]}' | jq '.memories | length'

# Check Cloudflare count
python scripts/sync/check_cloudflare_sync.py --tag hybrid-test --count

# Compare content hashes
python scripts/sync/check_cloudflare_sync.py --tag hybrid-test --verify-hashes
```

**Pass Criteria:**
- ✅ SQLite count == Cloudflare count
- ✅ All content hashes match
- ✅ Sync completes within 30 seconds
- ❌ FAIL if any memory missing or hash mismatch

---

### Test 4: Storage Backend Switching

**Context:** Verify switching backends doesn't corrupt existing data

**Setup:**
1. Start with sqlite-vec backend, store 20 memories
2. Stop server
3. Configure hybrid backend, restart server
4. Verify all memories still accessible

**Execution:**
1. **SQLite-vec phase:**
   ```bash
   export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
   memory launch

   # Store 20 memories
   for i in {1..20}; do
     curl -X POST http://127.0.0.1:8000/api/memories \
       -H "Content-Type: application/json" \
       -d "{\"content\":\"Backend switch test $i\",\"tags\":[\"switch-test\"]}"
   done

   memory stop
   ```

2. **Switch to hybrid:**
   ```bash
   export MCP_MEMORY_STORAGE_BACKEND=hybrid
   export MCP_HYBRID_SYNC_INTERVAL=10
   # Set Cloudflare credentials...
   memory launch

   # Wait for startup
   sleep 5
   ```

3. **Verify data integrity:**
   ```bash
   curl -s "http://127.0.0.1:8000/api/search/by-tag" \
     -H "Content-Type: application/json" \
     -d '{"tags":["switch-test"]}' | jq '.memories | length'
   ```

**Expected Results:**
- ✅ All 20 memories still accessible after switch
- ✅ Memories begin syncing to Cloudflare
- ✅ No data corruption or loss
- ✅ Health check shows "hybrid" backend

**Pass Criteria:**
- ✅ 20 memories retrieved successfully
- ✅ Backend reported as "hybrid" in health check
- ✅ No errors during backend initialization
- ❌ FAIL if any memory inaccessible or corrupted

---

## Dashboard Performance

### Test 5: Page Load Performance

**Context:** Dashboard should load in <2 seconds (v7.2.2 benchmark: 25ms)

**Setup:**
1. Database with 1000+ memories
2. HTTP server running: `memory launch`
3. Dashboard at `http://127.0.0.1:8000/`

**Execution:**
```bash
# Measure page load time (10 iterations)
for i in {1..10}; do
  time curl -s "http://127.0.0.1:8000/" > /dev/null
done
```

**Expected Results:**
- ✅ Average load time <500ms
- ✅ All static assets (HTML/CSS/JS) load successfully
- ✅ No JavaScript errors in browser console
- ✅ Dashboard functional on first load

**Evidence Collection:**
```bash
# Browser DevTools → Network tab
# - Check "Load" time in waterfall
# - Verify no 404/500 errors
# - Measure DOMContentLoaded and Load events

# Server-side timing
time curl -s "http://127.0.0.1:8000/" -o /dev/null -w "%{time_total}\n"
```

**Pass Criteria:**
- ✅ Page load <2 seconds (target: <500ms)
- ✅ Zero resource loading errors
- ✅ Dashboard interactive immediately
- ❌ FAIL if >2 seconds or JavaScript errors

---

### Test 6: Memory Operation Performance

**Context:** CRUD operations should complete in <1 second (v7.2.2 benchmark: 26ms)

**Setup:**
1. Clean database: `python scripts/database/reset_database.py --confirm`
2. HTTP server running

**Execution:**
1. **Store operation:**
   ```bash
   time curl -s -X POST http://127.0.0.1:8000/api/memories \
     -H "Content-Type: application/json" \
     -d '{"content":"Performance test memory","tags":["perf-test"]}' \
     -w "\n%{time_total}\n"
   ```

2. **Search operation:**
   ```bash
   time curl -s "http://127.0.0.1:8000/api/search" \
     -H "Content-Type: application/json" \
     -d '{"query":"performance test","limit":10}' \
     -w "\n%{time_total}\n"
   ```

3. **Tag search operation:**
   ```bash
   time curl -s "http://127.0.0.1:8000/api/search/by-tag" \
     -H "Content-Type: application/json" \
     -d '{"tags":["perf-test"]}' \
     -w "\n%{time_total}\n"
   ```

4. **Delete operation:**
   ```bash
   HASH=$(curl -s "http://127.0.0.1:8000/api/search/by-tag" \
     -H "Content-Type: application/json" \
     -d '{"tags":["perf-test"]}' | jq -r '.memories[0].hash')

   time curl -s -X DELETE "http://127.0.0.1:8000/api/memories/$HASH" \
     -w "\n%{time_total}\n"
   ```

**Expected Results:**
- ✅ Store: <100ms
- ✅ Search: <200ms
- ✅ Tag search: <100ms
- ✅ Delete: <100ms

**Pass Criteria:**
- ✅ All operations <1 second
- ✅ HTTP 200 responses
- ✅ Correct response format
- ❌ FAIL if any operation >1 second

---

## Tag Filtering Correctness

### Test 7: Exact Tag Matching (No False Positives)

**Context:** v8.13.0 fixed tag filtering to prevent false positives (e.g., "python" shouldn't match "python3")

**Setup:**
1. Clear database
2. Store memories with similar tags

**Execution:**
```bash
# Store test memories
curl -X POST http://127.0.0.1:8000/api/memories \
  -H "Content-Type: application/json" \
  -d '{"content":"Python programming","tags":["python"]}'

curl -X POST http://127.0.0.1:8000/api/memories \
  -H "Content-Type: application/json" \
  -d '{"content":"Python 3 features","tags":["python3"]}'

curl -X POST http://127.0.0.1:8000/api/memories \
  -H "Content-Type: application/json" \
  -d '{"content":"CPython internals","tags":["cpython"]}'

curl -X POST http://127.0.0.1:8000/api/memories \
  -H "Content-Type: application/json" \
  -d '{"content":"Jython compatibility","tags":["jython"]}'

# Search for exact tag "python"
curl -s "http://127.0.0.1:8000/api/search/by-tag" \
  -H "Content-Type: application/json" \
  -d '{"tags":["python"]}' | jq '.memories | length'
```

**Expected Results:**
- ✅ Searching "python" returns exactly 1 memory
- ✅ Does NOT return python3, cpython, jython
- ✅ Exact substring boundary matching works

**Evidence Collection:**
```bash
# Test each tag variation
for tag in python python3 cpython jython; do
  echo "Testing tag: $tag"
  curl -s "http://127.0.0.1:8000/api/search/by-tag" \
    -H "Content-Type: application/json" \
    -d "{\"tags\":[\"$tag\"]}" | jq -r '.memories[].tags[]'
done
```

**Pass Criteria:**
- ✅ Each search returns only exact tag matches
- ✅ Zero false positives (substring matches)
- ✅ All 4 memories retrievable individually
- ❌ FAIL if any false positive occurs

---

### Test 8: Tag Index Usage (Performance)

**Context:** v8.13.0 added tag normalization with relational tables for O(log n) performance

**Setup:**
1. Database with 10,000+ memories
2. Verify migration completed: `python scripts/database/validate_migration.py`

**Execution:**
```bash
# Check query plan uses index
sqlite3 ~/Library/Application\ Support/mcp-memory/sqlite_vec.db <<EOF
EXPLAIN QUERY PLAN
SELECT DISTINCT m.*
FROM memories m
JOIN memory_tags mt ON m.id = mt.memory_id
JOIN tags t ON mt.tag_id = t.id
WHERE t.name = 'test-tag';
EOF
```

**Expected Results:**
- ✅ Query plan shows `SEARCH` (using index)
- ✅ Query plan does NOT show `SCAN` (table scan)
- ✅ Tag search completes in <200ms even with 10K memories

**Evidence Collection:**
```bash
# Verify index exists
sqlite3 ~/Library/Application\ Support/mcp-memory/sqlite_vec.db \
  "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_memory_tags_tag_id';"

# Benchmark tag search
time curl -s "http://127.0.0.1:8000/api/search/by-tag" \
  -H "Content-Type: application/json" \
  -d '{"tags":["test-tag"]}' -o /dev/null -w "%{time_total}\n"
```

**Pass Criteria:**
- ✅ Index exists and is used (SEARCH in query plan)
- ✅ Tag search <200ms with 10K+ memories
- ✅ Sub-linear scaling (2x data ≠ 2x time)
- ❌ FAIL if SCAN appears or >500ms with 10K memories

---

## MCP Protocol Compliance

### Test 9: MCP Tool Schema Validation

**Context:** Ensure all MCP tools conform to protocol schema

**Setup:**
1. Start MCP server: `memory server`
2. Use MCP Inspector: `npx @modelcontextprotocol/inspector memory server`

**Execution:**
1. Connect with MCP Inspector
2. List all tools: `tools/list`
3. Validate each tool schema:
   - Required fields present (name, description, inputSchema)
   - Input schema is valid JSON Schema
   - All parameters documented

**Expected Results:**
- ✅ All 13 core tools listed
- ✅ Each tool has valid JSON Schema
- ✅ No schema validation errors
- ✅ Tool descriptions are concise (<300 tokens each)

**Evidence Collection:**
```bash
# Capture tools/list output
npx @modelcontextprotocol/inspector memory server \
  --command "tools/list" > tools_schema.json

# Validate schema format
cat tools_schema.json | jq '.tools[] | {name, inputSchema}'
```

**Pass Criteria:**
- ✅ 13 tools exposed (26 after v8.13.0 consolidation → 13)
- ✅ All schemas valid JSON Schema Draft 07
- ✅ No missing required fields
- ❌ FAIL if any tool lacks proper schema

---

### Test 10: MCP Tool Execution

**Context:** Verify all tools execute correctly via MCP protocol

**Setup:**
1. MCP server running
2. MCP Inspector connected

**Execution:**
1. **Test store_memory:**
   ```json
   {
     "name": "store_memory",
     "arguments": {
       "content": "MCP protocol test memory",
       "tags": ["mcp-test", "protocol-validation"],
       "metadata": {"type": "test"}
     }
   }
   ```

2. **Test recall_memory:**
   ```json
   {
     "name": "recall_memory",
     "arguments": {
       "query": "last week",
       "n_results": 5
     }
   }
   ```

3. **Test search_by_tag:**
   ```json
   {
     "name": "search_by_tag",
     "arguments": {
       "tags": ["mcp-test"],
       "match_mode": "any"
     }
   }
   ```

4. **Test delete_by_tag:**
   ```json
   {
     "name": "delete_by_tag",
     "arguments": {
       "tags": ["mcp-test"],
       "match_mode": "all"
     }
   }
   ```

**Expected Results:**
- ✅ All tool calls return valid MCP responses
- ✅ No protocol errors or timeouts
- ✅ Response format matches tool schema
- ✅ Operations reflect in database

**Pass Criteria:**
- ✅ 4/4 tools execute successfully
- ✅ Responses valid JSON
- ✅ Database state matches operations
- ❌ FAIL if any tool returns error or invalid format

---

## Test Execution Guide

### Running All Regression Tests

```bash
# 1. Set up test environment
export MCP_MEMORY_STORAGE_BACKEND=sqlite_vec
export MCP_MEMORY_SQLITE_PRAGMAS=busy_timeout=15000,journal_mode=WAL

# 2. Clear test data
python scripts/database/reset_database.py --confirm

# 3. Run automated tests
pytest tests/unit/test_exact_tag_matching.py
pytest tests/unit/test_query_plan_validation.py
pytest tests/unit/test_performance_benchmark.py

# 4. Run manual tests (follow each test's Execution section)
# - Document results in checklist format
# - Capture evidence (logs, screenshots, timing data)
# - Mark pass/fail for each test

# 5. Generate test report
python scripts/testing/generate_regression_report.py \
  --output docs/testing/regression-report-$(date +%Y%m%d).md
```

### Test Frequency

- **Pre-Release:** All regression tests MUST pass
- **Post-PR Merge:** Run affected test categories
- **Weekly:** Automated subset (performance, tag filtering)
- **Monthly:** Full regression suite

### Reporting Issues

If any test fails:
1. Create GitHub issue with label `regression`
2. Include test name, evidence, and reproduction steps
3. Link to relevant commit/PR that may have caused regression
4. Add to release blockers if critical functionality affected

---

## Appendix: Test Data Generation

### Create Large Test Dataset

```bash
# Generate 10,000 test memories for performance testing
python scripts/testing/generate_test_data.py \
  --count 10000 \
  --tags-per-memory 3 \
  --output test-data-10k.json

# Import into database
curl -X POST http://127.0.0.1:8000/api/memories/batch \
  -H "Content-Type: application/json" \
  -d @test-data-10k.json
```

### Cleanup Test Data

```bash
# Remove all test data by tag
curl -X POST http://127.0.0.1:8000/api/memories/delete-by-tag \
  -H "Content-Type: application/json" \
  -d '{"tags": ["test", "perf-test", "mcp-test", "hybrid-test", "switch-test"], "match_mode": "any"}'
```

---

**Last Updated:** 2025-11-05
**Version:** 1.0
**Related:** [Release Checklist](../development/release-checklist.md), [PR Review Guide](../development/pr-review-guide.md)
