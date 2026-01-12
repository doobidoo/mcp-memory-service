# Refactoring Safety Checklist

## ⚠️ MANDATORY for All Code Moves/Extractions

When extracting, moving, or refactoring code, follow this checklist to avoid production issues.

**Learned from**: Issues #299 (import errors), #300 (response format mismatch)

---

## 1. ✅ Import Path Validation

**Problem**: Relative imports break when code moves (`..services` → `...services`)

**Checklist**:
- [ ] Validate relative imports from new location
- [ ] Run `bash scripts/ci/validate_imports.sh` before commit
- [ ] Test actual imports (no mocks allowed for validation)

**Why**: 82% of import errors are undetected until production

---

## 2. ✅ Response Format Compatibility

**Problem**: Handler expects `result["message"]` but service returns `result["success"]`

**Checklist**:
- [ ] Handler matches service response keys (`success`/`error`, not `message`)
- [ ] Test both success AND error paths
- [ ] Check for KeyError risks in all code paths

**Why**: Response format mismatches cause runtime crashes

---

## 3. ✅ Integration Tests for ALL Extracted Functions

**Problem**: 82% of handlers had zero integration tests (3/17 tested)

**Checklist**:
- [ ] Create integration tests BEFORE committing refactoring
- [ ] 100% handler coverage required (17/17 handlers)
- [ ] Run `python scripts/validation/check_handler_coverage.py`

**Why**: Unit tests alone don't catch integration issues

---

## 4. ✅ Coverage Validation

**Problem**: Refactoring can inadvertently reduce test coverage

**Checklist**:
- [ ] Run `pytest --cov=src/mcp_memory_service --cov-fail-under=80`
- [ ] Coverage must not decrease (delta ≥ 0%)
- [ ] Add tests for new code before committing

**Why**: Coverage gate prevents untested code from merging

---

## 5. ✅ Pre-Commit Validation

**Run these commands before EVERY refactoring commit:**

```bash
# 1. Import validation
bash scripts/ci/validate_imports.sh

# 2. Handler coverage check
python scripts/validation/check_handler_coverage.py

# 3. Coverage gate
pytest tests/ --cov=src --cov-fail-under=80
```

**All must pass** before git commit.

---

## 6. ✅ Commit Strategy

**Problem**: Batching multiple extractions makes errors hard to trace

**Checklist**:
- [ ] Commit incrementally (one extraction per commit)
- [ ] Each commit has passing tests + coverage ≥80%
- [ ] Never batch multiple extractions in one commit

**Why**: Incremental commits = easy rollback if issues found

---

## Quick Command Reference

```bash
# Full pre-refactoring validation
bash scripts/ci/validate_imports.sh && \
python scripts/validation/check_handler_coverage.py && \
pytest tests/ --cov=src --cov-fail-under=80

# If all pass → safe to commit
git commit -m "refactor: extracted X function"
```

---

## Historical Context

**Issue #299**: Import error `..services` → `...services` undetected until production
**Issue #300**: Response format mismatch `result["message"]` vs `result["success"]`
**Root Cause**: 82% of handlers had zero integration tests (3/17 tested)
**Solution**: 9-check pre-PR validation + 100% handler coverage requirement

**Prevention is better than debugging in production.**
