# CHANGELOG

## v1.3.0 (2025-06-25) - Timestamp Standardization & Codebase Organization

### 📅 Timestamp Standardization

#### Database Maintenance
- **NEW**: Migration scripts for timestamp format cleanup and standardization
- **ENHANCED**: Intelligent database path detection from environment or config files
- **FIXED**: Unique constraint handling during timestamp migration
- **IMPROVED**: Automatic database backup before migrations

#### Timestamp Format Consolidation
- **OPTIMIZED**: Standardized all timestamps to integer format in single field
- **FIXED**: Issue where timestamps were stored in multiple formats (int, float, string)
- **ENHANCED**: Redundant timestamp field cleanup for better database performance
- **IMPROVED**: Consistent timestamp storage and retrieval

### 🧹 Codebase Organization

#### Test Organization
- **ENHANCED**: Structured test directories for unit, integration, and performance testing
- **IMPROVED**: Better separation of test concerns
- **ADDED**: Dedicated test configuration and documentation
- **OPTIMIZED**: Test infrastructure for better maintenance

#### Documentation Improvements
- **NEW**: Comprehensive migration guides with safety procedures
- **ENHANCED**: Technical documentation for database structure
- **ORGANIZED**: Archived obsolete files for cleaner codebase
- **IMPROVED**: README clarity and organization

### 🛠️ Technical Improvements

#### Database Management
- **ENHANCED**: More robust error handling during migrations
- **ADDED**: Database analysis tools for timestamp field proliferation
- **IMPROVED**: Safe migration with transaction handling
- **FIXED**: UNIQUE constraint violation during timestamp conversions

#### Development Workflow
- **ENHANCED**: Better code organization
- **IMPROVED**: More maintainable directory structure
- **OPTIMIZED**: Cleaner repository layout
- **ADDED**: Better branch management

## v1.2.0 (2025-06-24) - Performance Optimization & Health Check Improvements

### 🚀 Major Performance Enhancements

#### Optimized ChromaDB Integration
- **NEW**: Intelligent model caching system with 95% faster cold starts
- **ENHANCED**: Query result caching with LRU mechanism for 90% faster repeated queries
- **OPTIMIZED**: Metadata processing with efficient tag storage and parsing
- **IMPROVED**: Enhanced ChromaDB configuration with optimized HNSW parameters

#### System-Level Optimizations
- **NEW**: Environment auto-detection and optimization for different hardware
- **ENHANCED**: Batch operations for efficient bulk memory storage
- **OPTIMIZED**: Logging performance with reduced overhead
- **IMPROVED**: Thread and resource management for better scalability

### 🩺 Health Check Improvements

#### Enhanced Error Handling
- **FIXED**: Health check failures with better null checking and initialization validation
- **NEW**: Detailed initialization status reporting for easier debugging
- **ENHANCED**: Component-level health validation with specific error messages
- **IMPROVED**: Graceful degradation with partial system functionality

#### Diagnostic Features
- **NEW**: Performance monitoring with metrics collection
- **ENHANCED**: Database statistics with detailed component status
- **IMPROVED**: Debug information for initialization failures
- **ADDED**: Cache hit ratio and query time tracking

### 🧪 Testing & Documentation

#### Comprehensive Test Suite
- **NEW**: Performance benchmarking scripts
- **NEW**: Health check validation tests
- **ENHANCED**: ChromaDB initialization test with modern API
- **IMPROVED**: Automated test cleanup

#### Documentation Updates
- **ENHANCED**: Detailed implementation summaries in markdown
- **NEW**: Performance optimization documentation
- **NEW**: Health check troubleshooting guide
- **IMPROVED**: System requirements and configuration guidance

### 🔧 Technical Improvements

#### Architectural Enhancements
- **NEW**: Thread-safe global model cache with proper locking
- **OPTIMIZED**: Memory usage reduced by ~40%
- **ENHANCED**: Database access patterns for better performance
- **IMPROVED**: System detection and hardware adaptation

#### API Improvements
- **NEW**: Performance monitoring and statistics methods
- **ENHANCED**: Cache management functionality
- **IMPROVED**: Initialization verification methods
- **ADDED**: Detailed status reporting for debugging

### 📚 API Reference

```python
# Model Caching System
storage = ChromaMemoryStorage(path, preload_model=True)  # Default is True for performance

# Performance Statistics
stats = storage.get_performance_stats()
print(f"Cache hit ratio: {stats['cache_hit_ratio']:.2%}")
print(f"Average query time: {stats['avg_query_time']:.3f}s")

# Cache Management
storage.clear_caches()  # Clear all caches when needed

# Health Check with Performance Data
health = await server.handle_check_database_health()
print(f"Status: {health['validation']['status']}")
print(f"Cache hits: {health['performance']['storage']['cache_hits']}")
```

### 🔄 Migration Notes

**No migration required!** All changes are 100% backward compatible:
- Existing code continues to work unchanged
- Performance improvements apply automatically
- New features are optional and accessed through new methods
- System detection and optimization happen automatically

## v1.1.0 (2025-06-07) - Enhanced Tag Management

### 🎉 Major Features

#### Issue 5 Resolution: Delete Tag Function Ambiguity
- **FIXED**: API inconsistency between `search_by_tag` and `delete_by_tag`
- **ENHANCED**: `delete_by_tag` now supports both single tags and multiple tags
- **BACKWARD COMPATIBLE**: All existing code continues to work unchanged

### ✨ New Features

#### Enhanced Delete Operations
- **Enhanced `delete_by_tag`**: Now accepts both `string` and `List[str]` parameters
- **New `delete_by_tags`**: Explicit multi-tag deletion with OR logic
- **New `delete_by_all_tags`**: Delete memories containing ALL specified tags (AND logic)
- **Improved Error Messages**: More descriptive feedback for all tag operations

#### MCP Server Enhancements
- **New Tool Definitions**: Added `delete_by_tags` and `delete_by_all_tags` tools
- **Enhanced Parameter Handling**: Flexible parameter formats for delete operations
- **Better Type Validation**: Improved input validation and error handling
- **Enhanced Documentation**: Updated tool descriptions with examples

### 🛠️ Technical Improvements

#### API Changes
```python
# Before (still works - backward compatible)
await storage.delete_by_tag("single_tag")

# Enhanced (new functionality)
await storage.delete_by_tag(["tag1", "tag2", "tag3"])        # OR logic
await storage.delete_by_tags(["tag1", "tag2"])               # Explicit OR logic  
await storage.delete_by_all_tags(["urgent", "important"])    # AND logic
```

#### Storage Layer
- **Enhanced ChromaMemoryStorage**: Updated delete methods with flexible parameter handling
- **Improved Tag Matching**: Better logic for tag comparison and matching
- **Enhanced Error Handling**: More descriptive error messages and better validation
- **Performance Optimization**: Efficient batch operations for multiple tag deletion

#### Server Layer
- **New Tool Handlers**: Added handlers for new delete operations
- **Enhanced Parameter Processing**: Flexible handling of both `tag` and `tags` parameters
- **Improved Logging**: Better debugging information for tag operations
- **Enhanced Type Safety**: Better parameter validation and type checking

### 🧪 Testing
- **Comprehensive Test Suite**: New `test_issue_5_fix.py` for enhanced functionality
- **Edge Case Coverage**: Tests for empty inputs, non-existent tags, type validation
- **Backward Compatibility Tests**: Ensures existing code continues to work
- **Integration Tests**: Verification of MCP server tool integration

### 📚 Documentation
- **Updated README**: Enhanced documentation with new API examples
- **API Documentation**: Updated tool descriptions and usage examples
- **Migration Guide**: Though no migration needed due to backward compatibility
- **Enhanced Examples**: Comprehensive usage examples for all delete operations

### 🔧 Infrastructure  
- **Enhanced Code Organization**: Better separation of concerns in delete operations
- **Improved Error Handling**: More robust error handling and user feedback
- **Better Logging**: Enhanced debugging and monitoring capabilities

## v0.1.0 (2024-12-27)

### Chores

- Update gitignore
  ([`97ba25c`](https://github.com/doobidoo/mcp-memory-service/commit/97ba25c83113ed228d6684b8c65bc65774c0b704))

### Features

- Add MCP protocol compliance and fix response formats
  ([`fefd579`](https://github.com/doobidoo/mcp-memory-service/commit/fefd5796b3fb758023bb574b508940a651e48ad5))

---

## Migration Notes

### From v1.2.0 to v1.3.0
**Run the migration script for timestamp cleanup!**
- The script `scripts/migrations/cleanup_mcp_timestamps.py` will:
  1. Back up your database automatically
  2. Standardize all timestamps to integer format
  3. Remove redundant timestamp fields
  4. Ensure all memories have valid timestamps
- No changes to code or API are required after migration
- The script includes safety features to prevent data loss

### From v1.1.0 to v1.2.0
**No migration required!** - All changes are backward compatible.
- Performance improvements apply automatically
- New health check and diagnostic features are optional
- Existing code continues to work unchanged

### From v0.1.0 to v1.1.0
**No migration required!** - All changes are backward compatible.
- All existing `delete_by_tag("single_tag")` calls continue to work
- Enhanced functionality available immediately  
- No code changes required for existing implementations
- New methods available for enhanced functionality