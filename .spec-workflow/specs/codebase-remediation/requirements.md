# Requirements Document: Codebase Remediation

## Introduction

This spec defines the requirements for remediating the MCP Memory Service codebase. The service has grown to 34,233 lines with ~45% dead code, 3 CRITICAL security vulnerabilities, 4 unused storage backends, and 15 configuration classes. This remediation will reduce the codebase to ~2,000 lines while maintaining all core functionality.

## Alignment with Product Vision

Per `product.md`, this remediation directly supports:
- **Reduce Technical Debt**: 94% line reduction (34K → ~2K)
- **Eliminate Security Risk**: Fix all CRITICAL/HIGH vulnerabilities
- **Simplify Operations**: 1-2 storage backends vs 4
- **Enable V2**: Clean foundation for GPU acceleration, memory graphs

## Requirements

### REQ-1: Security Vulnerability Elimination

**User Story:** As a self-hoster, I want all CRITICAL and HIGH security vulnerabilities fixed, so that my memory service doesn't expose my infrastructure to attack.

#### Acceptance Criteria

1. WHEN the service starts THEN the system SHALL NOT load any code with CRITICAL vulnerabilities (authlib, python-jose)
2. WHEN anonymous access is configured THEN the system SHALL grant read-only permissions, NOT admin access
3. WHEN CORS is not explicitly configured THEN the system SHALL default to empty allowed origins `[]`, NOT wildcard `['*']`
4. WHEN the service runs THEN the system SHALL NOT expose mDNS discovery on the network

### REQ-2: Dead Code Removal

**User Story:** As a maintainer, I want all dead code removed, so that the codebase is understandable and maintainable.

#### Acceptance Criteria

1. WHEN the codebase is analyzed THEN the system SHALL contain zero unused modules (consolidation/, discovery/, sync/, ingestion/, embeddings/onnx_*)
2. WHEN the codebase is analyzed THEN the system SHALL contain zero duplicate server implementations (only one entry point)
3. WHEN the codebase is analyzed THEN the system SHALL contain no disabled-by-default features (consolidation, forgetting, SSE events)
4. IF a module is not imported anywhere THEN the system SHALL NOT include it in the final codebase

### REQ-3: Storage Backend Consolidation

**User Story:** As an operator, I want only the storage backends I use, so that I don't maintain code for features I never deploy.

#### Acceptance Criteria

1. WHEN running in production THEN the system SHALL support Qdrant as the primary storage backend
2. WHEN running in development THEN the system SHALL support SQLite-vec as a zero-dependency local option
3. WHEN the codebase is analyzed THEN the system SHALL NOT contain Cloudflare, Hybrid, or HTTP client storage backends
4. WHEN storage is configured THEN the system SHALL use a `typing.Protocol` interface for backend abstraction

### REQ-4: Configuration Simplification

**User Story:** As a developer, I want clear, minimal configuration, so that I can understand and modify settings without confusion.

#### Acceptance Criteria

1. WHEN configuration is loaded THEN the system SHALL use at most 4 Settings classes (Core, Qdrant, SQLite, Embedding)
2. WHEN sensitive values are configured THEN the system SHALL use `SecretStr` for secrets (API keys, tokens)
3. WHEN configuration is loaded THEN the system SHALL validate all settings via pydantic-settings
4. IF a configuration option is unused THEN the system SHALL NOT include it

### REQ-5: Architecture Alignment

**User Story:** As a developer, I want a clean hexagonal architecture, so that business logic is isolated from protocol and storage concerns.

#### Acceptance Criteria

1. WHEN business logic executes THEN `service.py` SHALL NOT import MCP, HTTP, or storage implementation modules
2. WHEN a new protocol adapter is added THEN the system SHALL require changes only in the `api/` directory
3. WHEN a new storage backend is added THEN the system SHALL require changes only in the `storage/` directory
4. WHEN the core is tested THEN tests SHALL NOT require MCP or storage infrastructure

### REQ-6: Line Count Target

**User Story:** As a maintainer, I want a minimal codebase, so that cognitive load is reduced and bugs are fewer.

#### Acceptance Criteria

1. WHEN the remediation is complete THEN the codebase SHALL contain approximately 2,000 lines (±500)
2. WHEN individual files are measured THEN each SHALL stay within hard limits defined in `structure.md`
3. IF a file exceeds its hard limit THEN the system SHALL be refactored to distribute responsibility

## Non-Functional Requirements

### Code Architecture and Modularity
- **Single Responsibility Principle**: `service.py` handles business logic only; adapters handle protocol translation only
- **Modular Design**: Storage backends are interchangeable via Protocol interface
- **Dependency Management**: Dependency arrows flow inward (api → service → storage interface)
- **Clear Interfaces**: `BaseStorage` Protocol defines the storage contract

### Performance
- Embedding generation latency: Establish baseline, maintain or improve post-remediation
- Memory store/retrieve operations: Establish baseline, maintain or improve post-remediation
- Container startup time: Target under 30 seconds with model loading

### Security
- All sensitive configuration uses `SecretStr`
- No timing attacks on secret comparison (use `secrets.compare_digest`)
- Parameterized SQL queries only (no string interpolation)
- API key authentication for all non-health endpoints

### Reliability
- Health endpoint `/health` returns within 100ms
- Storage failures surface immediately (no silent failures)
- Graceful degradation if embedding model fails to load

### Usability
- Single `uv run memory server` command starts the service
- Clear error messages with actionable remediation steps
- Configuration validated at startup, not at first use
