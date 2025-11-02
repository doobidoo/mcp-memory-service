# MCP Memory Service Makefile

.PHONY: help start stop restart logs test install validate health clean

# Load .env file if it exists
ifneq (,$(wildcard .env))
	include .env
	export
endif

# Defaults (override with .env or environment variables)
MCP_HTTP_ENABLED ?= true
MCP_HTTP_PORT ?= 8888
MCP_MEMORY_STORAGE_BACKEND ?= hybrid
MCP_OAUTH_ENABLED ?= false

# Python interpreter - use .venv if available
VENV_PATH := .venv
VENV_PYTHON := $(VENV_PATH)/bin/python
VENV_ACTIVATE := . $(VENV_PATH)/bin/activate

# Use venv python if available, otherwise fallback to system python
PYTHON := $(shell [ -f $(VENV_PYTHON) ] && echo $(VENV_PYTHON) || echo python3)
UV := uv

# ANSI colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)MCP Memory Service - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make $(GREEN)<target>$(NC)\n\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

start: ## Start the HTTP server with default configuration
	@echo "$(GREEN)Starting MCP Memory Service...$(NC)"
	@echo "$(BLUE)Configuration:$(NC)"
	@echo "  Python: $(PYTHON)"
	@echo "  Storage Backend: $(MCP_MEMORY_STORAGE_BACKEND)"
	@echo "  HTTP Port: $(MCP_HTTP_PORT)"
	@echo "  OAuth Enabled: $(MCP_OAUTH_ENABLED)"
	@if [ -f $(VENV_PATH)/bin/activate ]; then \
		$(VENV_ACTIVATE) && \
		MCP_HTTP_ENABLED=$(MCP_HTTP_ENABLED) \
		MCP_HTTP_PORT=$(MCP_HTTP_PORT) \
		MCP_MEMORY_STORAGE_BACKEND=$(MCP_MEMORY_STORAGE_BACKEND) \
		MCP_OAUTH_ENABLED=$(MCP_OAUTH_ENABLED) \
		$(VENV_PYTHON) run_server.py; \
	else \
		echo "$(YELLOW)Warning: .venv not found, using system Python$(NC)"; \
		MCP_HTTP_ENABLED=$(MCP_HTTP_ENABLED) \
		MCP_HTTP_PORT=$(MCP_HTTP_PORT) \
		MCP_MEMORY_STORAGE_BACKEND=$(MCP_MEMORY_STORAGE_BACKEND) \
		MCP_OAUTH_ENABLED=$(MCP_OAUTH_ENABLED) \
		$(PYTHON) run_server.py; \
	fi

dev: ## Start server with auto-reload for development
	@echo "$(GREEN)Starting MCP Memory Service in development mode...$(NC)"
	@if [ -f $(VENV_PATH)/bin/activate ]; then \
		$(VENV_ACTIVATE) && \
		MCP_HTTP_ENABLED=$(MCP_HTTP_ENABLED) \
		MCP_HTTP_PORT=$(MCP_HTTP_PORT) \
		MCP_MEMORY_STORAGE_BACKEND=$(MCP_MEMORY_STORAGE_BACKEND) \
		MCP_OAUTH_ENABLED=$(MCP_OAUTH_ENABLED) \
		uvicorn mcp_memory_service.web.app:app --reload --host 0.0.0.0 --port $(MCP_HTTP_PORT); \
	else \
		echo "$(YELLOW)Warning: .venv not found, using uv run$(NC)"; \
		MCP_HTTP_ENABLED=$(MCP_HTTP_ENABLED) \
		MCP_HTTP_PORT=$(MCP_HTTP_PORT) \
		MCP_MEMORY_STORAGE_BACKEND=$(MCP_MEMORY_STORAGE_BACKEND) \
		MCP_OAUTH_ENABLED=$(MCP_OAUTH_ENABLED) \
		$(UV) run uvicorn mcp_memory_service.web.app:app --reload --host 0.0.0.0 --port $(MCP_HTTP_PORT); \
	fi

mcp: ## Start MCP server (stdio mode for Claude Desktop)
	@echo "$(GREEN)Starting MCP Server (stdio mode)...$(NC)"
	@if [ -f $(VENV_PATH)/bin/activate ]; then \
		$(VENV_ACTIVATE) && memory server -s $(MCP_MEMORY_STORAGE_BACKEND); \
	else \
		$(UV) run memory server -s $(MCP_MEMORY_STORAGE_BACKEND); \
	fi

stop: ## Stop the running HTTP server
	@echo "$(YELLOW)Stopping MCP Memory Service...$(NC)"
	@pkill -f "run_server.py" || echo "$(RED)No server process found$(NC)"

restart: stop start ## Restart the HTTP server

health: ## Check service health
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -s http://127.0.0.1:$(MCP_HTTP_PORT)/api/health | jq . || echo "$(RED)Service not responding$(NC)"

logs: ## Show recent logs (if using systemd)
	@if command -v journalctl >/dev/null 2>&1; then \
		journalctl --user -u mcp-memory-http.service -n 50 -f; \
	else \
		echo "$(YELLOW)Systemd not available. Run 'make start' to see logs in foreground.$(NC)"; \
	fi

test: ## Run test suite
	@echo "$(BLUE)Running tests...$(NC)"
	@pytest tests/ -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	@pytest tests/ --cov=mcp_memory_service --cov-report=html --cov-report=term

install: ## Install dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	@$(UV) pip install -e .

validate: ## Validate configuration
	@echo "$(BLUE)Validating configuration...$(NC)"
	@$(PYTHON) scripts/validation/validate_configuration_complete.py

validate-backend: ## Validate backend configuration
	@echo "$(BLUE)Validating backend configuration...$(NC)"
	@$(PYTHON) scripts/validation/diagnose_backend_config.py

sync-status: ## Check backend sync status (hybrid mode)
	@echo "$(BLUE)Checking sync status...$(NC)"
	@$(PYTHON) scripts/sync/sync_memory_backends.py --status

sync-dry-run: ## Preview backend synchronization
	@echo "$(BLUE)Preview sync (dry run)...$(NC)"
	@$(PYTHON) scripts/sync/sync_memory_backends.py --dry-run

backup: ## Backup memories from Cloudflare to SQLite
	@echo "$(GREEN)Backing up memories...$(NC)"
	@$(PYTHON) scripts/sync/claude_sync_commands.py backup

restore: ## Restore memories from SQLite to Cloudflare
	@echo "$(GREEN)Restoring memories...$(NC)"
	@$(PYTHON) scripts/sync/claude_sync_commands.py restore

clean: ## Clean up generated files and caches
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache htmlcov .coverage
	@echo "$(GREEN)Cleanup complete$(NC)"

lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	@$(UV) run ruff check src/ tests/

format: ## Format code
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(UV) run ruff format src/ tests/

check: validate-backend lint test ## Run all checks (validate, lint, test)
	@echo "$(GREEN)All checks passed!$(NC)"

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	@docker build -t mcp-memory-service .

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	@docker run -p $(MCP_HTTP_PORT):$(MCP_HTTP_PORT) --env-file .env mcp-memory-service

.DEFAULT_GOAL := help
