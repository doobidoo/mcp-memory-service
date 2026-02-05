# Scripts Documentation

This document provides an overview of the available scripts in the `scripts/` directory and their purposes.

## Essential Scripts

### Server Management
- `run_memory_server.py`: Main script to start the memory service server
  ```bash
  python scripts/run_memory_server.py
  ```

### Environment Verification
- `verify_environment.py`: Verifies the installation environment and dependencies
  ```bash
  python scripts/verify_environment.py
  ```

### Installation Testing
- `test_installation.py`: Tests the installation and basic functionality
  ```bash
  python scripts/test_installation.py
  ```

### Memory Management
- `validate_memories.py`: Validates the integrity of stored memories
  ```bash
  python scripts/validate_memories.py
  ```
- `repair_memories.py`: Repairs corrupted or invalid memories
  ```bash
  python scripts/repair_memories.py
  ```
- `list-collections.py`: Lists all available memory collections
  ```bash
  python scripts/list-collections.py
  ```

## Migration Scripts
- `mcp-migration.py`: Handles migration of MCP-related data
  ```bash
  python scripts/mcp-migration.py
  ```
- `memory-migration.py`: Handles migration of memory data
  ```bash
  python scripts/memory-migration.py
  ```

## Troubleshooting Scripts
- `verify_pytorch_windows.py`: Verifies PyTorch installation on Windows
  ```bash
  python scripts/verify_pytorch_windows.py
  ```
- `verify_torch.py`: General PyTorch verification
  ```bash
  python scripts/verify_torch.py
  ```

## Usage Notes
- Most scripts can be run directly with Python
- Some scripts may require specific environment variables to be set
- Always run verification scripts after installation or major updates
- Use migration scripts with caution and ensure backups are available

## Script Dependencies
- Python 3.10+
- Required packages listed in `requirements.txt`
- Some scripts may require additional dependencies listed in `requirements-migration.txt`
