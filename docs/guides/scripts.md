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
- `query_memories.py`: Queries the configured memory store from the command line
  ```bash
  python scripts/utils/query_memories.py --help
  ```

## Migration Scripts
Active migration scripts live in `scripts/migration/`; see the
[`scripts/README.md`](../../scripts/README.md) migration section for the current
list. Legacy ChromaDB data is converted on the `chromadb-legacy-final` (v7.6.0)
tag, as described in [`chromadb-migration.md`](chromadb-migration.md).

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