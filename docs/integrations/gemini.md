# Gemini Context: MCP Memory Service

## Project Overview

This project is a sophisticated and feature-rich MCP (Memory Component Protocol) server designed to provide a persistent, semantic memory layer for AI assistants, particularly "Claude Desktop". It's built with Python and leverages a variety of technologies to deliver a robust and performant memory service.

The core of the project is the `MemoryServer` class, which handles all MCP tool calls. It features a "dream-inspired" memory consolidation system that autonomously organizes and compresses memories over time. The server is built on top of FastAPI, providing a modern and asynchronous API.

The project offers two distinct storage backends, allowing users to choose the best fit for their needs:

*   **ChromaDB:** A feature-rich vector database that provides advanced search capabilities and is well-suited for large memory collections.
*   **SQLite-vec:** A lightweight, file-based backend that uses the `sqlite-vec` extension for vector similarity search. This is a great option for resource-constrained environments.

The project also includes a comprehensive suite of scripts for installation, testing, and maintenance, as well as detailed documentation.

## Building and Running

### Installation

The project uses a custom installer that intelligently detects the user's hardware and selects the optimal configuration. To install the project, run the following commands:

```bash
# Clone the repository
git clone https://github.com/doobidoo/mcp-memory-service.git
cd mcp-memory-service

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

# Install in editable mode (use the [full] extra to get all optional dependencies)
pip install -e ".[full]"
```

### Running the Server

The server can be run in several ways, depending on the desired configuration. The primary entry point is the `mcp_memory_service.server:main` script, which can be executed as a Python module:

```bash
python -m mcp_memory_service.server
```

Alternatively, the `pyproject.toml` file defines a `memory` script that can be used to run the server:

```bash
memory
```

The server can also be run as a FastAPI application using `uvicorn`:

```bash
uvicorn mcp_memory_service.server:app --host 0.0.0.0 --port 8000
```

### Testing

The project includes a comprehensive test suite that can be run using `pytest`:

```bash
pytest tests/
```

## Development Conventions

*   **Python 3.10+:** The project requires Python 3.10 or higher.
*   **Type Hinting:** The codebase uses type hints extensively to improve code clarity and maintainability.
*   **Async/Await:** The project uses the `async/await` pattern for all I/O operations to ensure high performance and scalability.
*   **PEP 8:** The code follows the PEP 8 style guide.
*   **Dataclasses:** The project uses dataclasses for data models to reduce boilerplate code.
*   **Docstrings:** All modules and functions have triple-quoted docstrings that explain their purpose, arguments, and return values.
*   **Testing:** All new features should be accompanied by tests to ensure they work as expected and don't introduce regressions.
