# MCP Memory Service Documentation

Welcome to the comprehensive documentation for MCP Memory Service - a Model Context Protocol server that provides semantic memory and persistent storage capabilities for Claude Desktop and other MCP clients.

## Quick Start

- **New Users**: Start with the [Setup Guide](setup-guide.md)
- **Multi-Client Setup**: See [Multi-Client Integration](integration/multi-client.md)
- **Docker Users**: Check out [Docker Deployment](deployment/docker.md)
- **Troubleshooting**: Visit [General Troubleshooting](troubleshooting/general.md)

## Documentation Structure

### 📦 Installation & Setup

- **[Setup Guide](setup-guide.md)** - Comprehensive installation instructions for all platforms

### 🔗 Integration & Connectivity

- **[Multi-Client Setup](integration/multi-client.md)** - Share memory across multiple applications
- **[Homebrew Integration](integration/homebrew.md)** - Use system-installed PyTorch via Homebrew
- **[IDE Compatibility](ide-compatability.md)** - VS Code, Continue, and other IDE integrations
- **[Gemini Integration](integrations/gemini.md)** - Use MCP Memory Service with Google Gemini
- **[Groq Bridge](integrations/groq-bridge.md)** · **[Groq Integration Summary](integrations/groq-integration-summary.md)** · **[Groq Model Comparison](integrations/groq-model-comparison.md)** - Groq LLM integration for quality scoring

### 🚀 Deployment

- **[Docker Deployment](deployment/docker.md)** - Containerized deployment with various configurations
- **[Cloud Deployment](glama-deployment.md)** - Cloud platform deployment guides
- **[Production Guide](deployment/production-guide.md)** - Single-process production deployment
- **[Dual-Service Deployment](deployment/dual-service.md)** - FastMCP + HTTP two-service architecture
- **[External Embeddings](deployment/external-embeddings.md)** - Use vLLM, Ollama, TEI or OpenAI-compatible embedding APIs
- **[systemd Service](deployment/systemd-service.md)** - Linux systemd service unit setup

### 📚 User Guides

- **[MCP Protocol Enhancements](guides/mcp-enhancements.md)** - Resources, Prompts, and Progress Tracking (v4.1.0)
- **[Storage Backends](guides/STORAGE_BACKENDS.md)** - SQLite-vec / Cloudflare / Hybrid comparison and configuration
- **[Migration Guide](guides/migration.md)** - Migrate between storage backends and versions
- **[Scripts Reference](guides/scripts.md)** - Available utility scripts

### 🎯 Natural Memory Triggers

- **[Installation Guide](natural-memory-triggers/installation-guide.md)** - Set up automatic memory triggers in Claude Code
- **[CLI Reference](natural-memory-triggers/cli-reference.md)** - Trigger CLI command reference
- **[Performance Optimization](natural-memory-triggers/performance-optimization.md)** - Tune trigger sensitivity and latency

### 🏛️ Architecture & Design

- **[Graph Database Design](architecture/graph-database-design.md)** - Memory graph schema and relationships
- **[Search Enhancement Spec](architecture/search-enhancement-spec.md)** - Hybrid search architecture
- **[Search Examples](architecture/search-examples.md)** - Worked examples of search modes

### 🎯 Tutorials & Examples

- **[Data Analysis Examples](tutorials/data-analysis.md)** - Advanced data analysis with memory service
- **[Advanced Techniques](tutorials/advanced-techniques.md)** - Power user techniques and patterns
- **[Demo Session Walkthrough](tutorials/demo-session-walkthrough.md)** - Step-by-step usage examples

### 🔧 Maintenance & Administration

- **[Memory Maintenance](maintenance/memory-maintenance.md)** - Database cleanup, optimization, and backup
- **[Health Checks](implementation/health_checks.md)** - Monitoring and diagnostics
- **[Performance Tuning](implementation/performance.md)** - Optimization techniques

### 📖 API Reference

- **[Memory Metadata API](api/memory-metadata-api.md)** - Advanced metadata operations
- **[Tag Standardization](api/tag-standardization.md)** - Tag schema and conventions
- **[Token-Efficient Retrieval](guides/token-efficient-retrieval.md)** - Keep retrieval cheap in long sessions
- **[Memory Type Ontology](memory-ontology.md)** - Built-in taxonomy and the `MCP_CUSTOM_MEMORY_TYPES` env var
- **[Team Setup](setup-guide.md#path-4-full-stack)** - Full-stack path for shared deployments
- **[Wiki](https://github.com/doobidoo/mcp-memory-service/wiki)** - Programmatic usage and long-form guides

### 🎬 Media & External

- **[Technical Video Demo (2 min)](https://www.youtube.com/watch?v=veJME5qVu-A)** - Performance, architecture, AI/ML intelligence
- **[Ask DeepWiki](https://deepwiki.com/doobidoo/mcp-memory-service)** - AI-powered documentation assistant
- **[MCP Starter Kit](https://kruppster57.gumroad.com/l/glbhd)** - Build your own MCP server using the patterns from this project

### 🛠️ Development & Technical

- **[Development Guide](technical/development.md)** - Contributing and development setup
- **[Technical Implementation](technical/)** - Deep dive into technical details
  - [Memory Migration](technical/memory-migration.md)
  - [Tag Storage](technical/tag-storage.md)

### 🔍 Troubleshooting

- **[General Troubleshooting](troubleshooting/general.md)** - Common issues and solutions
- **[Docker Issues](deployment/docker.md#troubleshooting)** - Docker-specific troubleshooting

## Project Information

### About MCP Memory Service

MCP Memory Service enables persistent, semantic memory for AI applications through the Model Context Protocol. It provides:

- **Semantic Search**: Vector-based memory retrieval using sentence transformers
- **Multiple Storage Backends**: SQLite-vec (local), Cloudflare (cloud/edge), and Hybrid (recommended for production)
- **Multi-Client Support**: Shared memory across multiple applications
- **Cross-Platform**: Support for macOS, Windows, and Linux
- **Flexible Deployment**: Local installation, Docker containers, or cloud deployment

### Key Features

- ✅ **Semantic Memory Storage**: Store and retrieve memories using natural language
- ✅ **Multi-Client Access**: Share memories across Claude Desktop, VS Code, and other MCP clients
- ✅ **Flexible Storage**: Choose SQLite-vec (local/dev), Cloudflare (cloud/edge), or Hybrid (production)
- ✅ **Cross-Platform**: Native support for macOS (Intel & Apple Silicon), Windows, and Linux
- ✅ **Docker Ready**: Complete containerization support with multiple deployment options
- ✅ **Hardware Optimized**: Automatic detection and optimization for available hardware (CUDA, MPS, DirectML)
- ✅ **Production Ready**: HTTP/SSE API, authentication, monitoring, and scaling features

### Release History

Every release is in [CHANGELOG.md](../CHANGELOG.md); older entries live in
[docs/archive/CHANGELOG-HISTORIC.md](archive/CHANGELOG-HISTORIC.md).

## Getting Help

### Quick Links

- **Installation Issues**: Check the [Setup Guide](setup-guide.md) and platform-specific guides
- **Configuration Problems**: See [Troubleshooting](troubleshooting/general.md)
- **Multi-Client Setup**: Follow the [Multi-Client Guide](integration/multi-client.md)
- **Performance Issues**: Review [Performance Tuning](implementation/performance.md)

### Support Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides for all use cases
- **Community**: Share experiences and get help from other users

### Contributing

We welcome contributions! See the [Development Guide](technical/development.md) for information on:

- Setting up a development environment
- Running tests
- Submitting pull requests
- Code style and conventions

---

## Navigation Tips

- **📁 Folders**: Click on folder names to explore sections
- **🔗 Links**: All internal links are relative and work offline
- **📱 Mobile**: Documentation is mobile-friendly for on-the-go reference
- **🔍 Search**: Use your browser's search (Ctrl/Cmd+F) to find specific topics

**Happy memory-ing! 🧠✨**