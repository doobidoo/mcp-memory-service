# MCP Memory Service Documentation

Welcome to the comprehensive documentation for MCP Memory Service - a Model Context Protocol server that provides semantic memory and persistent storage capabilities for Claude Desktop and other MCP clients.

## Quick Start

- **New Users**: Start with the [Installation Guide](installation/master-guide.md)
- **Multi-Client Setup**: See [Multi-Client Integration](integration/multi-client.md)
- **Docker Users**: Check out [Docker Deployment](deployment/docker.md)
- **Troubleshooting**: Visit [General Troubleshooting](troubleshooting/general.md)

## Documentation Structure

### 📦 Installation & Setup

- **[Master Installation Guide](installation/master-guide.md)** - Comprehensive installation instructions for all platforms
- **[Platform-Specific Guides](platforms/)** - Detailed setup for specific operating systems
  - [macOS Intel](platforms/macos-intel.md) - Intel Mac setup (including legacy 2013-2017 models)
  - [Windows](platforms/windows.md) - Windows installation with CUDA/DirectML support
  - [Ubuntu](platforms/ubuntu.md) - Ubuntu setup for desktop and server

### 🔗 Integration & Connectivity

- **[Multi-Client Setup](integration/multi-client.md)** - Share memory across multiple applications
- **[Homebrew Integration](integration/homebrew.md)** - Use system-installed PyTorch via Homebrew
- **[Claude Desktop Integration](guides/claude_integration.md)** - Connect with Claude Desktop
- **[IDE Compatibility](ide-compatability.md)** - VS Code, Continue, and other IDE integrations

### 🚀 Deployment

- **[Docker Deployment](deployment/docker.md)** - Containerized deployment with various configurations
- **[Server Deployment](deployment/multi-client-server.md)** - Production server setups
- **[Cloud Deployment](glama-deployment.md)** - Cloud platform deployment guides

### 📚 User Guides

- **[MCP Protocol Enhancements](guides/mcp-enhancements.md)** - Resources, Prompts, and Progress Tracking (v4.1.0)
- **[Storage Backends](guides/STORAGE_BACKENDS.md)** - ChromaDB vs SQLite-vec comparison and configuration
- **[Migration Guide](guides/migration.md)** - Migrate between storage backends and versions
- **[Scripts Reference](guides/scripts.md)** - Available utility scripts
- **[Invocation Guide](guides/invocation_guide.md)** - Different ways to run the service

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
- **[HTTP/SSE API](IMPLEMENTATION_PLAN_HTTP_SSE.md)** - Web API documentation for multi-client setups

### 🛠️ Development & Technical

- **[Development Guide](technical/development.md)** - Contributing and development setup
- **[Architecture Overview](development/multi-client-architecture.md)** - System architecture and design patterns
- **[Technical Implementation](technical/)** - Deep dive into technical details
  - [Memory Migration](technical/memory-migration.md)
  - [Tag Storage](technical/tag-storage.md)

### 🔍 Troubleshooting

- **[General Troubleshooting](troubleshooting/general.md)** - Common issues and solutions
- **[Docker Issues](deployment/docker.md#troubleshooting)** - Docker-specific troubleshooting
- **[Platform-Specific Issues](platforms/)** - Platform-specific troubleshooting sections

## Project Information

### About MCP Memory Service

MCP Memory Service enables persistent, semantic memory for AI applications through the Model Context Protocol. It provides:

- **Semantic Search**: Vector-based memory retrieval using sentence transformers
- **Multiple Storage Backends**: ChromaDB for full features, SQLite-vec for lightweight deployments
- **Multi-Client Support**: Shared memory across multiple applications
- **Cross-Platform**: Support for macOS, Windows, and Linux
- **Flexible Deployment**: Local installation, Docker containers, or cloud deployment

### Key Features

- ✅ **Semantic Memory Storage**: Store and retrieve memories using natural language
- ✅ **Multi-Client Access**: Share memories across Claude Desktop, VS Code, and other MCP clients
- ✅ **Flexible Storage**: Choose between ChromaDB (full-featured) or SQLite-vec (lightweight)
- ✅ **Cross-Platform**: Native support for macOS (Intel & Apple Silicon), Windows, and Linux
- ✅ **Docker Ready**: Complete containerization support with multiple deployment options
- ✅ **Hardware Optimized**: Automatic detection and optimization for available hardware (CUDA, MPS, DirectML)
- ✅ **Production Ready**: HTTP/SSE API, authentication, monitoring, and scaling features

### Recent Updates

- **v0.2.2+**: Enhanced multi-client support with automatic MCP application detection
- **SQLite-vec Backend**: Lightweight alternative to ChromaDB for resource-constrained systems
- **Homebrew Integration**: Native support for Homebrew-installed PyTorch on macOS
- **Docker Improvements**: Fixed boot loops, added multiple deployment configurations
- **HTTP/SSE API**: Real-time multi-client communication with Server-Sent Events

## Getting Help

### Quick Links

- **Installation Issues**: Check the [Installation Guide](installation/master-guide.md) and platform-specific guides
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

## Version History

- **Latest**: Enhanced documentation organization, consolidated guides, improved navigation
- **v0.2.2**: Multi-client improvements, SQLite-vec backend, Homebrew integration
- **v0.2.1**: Docker deployment fixes, HTTP/SSE API enhancements
- **v0.2.0**: Multi-client support, cross-platform compatibility improvements

---

## Navigation Tips

- **📁 Folders**: Click on folder names to explore sections
- **🔗 Links**: All internal links are relative and work offline
- **📱 Mobile**: Documentation is mobile-friendly for on-the-go reference
- **🔍 Search**: Use your browser's search (Ctrl/Cmd+F) to find specific topics

**Happy memory-ing! 🧠✨**
