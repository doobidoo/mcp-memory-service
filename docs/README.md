# EchoVault Documentation

Welcome to the EchoVault documentation! This directory contains guides for installing, configuring, and using the EchoVault Memory Service.

## Getting Started

- [EchoVault Setup Guide](ECHOVAULT_SETUP.md): Comprehensive installation and configuration instructions
- [EchoVault Architecture](ECHOVAULT_ARCHITECTURE.md): Overview of the EchoVault architecture and how it works

## Original MCP Memory Service Guides

- [Installation Guide](guides/installation.md): Installation instructions for the standard MCP Memory Service
- [Troubleshooting Guide](guides/troubleshooting.md): Solutions for common issues
- [Migration Guide](guides/migration.md): Instructions for migrating between ChromaDB instances
- [Docker Guide](guides/docker.md): Docker deployment instructions
- [Invocation Guide](guides/invocation_guide.md): Keywords and phrases for memory operations

## Technical Documentation

- [Memory Migration](technical/memory-migration.md): Technical details about memory migration
- [Development Guidelines](technical/development.md): Development guidelines for contributing

## Overview of EchoVault Features

EchoVault extends the MCP Memory Service with:

1. **Durable Storage**: PostgreSQL with pgvector for reliable, persistent storage
2. **High-Performance Search**: Qdrant for fast approximate nearest neighbor (ANN) search
3. **Large Content Support**: Cloudflare R2 for efficient blob storage
4. **Observability**: OpenTelemetry and Prometheus integration
5. **Enterprise Security**: JWT authentication support
6. **Memory Lifecycle Management**: Automatic summarization of old memories

## Compatibility

EchoVault is fully compatible with the standard MCP Memory Service and integrates seamlessly with Claude Desktop. It can be enabled with a simple configuration change, and all existing memories will continue to work.

## Cloud Service Requirements

EchoVault leverages the following managed services:

- **Neon PostgreSQL**: Serverless PostgreSQL with pgvector extension
- **Qdrant Cloud**: Vector database optimized for semantic search
- **Cloudflare R2**: S3-compatible object storage

All of these services offer free tiers that are sufficient for personal use.

## Monitoring and Observability

EchoVault includes comprehensive monitoring and observability features:

- **Prometheus Metrics**: Performance and health metrics
- **OpenTelemetry Tracing**: Distributed tracing across all components
- **Grafana Dashboards**: Pre-configured dashboards for monitoring

## Docker Deployment

EchoVault can be deployed using Docker and Docker Compose. The docker-compose.yml file includes:

- The EchoVault Memory Service
- Local PostgreSQL with pgvector (for development)
- Qdrant vector database
- MinIO S3-compatible storage (for development)
- Jaeger for distributed tracing
- Prometheus for metrics collection
- Grafana for visualization

## Memory Summarization

EchoVault includes a memory summarization feature that automatically summarizes old memories to reduce database size while preserving semantic information. This process is configurable and can be scheduled to run regularly.

## Need Help?

If you have any questions or need assistance, please:

- Check the [EchoVault Setup Guide](ECHOVAULT_SETUP.md)
- Join our community support channel
- Open an issue on the GitHub repository