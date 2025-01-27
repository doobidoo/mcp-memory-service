# Generated by https://smithery.ai. See: https://smithery.ai/docs/config#dockerfile
# Start with an official Python image with a version >= 3.10
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY src/mcp_memory_service /app/src/mcp_memory_service

# Set environment variables if needed. Adjust paths to point to default container locations.
ENV MCP_MEMORY_CHROMA_PATH=/app/chroma_db \
    MCP_MEMORY_BACKUPS_PATH=/app/backups

# Create necessary directories for ChromaDB and backups
RUN mkdir -p /app/chroma_db /app/backups

# Specify the command to run the server
ENTRYPOINT ["python", "src/mcp_memory_service/server.py"]