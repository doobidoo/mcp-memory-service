# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Document Upload API Endpoints

Provides REST API endpoints for document ingestion through the web dashboard.
Supports single file upload, batch upload, progress tracking, and upload history.
"""

import os
import re
import uuid
import asyncio
import logging
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, unquote

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...ingestion import get_loader_for_file, SUPPORTED_FORMATS
from ...models.memory import Memory
from ...utils import create_memory_from_chunk, _process_and_store_chunk, generate_content_hash
from ..dependencies import get_storage

logger = logging.getLogger(__name__)

router = APIRouter()

# Constants
MAX_TAG_LENGTH = 100


def parse_and_validate_tags(tags: str) -> List[str]:
    """
    Parse and validate tags from user input.

    Handles comma-separated and space-separated tags, removes file:// prefixes,
    sanitizes path separators, and validates tag lengths.

    Args:
        tags: Raw tag string (comma or space separated)

    Returns:
        List of cleaned and validated tags

    Raises:
        HTTPException: If any tag exceeds MAX_TAG_LENGTH
    """
    if not tags or not tags.strip():
        return []

    # Split by comma OR space
    raw_tags = tags.replace(',', ' ').split()
    tag_list = []

    for tag in raw_tags:
        clean_tag = tag.strip()
        if not clean_tag:
            continue

        # Remove file:// protocol prefixes and extract filename
        # Uses urllib.parse for robust handling of URL-encoded chars and different path formats
        if clean_tag.startswith('file://'):
            path_str = unquote(urlparse(clean_tag).path)
            # On Windows, urlparse may add a leading slash (e.g., /C:/...), which needs to be removed
            if os.name == 'nt' and path_str.startswith('/') and len(path_str) > 2 and path_str[2] == ':':
                path_str = path_str[1:]
            clean_tag = Path(path_str).name

        # Remove common path separators to create clean tag names
        clean_tag = re.sub(r'[/\\]', '_', clean_tag)

        # Validate tag length - raise error instead of silently dropping
        if len(clean_tag) > MAX_TAG_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Tag '{clean_tag[:100]}...' exceeds maximum length of {MAX_TAG_LENGTH} characters. "
                       f"Please use shorter, more descriptive tags."
            )

        tag_list.append(clean_tag)

    return tag_list


async def ensure_storage_initialized():
    """Ensure storage is initialized for web API usage."""
    logger.info("🔍 Checking storage availability...")
    try:
        # Try to get storage
        storage = get_storage()
        logger.info("✅ Storage already available")
        return storage
    except Exception as e:
        logger.warning(f"⚠️ Storage not available ({e}), attempting to initialize...")
        try:
            # Import and initialize storage
            from ..dependencies import create_storage_backend, set_storage
            logger.info("🏗️ Creating storage backend...")
            storage = await create_storage_backend()
            set_storage(storage)
            logger.info("✅ Storage initialized successfully in API context")
            return storage
        except Exception as init_error:
            logger.error(f"❌ Failed to initialize storage: {init_error}")
            logger.error(f"Full error: {str(init_error)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't raise HTTPException here since this is called from background tasks
            raise init_error

# In-memory storage for upload tracking (in production, use database)
upload_sessions = {}

# Note: UploadRequest and BatchUploadRequest models removed - not used
# Endpoints read parameters directly from form data

class UploadStatus(BaseModel):
    upload_id: str
    status: str  # queued, processing, completed, failed
    filename: str = ""
    file_size: int = 0
    chunks_processed: int = 0
    chunks_stored: int = 0
    total_chunks: int = 0
    progress: float = 0.0
    errors: List[str] = []
    created_at: datetime
    completed_at: Optional[datetime] = None

@router.post("/upload", response_model=Dict[str, Any])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tags: str = Form(""),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    memory_type: str = Form("document")
):
    """
    Upload and ingest a single document.

    Args:
        file: The document file to upload
        tags: Comma-separated list of tags
        chunk_size: Target chunk size in characters
        chunk_overlap: Chunk overlap in characters
        memory_type: Type label for memories

    Returns:
        Upload session information with ID for tracking

    Uses FastAPI BackgroundTasks for proper async processing.
    """
    logger.info(f"🚀 Document upload endpoint called with file: {file.filename}")
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        logger.info(f"File content length: {file_size} bytes")

        # Validate file type
        file_ext = Path(file.filename).suffix.lower().lstrip('.')
        if file_ext not in SUPPORTED_FORMATS:
            supported = ", ".join(f".{ext}" for ext in SUPPORTED_FORMATS.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{file_ext}. Supported: {supported}"
            )

        # Parse and validate tags
        tag_list = parse_and_validate_tags(tags)

        # Create upload session
        upload_id = str(uuid.uuid4())

        # Create secure temporary file (avoids path traversal vulnerability)
        # Extract safe file extension for suffix
        file_ext = Path(file.filename).suffix if file.filename else ""
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            prefix=f"{upload_id}_",
            suffix=file_ext
        )
        temp_path = temp_file.name

        # Save uploaded file temporarily
        with temp_file:
            temp_file.write(file_content)

        # Initialize upload session
        session = UploadStatus(
            upload_id=upload_id,
            status="queued",
            filename=file.filename,
            file_size=file_size,
            created_at=datetime.now()
        )
        upload_sessions[upload_id] = session

        # Start background processing
        background_tasks.add_task(
            process_single_file_upload,
            upload_id,
            temp_path,
            file.filename,
            tag_list,
            chunk_size,
            chunk_overlap,
            memory_type
        )

        return {
            "upload_id": upload_id,
            "status": "queued",
            "message": f"Document {file.filename} queued for processing"
        }

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-upload", response_model=Dict[str, Any])
async def batch_upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    tags: str = Form(""),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    memory_type: str = Form("document")
):
    """
    Upload and ingest multiple documents in batch.

    Args:
        files: List of document files to upload
        tags: Comma-separated list of tags
        chunk_size: Target chunk size in characters
        chunk_overlap: Chunk overlap in characters
        memory_type: Type label for memories

    Returns:
        Batch upload session information
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        # Parse and validate tags
        tag_list = parse_and_validate_tags(tags)

        # Create batch upload session
        batch_id = str(uuid.uuid4())
        temp_paths = []

        # Validate and save all files
        for file in files:
            file_ext = Path(file.filename).suffix.lower().lstrip('.')
            if file_ext not in SUPPORTED_FORMATS:
                supported = ", ".join(f".{ext}" for ext in SUPPORTED_FORMATS.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type for {file.filename}: {file_ext}. Supported: {supported}"
                )

            # Create secure temporary file (avoids path traversal vulnerability)
            content = await file.read()
            safe_ext = Path(file.filename).suffix.lower() if file.filename else ""
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                prefix=f"{batch_id}_",
                suffix=safe_ext
            )
            temp_path = temp_file.name
            with temp_file:
                temp_file.write(content)
            temp_paths.append((file.filename, temp_path, len(content)))

        # Calculate total file size for the batch
        total_file_size = sum(file_size for _, _, file_size in temp_paths)

        # Initialize batch session
        session = UploadStatus(
            upload_id=batch_id,
            status="queued",
            filename=f"Batch ({len(files)} files)",
            file_size=total_file_size,
            created_at=datetime.now()
        )
        upload_sessions[batch_id] = session

        # Start background processing
        background_tasks.add_task(
            process_batch_upload,
            batch_id,
            temp_paths,
            tag_list,
            chunk_size,
            chunk_overlap,
            memory_type
        )

        return {
            "upload_id": batch_id,
            "status": "queued",
            "message": f"Batch of {len(files)} documents queued for processing"
        }

    except Exception as e:
        logger.error(f"Batch upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{upload_id}", response_model=UploadStatus)
async def get_upload_status(upload_id: str):
    """
    Get the status of an upload session.

    Args:
        upload_id: The upload session ID

    Returns:
        Current upload status
    """
    if upload_id not in upload_sessions:
        raise HTTPException(status_code=404, detail="Upload session not found")

    return upload_sessions[upload_id]

@router.get("/history", response_model=Dict[str, List[Dict[str, Any]]])
async def get_upload_history():
    """
    Get the history of all uploads.

    Returns:
        List of completed uploads with metadata
    """
    logger.info("Documents history endpoint called")
    try:
        # For now, return empty history since storage might not be initialized
        # In production, this would query a database
        history = []
        for session in upload_sessions.values():
            if session.status in ["completed", "failed"]:
                history.append({
                    "upload_id": session.upload_id,
                    "filename": session.filename,
                    "file_size": session.file_size,
                    "status": session.status,
                    "chunks_processed": session.chunks_processed,
                    "chunks_stored": session.chunks_stored,
                    "progress": session.progress,
                    "errors": session.errors,
                    "created_at": session.created_at.isoformat(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None
                })

        # Sort by creation time, most recent first
        history.sort(key=lambda x: x["created_at"], reverse=True)

        return {"uploads": history}
    except Exception as e:
        logger.error(f"Error in get_upload_history: {e}")
        # Return empty history on error so the UI doesn't break
        return {"uploads": []}

async def process_single_file_upload(
    upload_id: str,
    file_path: str,
    filename: str,
    tags: List[str],
    chunk_size: int,
    chunk_overlap: int,
    memory_type: str
):
    """Background task to process a single document upload."""
    try:
        logger.info(f"Starting document processing: {upload_id} - {filename}")
        session = upload_sessions[upload_id]
        session.status = "processing"

        # Get storage
        storage = await ensure_storage_initialized()

        # Get appropriate loader
        file_path_obj = Path(file_path)
        loader = get_loader_for_file(file_path_obj)
        if loader is None:
            raise ValueError(f"No loader available for file: {filename}")

        # Configure loader
        loader.chunk_size = chunk_size
        loader.chunk_overlap = chunk_overlap

        chunks_processed = 0
        chunks_stored = 0

        # Process chunks from the file
        async for chunk in loader.extract_chunks(file_path_obj):
            chunks_processed += 1

            try:
                # Add file-specific tags
                all_tags = tags.copy()
                all_tags.append(f"source_file:{filename}")
                all_tags.append(f"file_type:{file_path_obj.suffix.lstrip('.')}")
                all_tags.append(f"upload_id:{upload_id}")

                if chunk.metadata.get('tags'):
                    # Handle tags from chunk metadata (can be string or list)
                    chunk_tags = chunk.metadata['tags']
                    if isinstance(chunk_tags, str):
                        # Split comma-separated string into list
                        chunk_tags = [tag.strip() for tag in chunk_tags.split(',') if tag.strip()]
                    all_tags.extend(chunk_tags)

                # Add upload_id to metadata
                chunk_metadata = chunk.metadata.copy() if chunk.metadata else {}
                chunk_metadata['upload_id'] = upload_id
                chunk_metadata['source_file'] = filename

                # Create memory object
                memory = Memory(
                    content=chunk.content,
                    content_hash=generate_content_hash(chunk.content, chunk_metadata),
                    tags=list(set(all_tags)),  # Remove duplicates
                    memory_type=memory_type,
                    metadata=chunk_metadata
                )

                # Store the memory
                success, error = await storage.store(memory)
                if success:
                    chunks_stored += 1
                else:
                    session.errors.append(f"Chunk {chunk.chunk_index}: {error}")

            except Exception as e:
                session.errors.append(f"Chunk {chunk.chunk_index}: {str(e)}")

            # Update progress
            session.chunks_processed = chunks_processed
            session.chunks_stored = chunks_stored
            session.progress = (chunks_processed / max(chunks_processed, 1)) * 100

        # Mark as completed
        session.status = "completed"
        session.completed_at = datetime.now()
        session.progress = 100.0

        logger.info(f"Document processing completed: {upload_id}, {chunks_stored}/{chunks_processed} chunks")
        return {"chunks_processed": chunks_processed, "chunks_stored": chunks_stored}

    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        session = upload_sessions.get(upload_id)
        if session:
            session.status = "failed"
            session.errors.append(str(e))
            session.completed_at = datetime.now()
    finally:
        # Clean up temp file (always executed)
        try:
            os.unlink(file_path)
        except Exception as cleanup_error:
            logger.debug(f"Could not delete temp file {file_path}: {cleanup_error}")


async def process_batch_upload(
    batch_id: str,
    file_info: List[tuple],  # (filename, temp_path, size)
    tags: List[str],
    chunk_size: int,
    chunk_overlap: int,
    memory_type: str
):
    """Background task to process a batch document upload."""
    try:
        logger.info(f"Starting batch processing: {batch_id}")
        session = upload_sessions[batch_id]
        session.status = "processing"

        # Get storage
        storage = await ensure_storage_initialized()

        total_files = len(file_info)
        processed_files = 0
        total_chunks_processed = 0
        total_chunks_stored = 0
        all_errors = []

        for filename, temp_path, file_size in file_info:
            try:
                # Get appropriate loader
                file_path_obj = Path(temp_path)
                loader = get_loader_for_file(file_path_obj)
                if loader is None:
                    all_errors.append(f"{filename}: No loader available")
                    processed_files += 1
                    continue

                # Configure loader
                loader.chunk_size = chunk_size
                loader.chunk_overlap = chunk_overlap

                file_chunks_processed = 0
                file_chunks_stored = 0

                # Process chunks from this file
                async for chunk in loader.extract_chunks(file_path_obj):
                     file_chunks_processed += 1
                     total_chunks_processed += 1

                     # Process and store the chunk
                     success, error = await _process_and_store_chunk(
                         chunk,
                         storage,
                         filename,
                         base_tags=tags.copy(),
                         memory_type=memory_type,
                         context_tags={
                             "source_file": filename,
                             "file_type": file_path_obj.suffix.lstrip('.'),
                             "upload_id": batch_id
                         },
                         extra_metadata={
                             "upload_id": batch_id,
                             "source_file": filename
                         }
                     )

                     if success:
                         file_chunks_stored += 1
                         total_chunks_stored += 1
                     else:
                         all_errors.append(error)

                processed_files += 1

            except Exception as e:
                all_errors.append(f"{filename}: {str(e)}")
                processed_files += 1

            finally:
                # Clean up temp file (always executed)
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    logger.debug(f"Could not delete temp file {temp_path}: {cleanup_error}")

        # Finalize batch
        session.status = "completed" if total_chunks_stored > 0 else "failed"
        session.completed_at = datetime.now()
        session.chunks_processed = total_chunks_processed
        session.chunks_stored = total_chunks_stored
        session.progress = 100.0
        session.errors = all_errors

        logger.info(f"Batch processing completed: {batch_id}, {total_chunks_stored}/{total_chunks_processed} chunks")

    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        session = upload_sessions.get(batch_id)
        if session:
            session.status = "failed"
            session.errors.append(str(e))
            session.completed_at = datetime.now()
            # Note: send_progress_update removed - progress tracking via polling instead

# Clean up old completed sessions periodically
@router.on_event("startup")  # TODO: Migrate to lifespan context manager in app.py (FastAPI 0.109+)
async def cleanup_old_sessions():
    """Clean up old completed upload sessions."""
    async def cleanup():
        while True:
            await asyncio.sleep(3600)  # Clean up every hour
            current_time = datetime.now()
            to_remove = []

            for upload_id, session in upload_sessions.items():
                if session.status in ["completed", "failed"]:
                    # Keep sessions for 24 hours after completion
                    if session.completed_at and (current_time - session.completed_at).total_seconds() > 86400:
                        to_remove.append(upload_id)

            for upload_id in to_remove:
                del upload_sessions[upload_id]
                logger.debug(f"Cleaned up old upload session: {upload_id}")

    asyncio.create_task(cleanup())

@router.delete("/remove/{upload_id}")
async def remove_document(upload_id: str, remove_from_memory: bool = True):
    """
    Remove a document and optionally its memories.

    Args:
        upload_id: The upload session ID
        remove_from_memory: Whether to delete associated memories (default: True)

    Returns:
        Removal status with count of memories deleted
    """
    logger.info(f"Remove document request for upload_id: {upload_id}, remove_from_memory: {remove_from_memory}")

    # Get session info if available (may not exist after server restart)
    session = upload_sessions.get(upload_id)
    filename = session.filename if session else "Unknown file"
    memories_deleted = 0

    try:
        if remove_from_memory:
            # Get storage
            storage = get_storage()

            # Search by tag pattern: upload_id:{upload_id}
            upload_tag = f"upload_id:{upload_id}"
            logger.info(f"Searching for memories with tag: {upload_tag}")

            try:
                # Delete all memories with this upload_id tag
                count, _ = await storage.delete_by_tags([upload_tag])
                memories_deleted = count
                logger.info(f"Deleted {memories_deleted} memories with tag {upload_tag}")

                # If we deleted memories but don't have session info, try to get filename from first memory
                if memories_deleted > 0 and not session:
                    # Try to get source_file from metadata by checking remaining memories
                    # (we already deleted them, so we'll use a generic message)
                    filename = f"Document (upload_id: {upload_id[:8]}...)"

            except Exception as e:
                logger.warning(f"Could not delete memories by tag: {e}")
                # If deletion fails and we don't know about this upload, return 404
                if not session:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Upload ID not found and no memories with tag '{upload_tag}'"
                    )
                memories_deleted = 0

        # Remove upload session if it exists
        if session:
            del upload_sessions[upload_id]

        return {
            "status": "success",
            "upload_id": upload_id,
            "filename": filename,
            "memories_deleted": memories_deleted,
            "message": f"Document '{filename}' removed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing document: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to remove document: {str(e)}")

@router.delete("/remove-by-tags")
async def remove_documents_by_tags(tags: List[str]):
    """
    Remove documents by their tags.

    Args:
        tags: List of tags to search for

    Returns:
        Removal status with affected upload IDs and memory counts
    """
    logger.info(f"Remove documents by tags request: {tags}")

    try:
        # Get storage
        storage = get_storage()

        # Delete memories by tags
        result = await storage.delete_by_tags(tags)
        memories_deleted = result.get('deleted_count', 0) if isinstance(result, dict) else 0

        # Find and remove affected upload sessions
        affected_sessions = []
        to_remove = []

        for upload_id, session in upload_sessions.items():
            # Check if any of the document's tags match
            # This requires storing tags in the session object
            # For now, just track all sessions (placeholder)
            pass

        return {
            "status": "success",
            "tags": tags,
            "memories_deleted": memories_deleted,
            "affected_uploads": affected_sessions,
            "message": f"Deleted {memories_deleted} memories matching tags"
        }

    except Exception as e:
        logger.error(f"Error removing documents by tags: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to remove documents: {str(e)}")

@router.get("/search-content/{upload_id}")
async def search_document_content(upload_id: str, limit: int = 1000):
    """
    Search for all memories associated with an upload.

    Args:
        upload_id: The upload session ID
        limit: Maximum number of results to return (default: 1000)

    Returns:
        List of memories with their content and metadata
    """
    logger.info(f"Search document content for upload_id: {upload_id}, limit: {limit}")

    # Get session info if available (may not exist after server restart)
    session = upload_sessions.get(upload_id)

    # If no session, we'll still try to find memories by upload_id tag
    if not session:
        logger.info(f"No upload session found for {upload_id}, searching by tag only")

    try:
        # Get storage
        storage = get_storage()

        # Search for memories with upload_id tag
        upload_tag = f"upload_id:{upload_id}"
        logger.info(f"Searching for memories with tag: {upload_tag}")

        # Use tag search (search_by_tags doesn't support limit parameter)
        all_memories = await storage.search_by_tags([upload_tag])

        # If no memories found and no session, this upload_id doesn't exist
        if not all_memories and not session:
            raise HTTPException(status_code=404, detail=f"No memories found for upload_id: {upload_id}")

        # Apply limit after retrieval
        memories = all_memories[:limit] if limit and limit > 0 else all_memories

        # Format results
        results = []
        for memory in memories:
            # Handle created_at (stored as float timestamp)
            created_at_str = None
            if memory.created_at:
                if isinstance(memory.created_at, float):
                    created_at_str = datetime.fromtimestamp(memory.created_at).isoformat()
                elif hasattr(memory.created_at, 'isoformat'):
                    created_at_str = memory.created_at.isoformat()

            results.append({
                "content_hash": memory.content_hash,
                "content": memory.content,
                "tags": memory.tags,
                "metadata": memory.metadata,
                "created_at": created_at_str,
                "chunk_index": memory.metadata.get('chunk_index', 0) if memory.metadata else 0,
                "page": memory.metadata.get('page', None) if memory.metadata else None
            })

        # Sort by chunk index
        results.sort(key=lambda x: x.get('chunk_index', 0))

        # Get filename from session or from first memory's metadata
        filename = session.filename if session else None
        if not filename and results:
            # Try to get from first memory's metadata
            first_memory_metadata = results[0].get('metadata', {})
            filename = first_memory_metadata.get('source_file', f"Document (upload_id: {upload_id[:8]}...)")

        return {
            "status": "success",
            "upload_id": upload_id,
            "filename": filename or "Unknown Document",
            "total_found": len(results),
            "memories": results
        }

    except Exception as e:
        logger.error(f"Error searching document content: {str(e)}")
        # Get filename from session if available
        filename = session.filename if session else f"Document (upload_id: {upload_id[:8]}...)"
        # Return empty results instead of error to avoid breaking UI
        return {
            "status": "partial",
            "upload_id": upload_id,
            "filename": filename,
            "total_found": 0,
            "memories": [],
            "error": str(e)
        }
