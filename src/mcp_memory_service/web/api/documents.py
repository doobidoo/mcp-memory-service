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
import uuid
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...ingestion import get_loader_for_file, SUPPORTED_FORMATS
from ...models.memory import Memory
from ...utils.hashing import generate_content_hash
from ..dependencies import get_storage

logger = logging.getLogger(__name__)

router = APIRouter()


async def ensure_storage_initialized():
    """Ensure storage is initialized for web API usage."""
    try:
        # Try to get storage
        storage = get_storage()
        return storage
    except Exception as e:
        logger.error(f"Storage not available: {e}")
        raise HTTPException(status_code=503, detail="Storage backend not available")

# In-memory storage for upload tracking (in production, use database)
upload_sessions = {}

class UploadRequest(BaseModel):
    tags: List[str] = []
    chunk_size: int = 1000
    chunk_overlap: int = 200
    memory_type: str = "document"

class BatchUploadRequest(BaseModel):
    tags: List[str] = []
    chunk_size: int = 1000
    chunk_overlap: int = 200
    memory_type: str = "document"

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
    """
    logger.info(f"Document upload endpoint called with file: {file.filename}")
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower().lstrip('.')
        logger.info(f"File: {file.filename}, detected extension: '{file_ext}', supported: {list(SUPPORTED_FORMATS.keys())}")
        if file_ext not in SUPPORTED_FORMATS:
            supported = ", ".join(f".{ext}" for ext in SUPPORTED_FORMATS.keys())
            logger.error(f"Unsupported file type: {file_ext}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: .{file_ext}. Supported: {supported}"
            )

        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Create upload session
        upload_id = str(uuid.uuid4())
        temp_path = f"/tmp/{upload_id}_{file.filename}"

        # Save uploaded file temporarily
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Initialize upload session
        session = UploadStatus(
            upload_id=upload_id,
            status="queued",
            filename=file.filename,
            file_size=len(content),
            created_at=datetime.now()
        )
        upload_sessions[upload_id] = session

        # Start background processing
        background_tasks.add_task(
            process_document_upload,
            upload_id,
            temp_path,
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

        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]

        # Create batch upload session
        batch_id = str(uuid.uuid4())
        temp_paths = []

        # Validate and save all files
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in SUPPORTED_FORMATS:
                supported = ", ".join(f".{ext}" for ext in SUPPORTED_FORMATS.keys())
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type for {file.filename}: {file_ext}. Supported: {supported}"
                )

            temp_path = f"/tmp/{batch_id}_{file.filename}"
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
            temp_paths.append((file.filename, temp_path, len(content)))

        # Initialize batch session
        session = UploadStatus(
            upload_id=batch_id,
            status="queued",
            filename=f"Batch ({len(files)} files)",
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

async def process_document_upload(
    upload_id: str,
    file_path: str,
    tags: List[str],
    chunk_size: int,
    chunk_overlap: int,
    memory_type: str
):
    """Background task to process a single document upload."""
    try:
        logger.info(f"Starting document processing: {upload_id}")
        session = upload_sessions[upload_id]
        session.status = "processing"

        # Get storage
        storage = await ensure_storage_initialized()

        # Get appropriate loader
        file_path_obj = Path(file_path)
        loader = get_loader_for_file(file_path_obj)
        if loader is None:
            raise Exception(f"No loader available for {file_path_obj.suffix}")

        # Configure loader
        loader.chunk_size = chunk_size
        loader.chunk_overlap = chunk_overlap

        chunks_processed = 0
        chunks_stored = 0
        errors = []

        # Process chunks
        async for chunk in loader.extract_chunks(file_path_obj):
            chunks_processed += 1

            try:
                # Combine document tags with chunk metadata tags
                all_tags = tags.copy()
                if chunk.metadata.get('tags'):
                    all_tags.extend(chunk.metadata['tags'])

                # Create memory object
                memory = Memory(
                    content=chunk.content,
                    content_hash=generate_content_hash(chunk.content, chunk.metadata),
                    tags=list(set(all_tags)),  # Remove duplicates
                    memory_type=memory_type,
                    metadata=chunk.metadata
                )

                # Store the memory
                success, error = await storage.store(memory)
                if success:
                    chunks_stored += 1
                else:
                    errors.append(f"Chunk {chunk.chunk_index}: {error}")

                # Update progress
                progress = min(95.0, (chunks_processed / max(1, chunk.total_chunks)) * 100)
                session.chunks_processed = chunks_processed
                session.chunks_stored = chunks_stored
                session.total_chunks = chunk.total_chunks
                session.progress = progress
                session.errors = errors

            except Exception as e:
                errors.append(f"Chunk {chunk.chunk_index}: {str(e)}")

        # Finalize
        session.status = "completed" if chunks_stored > 0 else "failed"
        session.completed_at = datetime.now()
        session.progress = 100.0

        # Clean up temp file
        try:
            os.unlink(file_path)
        except:
            pass

        logger.info(f"Document processing completed: {upload_id}, {chunks_stored}/{chunks_processed} chunks")

    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        session = upload_sessions.get(upload_id)
        if session:
            session.status = "failed"
            session.errors.append(str(e))
            session.completed_at = datetime.now()
            await send_progress_update(upload_id, 0.0, f"Failed: {str(e)}")

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

                    try:
                        # Add file-specific tags
                        all_tags = tags.copy()
                        all_tags.append(f"source_file:{filename}")
                        all_tags.append(f"file_type:{file_path_obj.suffix.lstrip('.')}")

                        if chunk.metadata.get('tags'):
                            all_tags.extend(chunk.metadata['tags'])

                        # Create memory object
                        memory = Memory(
                            content=chunk.content,
                            content_hash=generate_content_hash(chunk.content, chunk.metadata),
                            tags=list(set(all_tags)),  # Remove duplicates
                            memory_type=memory_type,
                            metadata=chunk.metadata
                        )

                        # Store the memory
                        success, error = await storage.store(memory)
                        if success:
                            file_chunks_stored += 1
                            total_chunks_stored += 1
                        else:
                            all_errors.append(f"{filename} chunk {chunk.chunk_index}: {error}")

                    except Exception as e:
                        all_errors.append(f"{filename} chunk {chunk.chunk_index}: {str(e)}")

                processed_files += 1

                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

            except Exception as e:
                all_errors.append(f"{filename}: {str(e)}")
                processed_files += 1

                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

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
            await send_progress_update(batch_id, 0.0, f"Batch failed: {str(e)}")

# Clean up old completed sessions periodically
@router.on_event("startup")
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
                    if session.completed_at and (current_time - session.completed_at).seconds > 86400:
                        to_remove.append(upload_id)

            for upload_id in to_remove:
                del upload_sessions[upload_id]
                logger.debug(f"Cleaned up old upload session: {upload_id}")

    asyncio.create_task(cleanup())
