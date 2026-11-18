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

"""Regression coverage for document upload validation error propagation and temp file cleanup (Issue #2041)."""

import io
import os
from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException, UploadFile

from mcp_memory_service.web.api import documents


def _make_upload_file(filename: str, content: bytes = b"sample text") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


@pytest.mark.asyncio
async def test_upload_document_unsupported_file_type_raises_400():
    """Verify single file upload with unsupported extension raises 400 Bad Request, not 500."""
    bad_file = _make_upload_file("program.exe")
    bg_tasks = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await documents.upload_document(
            background_tasks=bg_tasks,
            file=bad_file,
            tags="test",
            user=None,
        )

    assert exc_info.value.status_code == 400
    assert "Unsupported file type: .exe" in exc_info.value.detail
    bg_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_upload_document_tag_too_long_raises_400():
    """Verify tag exceeding MAX_TAG_LENGTH raises 400 Bad Request, not 500."""
    good_file = _make_upload_file("notes.txt")
    long_tag = "x" * 105
    bg_tasks = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await documents.upload_document(
            background_tasks=bg_tasks,
            file=good_file,
            tags=long_tag,
            user=None,
        )

    assert exc_info.value.status_code == 400
    assert "exceeds maximum length" in exc_info.value.detail
    bg_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_upload_document_cleans_up_temp_file_on_error(monkeypatch):
    """Verify temp file is unlinked if an unexpected exception occurs before background queueing."""
    good_file = _make_upload_file("doc.txt", b"hello world")
    bg_tasks = MagicMock()

    created_temp_files = []
    real_named_temp = documents.tempfile.NamedTemporaryFile

    def tracking_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf.name)
        return tf

    monkeypatch.setattr(documents.tempfile, "NamedTemporaryFile", tracking_temp_file)

    def boom(*args, **kwargs):
        raise RuntimeError("queue failed")

    monkeypatch.setattr(documents.UploadStatus, "__init__", boom)

    with pytest.raises(HTTPException) as exc_info:
        await documents.upload_document(
            background_tasks=bg_tasks,
            file=good_file,
            tags="valid",
            user=None,
        )

    assert exc_info.value.status_code == 500
    assert len(created_temp_files) == 1
    assert not os.path.exists(created_temp_files[0]), f"Temp file {created_temp_files[0]} was leaked on disk!"


@pytest.mark.asyncio
async def test_batch_upload_empty_files_raises_400():
    """Verify empty file list raises 400 Bad Request, not 500."""
    bg_tasks = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await documents.batch_upload_documents(
            background_tasks=bg_tasks,
            files=[],
            tags="test",
            user=None,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No files provided"
    bg_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_batch_upload_tag_too_long_raises_400():
    """Verify tag exceeding MAX_TAG_LENGTH in batch upload raises 400 Bad Request, not 500."""
    file1 = _make_upload_file("doc1.txt")
    long_tag = "y" * 105
    bg_tasks = MagicMock()

    with pytest.raises(HTTPException) as exc_info:
        await documents.batch_upload_documents(
            background_tasks=bg_tasks,
            files=[file1],
            tags=long_tag,
            user=None,
        )

    assert exc_info.value.status_code == 400
    assert "exceeds maximum length" in exc_info.value.detail
    bg_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_batch_upload_unsupported_file_cleans_up_partial_temp_files(monkeypatch):
    """Verify batch upload unlinks previously written temporary files when a subsequent file fails validation."""
    valid_file = _make_upload_file("notes.txt", b"valid document content")
    invalid_file = _make_upload_file("script.sh", b"echo 123")
    bg_tasks = MagicMock()

    created_temp_files = []
    real_named_temp = documents.tempfile.NamedTemporaryFile

    def tracking_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf.name)
        return tf

    monkeypatch.setattr(documents.tempfile, "NamedTemporaryFile", tracking_temp_file)

    with pytest.raises(HTTPException) as exc_info:
        await documents.batch_upload_documents(
            background_tasks=bg_tasks,
            files=[valid_file, invalid_file],
            tags="batch_test",
            user=None,
        )

    # Must return 400, not 500
    assert exc_info.value.status_code == 400
    assert "Unsupported file type for script.sh: sh" in exc_info.value.detail
    bg_tasks.add_task.assert_not_called()

    # The temporary file created for valid_file must have been cleaned up and deleted from disk
    assert len(created_temp_files) == 1
    leaked_temp_path = created_temp_files[0]
    assert not os.path.exists(leaked_temp_path), f"Temp file {leaked_temp_path} was leaked on disk!"
