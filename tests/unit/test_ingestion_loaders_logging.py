"""
Log-injection tests for the ingestion/ document loaders (#1146).

Each loader logs the path of the document it is asked to read when it starts
and again, with the parser's error text, when it fails; the semtools loader
also logs where the ``semtools`` binary was found and what it wrote to stderr.
A newline in any of those must not reach the log as a line break.

Nothing is written to disk and ``semtools`` is not run: ``validate_file`` and
the per-loader read step are stubbed, and the subprocess is a fake whose
``communicate`` returns a forged stderr.
"""

import asyncio
import logging
import shutil
from pathlib import Path

import pytest

from mcp_memory_service.ingestion import pdf_loader as pdf_module
from mcp_memory_service.ingestion.csv_loader import CSVLoader
from mcp_memory_service.ingestion.json_loader import JSONLoader
from mcp_memory_service.ingestion.pdf_loader import PDFLoader
from mcp_memory_service.ingestion.semtools_loader import SemtoolsLoader
from mcp_memory_service.ingestion.text_loader import TextLoader

FORGED = "FORGED ingestion verified"
LOGGERS = "mcp_memory_service.ingestion"


def _messages(caplog):
    return [record.getMessage() for record in caplog.records]


def _assert_clean(caplog, *expected):
    messages = _messages(caplog)
    for text in expected:
        assert any(text in m for m in messages), (text, messages)
    assert not any(f"\n{FORGED}" in m for m in messages), messages


async def _no_validation(_file_path):
    return None


async def _failing_read(*_args, **_kwargs):
    raise ValueError(f"parser said\n{FORGED}")


async def _drain(loader, file_path):
    with pytest.raises(ValueError):
        async for _chunk in loader.extract_chunks(file_path):
            pass


@pytest.fixture
def forged_path():
    return Path("docs", f"report\n{FORGED}.ext")


@pytest.mark.asyncio
async def test_json_loader_logs_do_not_carry_newlines(monkeypatch, caplog, forged_path):
    loader = JSONLoader()
    monkeypatch.setattr(loader, "validate_file", _no_validation)
    monkeypatch.setattr(loader, "_read_json_file", _failing_read)

    with caplog.at_level(logging.INFO, logger=LOGGERS):
        await _drain(loader, forged_path)

    _assert_clean(
        caplog,
        f"Extracting chunks from JSON file: docs/report\\n{FORGED}.ext",
        f"Error extracting from JSON file docs/report\\n{FORGED}.ext: ValueError - parser said\\n{FORGED}",
    )


@pytest.mark.asyncio
async def test_csv_loader_logs_do_not_carry_newlines(monkeypatch, caplog, forged_path):
    loader = CSVLoader()
    monkeypatch.setattr(loader, "validate_file", _no_validation)
    monkeypatch.setattr(loader, "_read_csv_file", _failing_read)

    with caplog.at_level(logging.INFO, logger=LOGGERS):
        await _drain(loader, forged_path)

    _assert_clean(
        caplog,
        f"Extracting chunks from CSV file: docs/report\\n{FORGED}.ext",
        f"Error extracting from CSV file docs/report\\n{FORGED}.ext: ValueError - parser said\\n{FORGED}",
    )


@pytest.mark.asyncio
async def test_text_loader_logs_do_not_carry_newlines(monkeypatch, caplog, forged_path):
    loader = TextLoader()
    monkeypatch.setattr(loader, "validate_file", _no_validation)
    monkeypatch.setattr(loader, "_read_file_content", _failing_read)

    with caplog.at_level(logging.INFO, logger=LOGGERS):
        await _drain(loader, forged_path)

    _assert_clean(
        caplog,
        f"Extracting chunks from text file: docs/report\\n{FORGED}.ext",
        f"Error extracting from text file docs/report\\n{FORGED}.ext: parser said\\n{FORGED}",
    )


@pytest.mark.asyncio
async def test_pdf_loader_logs_do_not_carry_newlines(monkeypatch, caplog, forged_path):
    # Pin the backend so the test does not depend on which PDF library is installed.
    monkeypatch.setattr(pdf_module, "HAS_PDFPLUMBER", True)
    loader = PDFLoader()

    async def _failing_pages(*_args, **_kwargs):
        raise ValueError(f"parser said\n{FORGED}")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(loader, "validate_file", _no_validation)
    monkeypatch.setattr(loader, "_extract_with_pdfplumber", _failing_pages)

    with caplog.at_level(logging.INFO, logger=LOGGERS):
        await _drain(loader, forged_path)

    _assert_clean(
        caplog,
        f"Extracting chunks from PDF: docs/report\\n{FORGED}.ext using pdfplumber",
        f"Error extracting from PDF docs/report\\n{FORGED}.ext: parser said\\n{FORGED}",
    )


class _FailingProcess:
    returncode = 1

    async def communicate(self):
        return b"", f"semtools said\n{FORGED}".encode()


@pytest.mark.asyncio
async def test_semtools_loader_logs_do_not_carry_newlines(monkeypatch, caplog, forged_path):
    monkeypatch.setattr(shutil, "which", lambda _name: f"/opt/bin/semtools\n{FORGED}")

    async def _fake_exec(*_args, **_kwargs):
        return _FailingProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    with caplog.at_level(logging.INFO, logger=LOGGERS):
        loader = SemtoolsLoader()
        monkeypatch.setattr(loader, "validate_file", _no_validation)
        await _drain(loader, forged_path)

    _assert_clean(
        caplog,
        f"Semtools found at: /opt/bin/semtools\\n{FORGED}",
        f"Extracting chunks from docs/report\\n{FORGED}.ext using semtools",
        f"Semtools parsing failed: semtools said\\n{FORGED}",
        f"Error running semtools: Semtools returned error: semtools said\\n{FORGED}",
        f"Error processing docs/report\\n{FORGED}.ext with semtools: Semtools returned error: semtools said\\n{FORGED}",
    )
