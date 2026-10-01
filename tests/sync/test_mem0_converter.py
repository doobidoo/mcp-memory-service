import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mcp_memory_service.models.tag_taxonomy import validate_tag
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.sync.converters.mem0 import (
    Mem0ExportError,
    convert_mem0_export,
)
from mcp_memory_service.sync.importer import MemoryImporter
from mcp_memory_service.utils.hashing import generate_content_hash

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mem0" / "oss_qdrant_export.json"


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, ensure_ascii=False)


def test_converts_real_mem0_export_to_importer_shape(tmp_path):
    output_path = tmp_path / "converted.json"

    result = convert_mem0_export(FIXTURE, output_path)

    assert result["converted"] == 2
    assert result["skipped"] == 0
    assert result["warnings"] == 1
    assert output_path.exists()


def test_maps_real_mem0_fields_and_preserves_timestamps(tmp_path):
    output_path = tmp_path / "converted.json"

    convert_mem0_export(FIXTURE, output_path)
    converted = _read_json(output_path)
    first = converted["memories"][0]

    assert converted["export_metadata"]["source_machine"] == "mem0"
    assert converted["export_metadata"]["total_memories"] == 2
    assert converted["export_metadata"]["source_record_count"] == 2
    assert first["content"] == "User likes dark mode"
    assert first["content_hash"] != generate_content_hash("User likes dark mode")
    assert first["created_at"] == datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
    assert first["updated_at"] == first["created_at"]
    assert first["tags"] == [
        "user:alice",
        "agent:agent-1",
        "sys:mem0-run:run-1",
    ]
    assert all(validate_tag(tag) for tag in first["tags"])
    assert first["metadata"]["agent_id"] == "agent-1"
    assert first["metadata"]["mem0_user_id"] == "alice"
    assert first["metadata"]["mem0_agent_id"] == "agent-1"
    assert first["metadata"]["mem0_run_id"] == "run-1"
    assert first["metadata"]["mem0_record_id"] == "point-1"
    assert first["metadata"]["topic"] == "preferences"
    assert (
        first["metadata"]["mem0_source_payload"]["text_lemmatized"]
        == "user like dark mode"
    )


def test_accepts_platform_get_all_and_export_array_shapes(tmp_path):
    created_at = "2026-05-01T00:00:00Z"
    cases = [
        {
            "count": 1,
            "results": [{"id": "m1", "memory": "one", "created_at": created_at}],
        },
        {
            "memories": [
                {
                    "id": "m2",
                    "content": "two",
                    "created_at": created_at,
                }
            ]
        },
        [
            {
                "id": "m3",
                "memory": "three",
                "created_at": created_at,
            }
        ],
    ]

    for index, payload in enumerate(cases):
        input_path = tmp_path / f"input-{index}.json"
        output_path = tmp_path / f"output-{index}.json"
        _write_json(input_path, payload)

        result = convert_mem0_export(input_path, output_path)

        assert result["converted"] == 1
        memory = _read_json(output_path)["memories"][0]
        assert memory["content"] in {"one", "two", "three"}
        assert memory["content_hash"] == generate_content_hash(memory["content"])


def test_preserves_fields_that_exist_only_in_source_payload(tmp_path):
    input_path = tmp_path / "source-payload.json"
    output_path = tmp_path / "source-payload-converted.json"
    _write_json(
        input_path,
        [
            {
                "id": "m1",
                "memory": "Keep payload fields",
                "created_at": "2026-05-01T00:00:00Z",
                "source_payload": {
                    "data": "Keep payload fields",
                    "text_lemmatized": "keep payload field",
                },
            }
        ],
    )

    convert_mem0_export(input_path, output_path)

    metadata = _read_json(output_path)["memories"][0]["metadata"]
    assert metadata["mem0_source_payload"]["text_lemmatized"] == "keep payload field"


def test_distinct_identities_receive_distinct_dedupe_keys(tmp_path):
    input_path = tmp_path / "duplicates.json"
    output_path = tmp_path / "duplicates-converted.json"
    _write_json(
        input_path,
        [
            {
                "id": "m1",
                "memory": "Shared preference",
                "created_at": "2026-05-01T00:00:00Z",
                "user_id": "alice",
                "agent_id": "agent-a",
                "run_id": "run-a",
            },
            {
                "id": "m2",
                "memory": "Shared preference",
                "created_at": "2026-05-02T00:00:00Z",
                "user_id": "bob",
                "agent_id": "agent-b",
                "run_id": "run-b",
            },
        ],
    )

    result = convert_mem0_export(input_path, output_path)

    assert result["converted"] == 2
    memories = _read_json(output_path)["memories"]
    assert len(memories) == 2
    assert memories[0]["content"] == memories[1]["content"]
    assert memories[0]["content_hash"] != memories[1]["content_hash"]
    assert set(memories[0]["tags"]) == {
        "user:alice",
        "agent:agent-a",
        "sys:mem0-run:run-a",
    }
    assert set(memories[1]["tags"]) == {
        "user:bob",
        "agent:agent-b",
        "sys:mem0-run:run-b",
    }


def test_uses_conversion_time_and_records_auditable_warning(tmp_path):
    input_path = tmp_path / "missing-time.json"
    output_path = tmp_path / "missing-time-converted.json"
    _write_json(input_path, [{"id": "m1", "memory": "No source timestamp"}])

    before = datetime.now(timezone.utc).timestamp()
    result = convert_mem0_export(input_path, output_path)
    after = datetime.now(timezone.utc).timestamp()

    converted = _read_json(output_path)
    memory = converted["memories"][0]
    assert result["warnings"] == 1
    assert before <= memory["created_at"] <= after
    assert memory["updated_at"] == memory["created_at"]
    assert memory["metadata"]["mem0_timestamp_source"] == "conversion_time"
    warning = converted["export_metadata"]["conversion_warnings"][0]
    assert warning["record_index"] == 0
    assert warning["code"] == "missing_created_at"
    assert warning["action"] == "used_conversion_time"


def test_rejects_unsupported_export_shape(tmp_path):
    input_path = tmp_path / "invalid.json"
    output_path = tmp_path / "converted.json"
    _write_json(input_path, {"unexpected": []})

    with pytest.raises(Mem0ExportError):
        convert_mem0_export(input_path, output_path)


def test_skips_invalid_records_and_keeps_valid_records(tmp_path):
    input_path = tmp_path / "partial.json"
    output_path = tmp_path / "partial-converted.json"
    created_at = "2026-05-01T00:00:00Z"
    _write_json(
        input_path,
        [
            {"id": "valid-1", "memory": "first", "created_at": created_at},
            {"id": "missing-content", "created_at": created_at},
            {"id": "bad-time", "memory": "bad", "created_at": "not-a-time"},
            {"id": "valid-2", "memory": "second", "created_at": created_at},
        ],
    )

    result = convert_mem0_export(input_path, output_path)

    assert result["converted"] == 2
    assert result["skipped"] == 2
    assert result["warnings"] == 2
    converted = _read_json(output_path)
    assert [memory["content"] for memory in converted["memories"]] == [
        "first",
        "second",
    ]
    assert converted["export_metadata"]["source_record_count"] == 4
    warnings = converted["export_metadata"]["conversion_warnings"]
    assert [warning["record_index"] for warning in warnings] == [1, 2]
    assert all(warning["action"] == "skipped" for warning in warnings)


def test_expands_home_and_accepts_custom_string_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    source_path = tmp_path / "source.json"
    _write_json(source_path, [{"id": "m1", "memory": "home path"}])

    result = convert_mem0_export("~/source.json", "~/custom/nested/output.json")

    assert result["converted"] == 1
    assert (tmp_path / "custom" / "nested" / "output.json").is_file()


def test_rejects_directory_paths(tmp_path):
    source_path = tmp_path / "source.json"
    _write_json(source_path, [{"id": "m1", "memory": "path check"}])

    with pytest.raises(Mem0ExportError, match="not a file"):
        convert_mem0_export(tmp_path, tmp_path / "output.json")

    with pytest.raises(Mem0ExportError, match="directory"):
        convert_mem0_export(source_path, tmp_path)


def test_handles_large_export_boundary(tmp_path):
    source = _read_json(FIXTURE)
    records = []
    for index in range(5000):
        record = copy.deepcopy(source["records"][0])
        record["id"] = f"large-{index:05d}"
        record["memory"] = f"Large export memory {index:05d}"
        record["created_at"] = "2026-05-01T00:00:00Z"
        record["updated_at"] = "2026-05-01T00:00:00Z"
        records.append(record)

    input_path = tmp_path / "large.json"
    output_path = tmp_path / "large-converted.json"
    _write_json(
        input_path, {**source, "record_count": len(records), "records": records}
    )

    result = convert_mem0_export(input_path, output_path)

    converted = _read_json(output_path)
    assert result["converted"] == 5000
    assert converted["export_metadata"]["total_memories"] == 5000
    assert len(converted["memories"]) == 5000
    assert converted["memories"][-1]["content"] == "Large export memory 04999"


@pytest.mark.asyncio
async def test_real_sqlite_vec_import_duplicate_and_authorship(temp_db_path, tmp_path):
    converted_path = tmp_path / "converted.json"
    convert_mem0_export(FIXTURE, converted_path)

    storage = SqliteVecMemoryStorage(str(Path(temp_db_path) / "mem0-import.db"))
    await storage.initialize()
    try:
        importer = MemoryImporter(storage)
        first = await importer.import_from_json([converted_path], add_source_tags=False)
        second = await importer.import_from_json(
            [converted_path], add_source_tags=False
        )

        assert first["imported"] == 2
        assert first["duplicates_skipped"] == 0
        assert second["imported"] == 0
        assert second["duplicates_skipped"] == 2

        memories = await storage.get_all_memories()
        first_memory = next(
            memory for memory in memories if memory.content == "User likes dark mode"
        )
        assert first_memory.agent_id == "agent-1"
        assert {"user:alice", "agent:agent-1", "sys:mem0-run:run-1"} <= set(
            first_memory.tags
        )
        assert all(validate_tag(tag) for tag in first_memory.tags)
        assert first_memory.metadata["mem0_run_id"] == "run-1"
        assert (
            first_memory.created_at
            == datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
        )
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_identity_scoped_duplicates_reach_sqlite_vec_importer(
    temp_db_path, tmp_path
):
    input_path = tmp_path / "identity-duplicates.json"
    converted_path = tmp_path / "identity-duplicates-converted.json"
    created_at = "2026-05-01T00:00:00Z"
    _write_json(
        input_path,
        [
            {
                "id": "m1",
                "memory": "Shared preference",
                "created_at": created_at,
                "user_id": "alice",
            },
            {
                "id": "m2",
                "memory": "Shared preference",
                "created_at": created_at,
                "user_id": "bob",
            },
        ],
    )
    convert_mem0_export(input_path, converted_path)

    storage = SqliteVecMemoryStorage(str(Path(temp_db_path) / "identity-import.db"))
    await storage.initialize()
    try:
        importer = MemoryImporter(storage)
        first = await importer.import_from_json([converted_path])
        second = await importer.import_from_json([converted_path])

        assert first["imported"] == 2
        assert first["duplicates_skipped"] == 0
        assert second["imported"] == 0
        assert second["duplicates_skipped"] == 2

        memories = await storage.get_all_memories()
        assert len(memories) == 2
        assert {tuple(sorted(memory.tags)) for memory in memories} == {
            ("source:mem0", "user:alice"),
            ("source:mem0", "user:bob"),
        }
    finally:
        await storage.close()
