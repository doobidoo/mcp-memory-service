from pathlib import Path

from mcp_memory_service.sync.converters.mem0 import convert_mem0_export

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mem0" / "oss_qdrant_export.json"


def test_converts_real_mem0_export_to_importer_shape(tmp_path):
    output_path = tmp_path / "converted.json"

    result = convert_mem0_export(FIXTURE, output_path)

    assert result["converted"] == 2
    assert output_path.exists()


import copy
import json
from datetime import datetime, timezone

import pytest

from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage
from mcp_memory_service.sync.importer import MemoryImporter
from mcp_memory_service.utils.hashing import generate_content_hash


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as source:
        return json.load(source)


def _write_json(path: Path, payload):
    with open(path, "w", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, ensure_ascii=False)


def test_maps_real_mem0_fields_and_preserves_timestamps(tmp_path):
    output_path = tmp_path / "converted.json"

    convert_mem0_export(FIXTURE, output_path)
    converted = _read_json(output_path)
    first = converted["memories"][0]

    assert converted["export_metadata"]["source_machine"] == "mem0"
    assert converted["export_metadata"]["total_memories"] == 2
    assert first["content"] == "User likes dark mode"
    assert first["content_hash"] == generate_content_hash("User likes dark mode")
    assert first["created_at"] == datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
    assert first["updated_at"] == first["created_at"]
    assert first["tags"] == [
        "user:alice",
        "agent:agent-1",
        "sys:mem0-run:run-1",
    ]
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
        assert _read_json(output_path)["memories"][0]["content"] in {
            "one",
            "two",
            "three",
        }


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


def test_merges_duplicate_content_with_distinct_identities(tmp_path):
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

    assert result["converted"] == 1
    memory = _read_json(output_path)["memories"][0]
    assert {
        "user:alice",
        "user:bob",
        "agent:agent-a",
        "agent:agent-b",
        "sys:mem0-run:run-a",
        "sys:mem0-run:run-b",
    } <= set(memory["tags"])
    assert memory["metadata"]["mem0_record_id"] == ["m1", "m2"]
    assert memory["metadata"]["mem0_user_id"] == ["alice", "bob"]
    assert memory["metadata"]["agent_id"] == "agent-a"


def test_uses_conversion_time_when_timestamp_is_missing(tmp_path):
    input_path = tmp_path / "missing-time.json"
    output_path = tmp_path / "missing-time-converted.json"
    _write_json(input_path, [{"id": "m1", "memory": "No source timestamp"}])

    before = datetime.now(timezone.utc).timestamp()
    convert_mem0_export(input_path, output_path)
    after = datetime.now(timezone.utc).timestamp()

    memory = _read_json(output_path)["memories"][0]
    assert before <= memory["created_at"] <= after
    assert memory["updated_at"] == memory["created_at"]
    assert memory["metadata"]["mem0_timestamp_source"] == "conversion_time"


@pytest.mark.parametrize(
    "payload",
    [
        {"unexpected": []},
        {"records": [{"id": "missing-content", "created_at": "2026-05-01T00:00:00Z"}]},
        {"records": [{"id": "bad-time", "memory": "x", "created_at": "not-a-time"}]},
    ],
)
def test_rejects_invalid_schemas(tmp_path, payload):
    input_path = tmp_path / "invalid.json"
    output_path = tmp_path / "converted.json"
    _write_json(input_path, payload)

    with pytest.raises(ValueError):
        convert_mem0_export(input_path, output_path)


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
        first = await importer.import_from_json([converted_path])
        second = await importer.import_from_json([converted_path])

        assert first["imported"] == 2
        assert first["duplicates_skipped"] == 0
        assert second["imported"] == 0
        assert second["duplicates_skipped"] == 2

        memories = await storage.get_all_memories()
        first_memory = next(
            memory for memory in memories if memory.content == "User likes dark mode"
        )
        assert first_memory.agent_id == "agent-1"
        assert {
            "user:alice",
            "agent:agent-1",
            "sys:mem0-run:run-1",
            "source:mem0",
        } <= set(first_memory.tags)
        assert first_memory.metadata["mem0_run_id"] == "run-1"
        assert (
            first_memory.created_at
            == datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
        )
    finally:
        await storage.close()
