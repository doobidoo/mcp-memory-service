"""Convert mem0 exports into MemoryImporter-compatible JSON."""

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...utils.hashing import generate_content_hash

_MISSING = object()
_MEMORY_COLLECTION_KEYS = ("records", "results", "memories")


class Mem0ExportError(ValueError):
    """Raised when a mem0 export does not match a supported shape."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_path(value: str | Path, *, field: str) -> Path:
    """Expand user paths and reject empty or non-path values."""
    if not isinstance(value, (str, Path)):
        raise Mem0ExportError(f"Invalid {field} path: expected a string or Path")
    if isinstance(value, str) and not value.strip():
        raise Mem0ExportError(f"Invalid {field} path: path is empty")

    try:
        return Path(value).expanduser()
    except (RuntimeError, TypeError) as exc:
        raise Mem0ExportError(f"Invalid {field} path {value!r}: {exc}") from exc


def _load_export_records(
    payload: Any,
) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
        source_metadata: dict[str, Any] = {}
    elif isinstance(payload, dict):
        records = _MISSING
        for key in _MEMORY_COLLECTION_KEYS:
            if key in payload:
                records = payload[key]
                break
        else:
            data = payload.get("data")
            if isinstance(data, dict):
                nested_records, nested_metadata = _load_export_records(data)
                return nested_records, {
                    "kind": payload.get("kind", nested_metadata.get("kind")),
                    "version": payload.get("version", nested_metadata.get("version")),
                    "exported_at": payload.get(
                        "exported_at", nested_metadata.get("exported_at")
                    ),
                    "source": payload.get("source", nested_metadata.get("source")),
                }

        source_metadata = {
            "kind": payload.get("kind", "mem0_export"),
            "version": payload.get("version"),
            "exported_at": payload.get("exported_at"),
            "source": payload.get("source"),
        }
    else:
        raise Mem0ExportError(
            "Unsupported mem0 export: expected an object with records/results/memories "
            "or a JSON array"
        )

    if records is _MISSING or not isinstance(records, list):
        raise Mem0ExportError(
            "Unsupported mem0 export: memory collection is not an array"
        )

    return records, source_metadata


def _parse_timestamp(value: Any, *, field: str, index: int) -> float:
    if isinstance(value, bool):
        raise Mem0ExportError(f"Invalid {field} in mem0 record {index}")

    if isinstance(value, (int, float)):
        try:
            timestamp = float(value)
        except OverflowError as exc:
            raise Mem0ExportError(f"Invalid {field} in mem0 record {index}") from exc
        if not math.isfinite(timestamp):
            raise Mem0ExportError(f"Invalid {field} in mem0 record {index}")
        if timestamp > 100_000_000_000:
            timestamp /= 1000.0
        return timestamp

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise Mem0ExportError(f"Invalid {field} in mem0 record {index}")
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise Mem0ExportError(
                f"Invalid {field} in mem0 record {index}: expected ISO-8601 or Unix time"
            ) from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    raise Mem0ExportError(f"Invalid {field} in mem0 record {index}")


def _record_value(
    record: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    value = record.get(key, _MISSING)
    if value is _MISSING or value is None or value == "":
        value = source_payload.get(key, default)
    return value


def _record_tags(record: Mapping[str, Any]) -> list[str]:
    raw_tags = record.get("tags")
    if not isinstance(raw_tags, list):
        return []
    return [tag.strip() for tag in raw_tags if isinstance(tag, str) and tag.strip()]


def _identity_payload(
    *,
    user_id: Any,
    agent_id: Any,
    run_id: Any,
    tags: list[str],
) -> dict[str, Any]:
    return {
        "user_id": str(user_id) if user_id else None,
        "agent_id": str(agent_id) if agent_id else None,
        "run_id": str(run_id) if run_id else None,
        "tags": sorted(tags),
    }


def _generate_dedupe_hash(content: str, identity: Mapping[str, Any]) -> str:
    """Scope mem0 dedupe identity without changing the shared content hash helper."""
    if not any(identity.values()):
        return generate_content_hash(content)

    payload = json.dumps(
        {
            "kind": "mem0_identity_scoped",
            "content": content.strip().lower(),
            "identity": identity,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_record(
    record: Mapping[str, Any], index: int, fallback_timestamp: Any
) -> dict[str, Any]:
    source_payload = record.get("source_payload")
    if not isinstance(source_payload, Mapping):
        source_payload = {}

    content = record.get("memory")
    if not isinstance(content, str) or not content.strip():
        content = record.get("content")
    if not isinstance(content, str) or not content.strip():
        content = source_payload.get("data")
    if not isinstance(content, str) or not content.strip():
        raise Mem0ExportError(
            f"Invalid mem0 record {index}: memory/content must be a non-empty string"
        )

    created_at_raw = _record_value(record, source_payload, "created_at")
    updated_at_raw = _record_value(record, source_payload, "updated_at")
    timestamp_source = "record"

    if created_at_raw is None:
        if updated_at_raw is not None:
            created_at_raw = updated_at_raw
            timestamp_source = "updated_at"
        elif fallback_timestamp is not None:
            created_at_raw = fallback_timestamp
            timestamp_source = "exported_at"
        else:
            created_at_raw = datetime.now(timezone.utc).timestamp()
            timestamp_source = "conversion_time"

    created_at = _parse_timestamp(created_at_raw, field="created_at", index=index)
    updated_at = (
        _parse_timestamp(updated_at_raw, field="updated_at", index=index)
        if updated_at_raw is not None
        else created_at
    )

    user_id = _record_value(record, source_payload, "user_id")
    agent_id = _record_value(record, source_payload, "agent_id")
    run_id = _record_value(record, source_payload, "run_id")

    tags = _record_tags(record)
    if user_id:
        tags.append(f"user:{user_id}")
    if agent_id:
        tags.append(f"agent:{agent_id}")
    if run_id:
        tags.append(f"sys:mem0-run:{run_id}")

    source_metadata = record.get("metadata", {})
    if source_metadata is None:
        source_metadata = {}
    if not isinstance(source_metadata, Mapping):
        raise Mem0ExportError(
            f"Invalid mem0 record {index}: metadata must be an object"
        )
    metadata = dict(source_metadata)
    if source_payload:
        metadata["mem0_source_payload"] = dict(source_payload)

    metadata_fields = {
        "mem0_record_id": record.get("id"),
        "mem0_user_id": user_id,
        "mem0_agent_id": agent_id,
        "mem0_run_id": run_id,
        "mem0_app_id": _record_value(record, source_payload, "app_id"),
        "mem0_actor_id": _record_value(record, source_payload, "actor_id"),
        "mem0_role": _record_value(record, source_payload, "role"),
        "mem0_hash": _record_value(record, source_payload, "hash"),
        "mem0_source": _record_value(record, source_payload, "source"),
    }
    for key, value in metadata_fields.items():
        if value is not None:
            metadata[key] = value

    categories = _record_value(record, source_payload, "categories")
    if categories is not None:
        metadata["mem0_categories"] = categories
    if timestamp_source != "record":
        metadata["mem0_timestamp_source"] = timestamp_source

    if agent_id:
        # Memory.agent_id is metadata-backed; this is the canonical author field.
        metadata["agent_id"] = agent_id

    identity = _identity_payload(
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
        tags=tags,
    )
    return {
        "content": content,
        "content_hash": _generate_dedupe_hash(content, identity),
        "tags": tags,
        "created_at": created_at,
        "updated_at": updated_at,
        "memory_type": record.get("memory_type", "note"),
        "metadata": metadata,
    }


def _normalize_exported_at(value: Any, fallback: Any) -> float | None:
    candidate = value if value is not None else fallback
    if candidate is None:
        return None
    return _parse_timestamp(candidate, field="exported_at", index=0)


def _conversion_warning(
    *, record_index: int | None, code: str, action: str, message: str
) -> dict[str, Any]:
    return {
        "record_index": record_index,
        "code": code,
        "action": action,
        "message": message,
    }


def convert_mem0_export(
    input_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    """Convert a mem0 export file into the MemoryImporter JSON format."""
    input_path = _normalize_path(input_path, field="input")
    output_path = _normalize_path(output_path, field="output")

    if not input_path.is_file():
        raise Mem0ExportError(f"Input path is not a file: {input_path}")
    if output_path.exists() and output_path.is_dir():
        raise Mem0ExportError(f"Output path is a directory: {output_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise Mem0ExportError(
            f"Could not read mem0 export {input_path}: {exc}"
        ) from exc

    records, source_metadata = _load_export_records(payload)
    warnings: list[dict[str, Any]] = []

    try:
        fallback_timestamp = _normalize_exported_at(
            source_metadata.get("exported_at"),
            payload.get("exported_at") if isinstance(payload, dict) else None,
        )
    except Mem0ExportError as exc:
        fallback_timestamp = None
        warnings.append(
            _conversion_warning(
                record_index=None,
                code="invalid_exported_at",
                action="ignored",
                message=str(exc),
            )
        )

    memories = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            warnings.append(
                _conversion_warning(
                    record_index=index,
                    code="invalid_record",
                    action="skipped",
                    message=f"Invalid mem0 record {index}: expected an object",
                )
            )
            continue

        try:
            memory = _parse_record(record, index, fallback_timestamp)
        except Mem0ExportError as exc:
            warnings.append(
                _conversion_warning(
                    record_index=index,
                    code="invalid_record",
                    action="skipped",
                    message=str(exc),
                )
            )
            continue

        timestamp_source = memory["metadata"].get("mem0_timestamp_source")
        if timestamp_source:
            warnings.append(
                _conversion_warning(
                    record_index=index,
                    code="missing_created_at",
                    action=f"used_{timestamp_source}",
                    message=(
                        f"mem0 record {index} has no created_at; "
                        f"used {timestamp_source}"
                    ),
                )
            )
        memories.append(memory)

    skipped = sum(1 for warning in warnings if warning["action"] == "skipped")
    output = {
        "export_metadata": {
            "source_machine": "mem0",
            "export_timestamp": _utc_now_iso(),
            "source_record_count": len(records),
            "total_memories": len(memories),
            "skipped_records": skipped,
            "conversion_warnings": warnings,
            "converter_version": "1.1.0",
            "source_export": {
                key: value
                for key, value in source_metadata.items()
                if value is not None
            },
        },
        "memories": memories,
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as target:
            json.dump(output, target, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise Mem0ExportError(
            f"Could not write converted export {output_path}: {exc}"
        ) from exc

    return {
        "source_records": len(records),
        "converted": len(memories),
        "skipped": skipped,
        "warnings": len(warnings),
        "output_file": str(output_path),
        "source_kind": source_metadata.get("kind", "mem0_export"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    """Run the converter as a module."""
    parser = argparse.ArgumentParser(
        description="Convert a mem0 export into MemoryImporter JSON"
    )
    parser.add_argument("input", type=Path, help="mem0 export JSON file")
    parser.add_argument("output", type=Path, help="MemoryImporter JSON output file")
    args = parser.parse_args(argv)

    try:
        result = convert_mem0_export(args.input, args.output)
    except Mem0ExportError as exc:
        parser.exit(2, f"error: {exc}\n")

    sys.stdout.write(
        f"Converted {result['converted']} memories to {result['output_file']}\n"
    )
    if result["skipped"] or result["warnings"]:
        sys.stderr.write(
            f"Wrote {result['warnings']} conversion warning(s); "
            f"see export_metadata.conversion_warnings in {result['output_file']}\n"
        )
    if result["source_records"] and not result["converted"]:
        sys.stderr.write(
            f"error: no memories converted from {result['source_records']} "
            "source record(s)\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
