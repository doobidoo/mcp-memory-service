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

"""REST API for the ontology overlay (#1897, phase 5).

``GET /api/ontology`` returns the merged ontology and the stored rows. The
write routes call the ``SqliteVecMemoryStorage`` overlay methods, so they
validate, record a sync event and bump ``ontology_version`` the same way. The
ontology is global, not store-scoped, so writes need the global ``write``
scope; a ``write:<store>`` token is refused.
"""

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr

from ...config.consolidation import CONSOLIDATION_CONFIG
from ...models import ontology
from ...storage.base import MemoryStorage
from ...storage.mixins.base import _sanitize_log_value
from ...storage.ontology_overlay import parse_row_payload, read_ontology_rows
from ..dependencies import get_storage
from ..oauth.middleware import AuthenticationResult, require_read_access, require_write_access

router = APIRouter()
logger = logging.getLogger(__name__)


class MemoryTypeRequest(BaseModel):
    name: StrictStr
    parent: Optional[StrictStr] = Field(None, description="A base type, or none for a new base type")


class RetentionRequest(BaseModel):
    days: StrictInt = Field(..., description="Retention period in days, 1..3650")


class RelationshipRequest(BaseModel):
    name: StrictStr
    valid_patterns: List[StrictStr] = Field(..., description='e.g. ["decision → error"]; "->" works too')
    description: StrictStr = ""
    symmetric: StrictBool = False


class TypePatternRequest(BaseModel):
    relationship: StrictStr
    confidence: StrictFloat = Field(..., description="In (0, 1]; a JSON number, not a bool or string")


def _writable(storage: MemoryStorage) -> MemoryStorage:
    """The overlay lives in the sqlite_vec database; other backends have no write methods."""
    if not hasattr(storage, "add_memory_type"):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="The ontology overlay needs the sqlite_vec backend; use MCP_CUSTOM_MEMORY_TYPES.",
        )
    return storage


async def _write(write: Callable[[], Awaitable[Any]], missing: str) -> Any:
    """Run an overlay write: ValueError is a 400, a False (nothing to remove) a 404."""
    try:
        result = await write()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    if result is False:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=missing)
    return result


def _stored_rows(storage: MemoryStorage) -> List[Dict[str, Any]]:
    conn = getattr(storage, "conn", None)
    if conn is None:
        return []
    return [
        {"kind": kind, "key": key, "parent": parent, "payload": parse_row_payload(payload_json),
         "updated_at": updated_at}
        for kind, key, parent, payload_json, updated_at, deleted_at in read_ontology_rows(conn)
        if deleted_at is None
    ]


def _parent(memory_type: str) -> Optional[str]:
    """The base type of a subtype; None for a base type (get_parent_type returns itself)."""
    parent = ontology.get_parent_type(memory_type)
    return None if parent == memory_type else parent


@router.get("/ontology")
async def get_ontology(
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_read_access),
) -> Dict[str, Any]:
    """The merged ontology (built-in, ``MCP_CUSTOM_MEMORY_TYPES``, stored) and the stored rows.

    ``stored`` lists the live overlay rows, which are the entries the write routes
    can remove. Everything else is built in or comes from the environment.
    """
    return {
        "memory_types": [{"name": t, "parent": _parent(t)} for t in ontology.get_all_types()],
        "relationships": ontology.get_relationship_types(),
        "type_patterns": [
            {"source": s, "target": t, "relationship": rel, "confidence": conf}
            for (s, t), (rel, conf) in sorted(ontology.get_type_patterns().items())
        ],
        "retention": {
            **ontology.get_retention_overlay(),
            **CONSOLIDATION_CONFIG["retention_periods"],
        },
        "stored": _stored_rows(storage),
        "writable": hasattr(storage, "add_memory_type"),
    }


@router.post("/ontology/memory-types", status_code=status.HTTP_201_CREATED)
async def add_memory_type(
    request: MemoryTypeRequest,
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    s = _writable(storage)
    name = await _write(lambda: s.add_memory_type(request.name, request.parent), "")
    logger.info("Ontology: added memory type %s", _sanitize_log_value(name))
    return {"name": name, "parent": request.parent}


@router.delete("/ontology/memory-types/{name}")
async def remove_memory_type(
    name: str,
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    s = _writable(storage)
    await _write(lambda: s.remove_memory_type(name), "No stored memory type with that name")
    return {"removed": name}


@router.put("/ontology/retention/{memory_type}")
async def set_retention(
    memory_type: str,
    request: RetentionRequest,
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    s = _writable(storage)
    name = await _write(lambda: s.set_retention(memory_type, request.days), "")
    return {"memory_type": name, "days": request.days}


@router.delete("/ontology/retention/{memory_type}")
async def clear_retention(
    memory_type: str,
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    s = _writable(storage)
    await _write(lambda: s.clear_retention(memory_type), "No stored retention for that type")
    return {"removed": memory_type}


@router.post("/ontology/relationships", status_code=status.HTTP_201_CREATED)
async def add_relationship_type(
    request: RelationshipRequest,
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    s = _writable(storage)
    name = await _write(lambda: s.add_relationship_type(
        request.name, request.valid_patterns, request.description, request.symmetric), "")
    logger.info("Ontology: added relationship type %s", _sanitize_log_value(name))
    return {"name": name}


@router.delete("/ontology/relationships/{name}")
async def remove_relationship_type(
    name: str,
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    s = _writable(storage)
    await _write(lambda: s.remove_relationship_type(name), "No stored relationship type with that name")
    return {"removed": name}


@router.put("/ontology/type-patterns/{source}/{target}")
async def set_type_pattern(
    source: str,
    target: str,
    request: TypePatternRequest,
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    s = _writable(storage)
    key = await _write(lambda: s.set_type_pattern(
        source, target, request.relationship, request.confidence), "")
    return {"key": key, "relationship": request.relationship, "confidence": request.confidence}


@router.delete("/ontology/type-patterns/{source}/{target}")
async def clear_type_pattern(
    source: str,
    target: str,
    storage: MemoryStorage = Depends(get_storage),
    _auth: AuthenticationResult = Depends(require_write_access),
) -> Dict[str, Any]:
    s = _writable(storage)
    await _write(lambda: s.clear_type_pattern(source, target), "No stored type pattern for that pair")
    return {"removed": f"{source} → {target}"}
