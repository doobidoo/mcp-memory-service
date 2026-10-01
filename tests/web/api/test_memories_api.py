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
Tests for Memory CRUD API endpoints.

Tests the store_memory endpoint for:
- X-Agent-ID header auto-tagging (agent:<id> appended to tags)
"""

import pytest
import pytest_asyncio
import tempfile
import os
from fastapi.testclient import TestClient

from mcp_memory_service.web.dependencies import set_storage
from mcp_memory_service.storage.sqlite_vec import SqliteVecMemoryStorage


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memories.db")
        yield db_path


@pytest_asyncio.fixture
async def initialized_storage(temp_db, monkeypatch):
    """Create and initialize a real SQLite storage backend."""
    monkeypatch.setenv('MCP_SEMANTIC_DEDUP_ENABLED', 'false')

    storage = SqliteVecMemoryStorage(temp_db)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def test_app(initialized_storage, monkeypatch):
    """Create a FastAPI test application with initialized storage."""
    # Auth is bypassed via `app.dependency_overrides` below — we deliberately
    # do NOT `importlib.reload` the middleware module here. Reloading rebinds
    # the dependency functions to fresh objects while the FastAPI route graph
    # still holds the *original* references captured at app-import time;
    # subsequent tests (e.g. `test_harvest_api::test_harvest_requires_auth`)
    # then register overrides keyed by the post-reload objects, miss the
    # route-captured originals, and fall through to the live middleware,
    # which under the CI-wide `MCP_ALLOW_ANONYMOUS_ACCESS=true` silently
    # returns 200 instead of 401. See PR #844 / issue #843 follow-up.
    monkeypatch.setenv('MCP_API_KEY', '')
    monkeypatch.setenv('MCP_OAUTH_ENABLED', 'false')
    monkeypatch.setenv('MCP_ALLOW_ANONYMOUS_ACCESS', 'true')
    monkeypatch.setenv('INCLUDE_HOSTNAME', 'false')

    from mcp_memory_service.web.app import app
    from mcp_memory_service.web.oauth.middleware import (
        get_current_user, require_write_access, require_read_access,
        AuthenticationResult
    )

    set_storage(initialized_storage)

    async def mock_get_current_user():
        return AuthenticationResult(
            authenticated=True,
            client_id="test_client",
            scope="read write admin",
            auth_method="test"
        )

    app.dependency_overrides[get_current_user] = mock_get_current_user
    app.dependency_overrides[require_write_access] = mock_get_current_user
    app.dependency_overrides[require_read_access] = mock_get_current_user

    client = TestClient(app)
    yield client

    app.dependency_overrides.clear()


@pytest.mark.integration
def test_store_memory_with_agent_id_header_appends_agent_tag(test_app):
    """X-Agent-ID header auto-appends agent:<id> tag to stored memory."""
    response = test_app.post(
        "/api/memories",
        json={
            "content": "Researcher found that the API rate limit is 100 req/min",
            "tags": ["api", "rate-limit"],
        },
        headers={"X-Agent-ID": "researcher"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    stored_tags = data["memory"]["tags"]
    assert "agent:researcher" in stored_tags
    assert "api" in stored_tags
    assert "rate-limit" in stored_tags


@pytest.mark.integration
def test_store_memory_without_agent_id_header_no_agent_tag(test_app):
    """Without X-Agent-ID header, no agent: tag is added."""
    response = test_app.post(
        "/api/memories",
        json={
            "content": "Regular memory without agent context",
            "tags": ["general"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    stored_tags = data["memory"]["tags"]
    assert not any(tag.startswith("agent:") for tag in stored_tags)


@pytest.mark.integration
def test_store_memory_agent_id_not_duplicated_if_already_in_tags(test_app):
    """X-Agent-ID header does not duplicate an agent: tag already in tags."""
    response = test_app.post(
        "/api/memories",
        json={
            "content": "Memory with pre-existing agent tag",
            "tags": ["agent:researcher", "other"],
        },
        headers={"X-Agent-ID": "researcher"},
    )

    assert response.status_code == 200
    data = response.json()
    stored_tags = data["memory"]["tags"]
    assert stored_tags.count("agent:researcher") == 1


@pytest.mark.integration
def test_list_memories_tag_match_any_returns_union(test_app):
    """tag_match=any should return memories with ANY of the specified tags (OR logic)."""
    # Store memory with tag "python"
    test_app.post("/api/memories", json={
        "content": "Python is great for scripting",
        "tags": ["python", "scripting"]
    })
    # Store memory with tag "reference"
    test_app.post("/api/memories", json={
        "content": "Reference guide for Docker commands",
        "tags": ["reference", "docker"]
    })
    # Store memory with neither tag
    test_app.post("/api/memories", json={
        "content": "Unrelated memory about lunch",
        "tags": ["personal"]
    })

    # Query with tag_match=any for "python" — should find the first
    response = test_app.get("/api/memories", params={"tag": "python", "tag_match": "any"})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    # All returned memories should have "python" tag
    for mem in data["memories"]:
        assert "python" in mem["tags"]


@pytest.mark.integration
def test_list_memories_tag_match_all_returns_intersection(test_app):
    """tag_match=all should return only memories with ALL specified tags (AND logic)."""
    # Store memory with both tags
    test_app.post("/api/memories", json={
        "content": "Python reference for async patterns",
        "tags": ["python", "reference"]
    })
    # Store memory with only "python"
    test_app.post("/api/memories", json={
        "content": "Python basics tutorial",
        "tags": ["python", "tutorial"]
    })

    # Query with tag_match=all for "python,reference" — should find only the first
    response = test_app.get("/api/memories", params={"tag": "python,reference", "tag_match": "all"})
    assert response.status_code == 200
    data = response.json()
    # Only memories with BOTH tags should be returned
    for mem in data["memories"]:
        assert "python" in mem["tags"]
        assert "reference" in mem["tags"]


@pytest.mark.integration
def test_list_memories_tag_match_any_vs_all_different_results(test_app):
    """ANY and ALL modes should produce different result sets for the same tags."""
    # Store memory with both tags
    test_app.post("/api/memories", json={
        "content": "Has both alpha and beta tags",
        "tags": ["alpha", "beta"]
    })
    # Store memory with only alpha
    test_app.post("/api/memories", json={
        "content": "Has only alpha tag",
        "tags": ["alpha", "gamma"]
    })

    # ANY should return both (either alpha OR beta)
    resp_any = test_app.get("/api/memories", params={"tag": "alpha,beta", "tag_match": "any"})
    # ALL should return only the first (both alpha AND beta)
    resp_all = test_app.get("/api/memories", params={"tag": "alpha,beta", "tag_match": "all"})

    assert resp_any.status_code == 200
    assert resp_all.status_code == 200

    any_total = resp_any.json()["total"]
    all_total = resp_all.json()["total"]

    # ANY should return more results than ALL
    assert any_total > all_total
    assert all_total >= 1


@pytest.mark.integration
def test_http_store_scope_memory_search_tags_and_analytics(test_app):
    home = "Home store alpha memory about the garden"
    work = "Work store beta memory about the roadmap"
    for content, store, tag in ((home, "home", "home-only"), (work, "work", "work-only")):
        response = test_app.post(
            "/api/memories",
            json={"content": content, "tags": [tag], "store": store},
        )
        assert response.status_code == 200, response.text
        assert response.json()["success"] is True, response.text

    listed = test_app.get("/api/memories", params={"store": "home"}).json()
    assert listed["total"] == 1
    assert [memory["content"] for memory in listed["memories"]] == [home]
    assert test_app.get("/api/memories").json()["total"] == 0

    searched = test_app.post(
        "/api/search",
        json={"query": home, "n_results": 10, "store": "home"},
    )
    assert searched.status_code == 200, searched.text
    assert {result["memory"]["content"] for result in searched.json()["results"]} == {home}
    assert test_app.post("/api/search", json={"query": home}).json()["results"] == []

    tags = test_app.get("/api/tags", params={"store": "home"}).json()["tags"]
    assert tags == [{"tag": "home-only", "count": 1}]
    assert test_app.get("/api/tags").json()["tags"] == []

    home_types = test_app.get("/api/analytics/memory-types", params={"store": "home"}).json()
    assert home_types["total_memories"] == 1
    assert test_app.get("/api/analytics/memory-types").json()["total_memories"] == 0
    all_types = test_app.get("/api/analytics/memory-types", params={"store": "all"}).json()
    assert all_types["total_memories"] == 2

    stats = test_app.get("/api/analytics/storage-stats", params={"store": "home"}).json()
    previews = [item["preview"] for item in stats["largest_memories"]]
    assert previews and all("Home store" in preview for preview in previews)
    all_stats = test_app.get("/api/analytics/storage-stats", params={"store": "all"}).json()
    all_previews = [item["preview"] for item in all_stats["largest_memories"]]
    assert any("Home store" in preview for preview in all_previews)
    assert any("Work store" in preview for preview in all_previews)


@pytest.mark.integration
def test_hash_crud_respects_store_scope(test_app):
    """Hash-level CRUD must not cross store boundaries (#1106).

    Before the fix a client scoped to the default store could read, update or
    delete a memory that lived in another partition just by knowing its hash.
    """
    response = test_app.post(
        "/api/memories",
        json={"content": "Work store roadmap entry", "store": "work"},
    )
    assert response.status_code == 200, response.text
    content_hash = response.json()["memory"]["content_hash"]

    # The default scope must treat it as missing.
    assert test_app.get(f"/api/memories/{content_hash}").status_code == 404
    assert (
        test_app.put(f"/api/memories/{content_hash}", json={"tags": ["hijacked"]}).status_code
        == 404
    )
    assert test_app.delete(f"/api/memories/{content_hash}").status_code == 404

    # `all` is the explicit federated read scope.
    assert (
        test_app.get(f"/api/memories/{content_hash}", params={"store": "all"}).status_code == 200
    )

    # The owning scope still reads and deletes it.
    own = test_app.get(f"/api/memories/{content_hash}", params={"store": "work"})
    assert own.status_code == 200
    assert own.json()["content"] == "Work store roadmap entry"

    assert (
        test_app.delete(f"/api/memories/{content_hash}", params={"store": "work"}).status_code
        == 200
    )
    assert (
        test_app.get(f"/api/memories/{content_hash}", params={"store": "work"}).status_code == 404
    )


@pytest.mark.integration
def test_store_all_write_is_rejected_with_400(test_app):
    """`store='all'` is read-only: the API must answer 400, not 500 (#1106)."""
    response = test_app.post(
        "/api/memories",
        json={"content": "must not be stored", "store": "all"},
    )
    assert response.status_code == 400, response.text
    assert "read scopes" in response.json()["detail"]

    session = test_app.post(
        "/api/sessions",
        json={"turns": [{"role": "user", "content": "hi"}], "store": "all"},
    )
    assert session.status_code == 400, session.text


@pytest.mark.integration
def test_memory_type_distribution_excludes_soft_deleted(test_app):
    """Soft-deleted tombstones must not inflate the type breakdown (#1106)."""
    for content in ("Keep me in home", "Delete me from home"):
        stored = test_app.post(
            "/api/memories", json={"content": content, "store": "home"}
        )
        assert stored.status_code == 200, stored.text

    before = test_app.get("/api/analytics/memory-types", params={"store": "home"}).json()
    assert before["total_memories"] == 2

    memories = test_app.get("/api/memories", params={"store": "home"}).json()["memories"]
    target = next(m for m in memories if m["content"].startswith("Delete me"))
    deleted = test_app.delete(
        f"/api/memories/{target['content_hash']}", params={"store": "home"}
    )
    assert deleted.status_code == 200, deleted.text

    after = test_app.get("/api/analytics/memory-types", params={"store": "home"}).json()
    assert after["total_memories"] == 1


@pytest.mark.integration
def test_time_search_applies_store_scope(test_app):
    """Time-only search stays inside the requested partition (#1106)."""
    for content, store in (("Home timeline entry", "home"), ("Work timeline entry", "work")):
        stored = test_app.post("/api/memories", json={"content": content, "store": store})
        assert stored.status_code == 200, stored.text

    # "today" is a window that contains the rows just created; "last week" is
    # the previous calendar week and would legitimately return nothing.
    response = test_app.post(
        "/api/search/by-time",
        json={"query": "today", "n_results": 10, "store": "home"},
    )
    assert response.status_code == 200, response.text
    contents = {result["memory"]["content"] for result in response.json()["results"]}
    assert contents == {"Home timeline entry"}


@pytest.mark.integration
def test_overview_scoped_metrics_count_beyond_sample(test_app):
    """Overview counts must not be capped by an internal sample size (#1106)."""
    for index in range(5):
        stored = test_app.post(
            "/api/memories",
            json={
                "content": f"Scoped overview memory {index}",
                "store": "home",
                "tags": ["ov-scope"],
            },
        )
        assert stored.status_code == 200, stored.text
    stored = test_app.post(
        "/api/memories", json={"content": "Other store memory", "store": "work"}
    )
    assert stored.status_code == 200, stored.text

    overview = test_app.get("/api/analytics/overview", params={"store": "home"}).json()
    assert overview["total_memories"] == 5
    assert overview["memories_this_week"] == 5
    assert overview["memories_this_month"] == 5
    assert overview["unique_tags"] == 1


@pytest.mark.integration
def test_semantic_time_search_does_not_leak_out_of_window(test_app):
    """A semantic by-time search must not return memories outside the window.

    The route goes through `recall()`, the one storage method whose contract is
    "time-windowed retrieval, semantically ranked when a query is given". Using
    `retrieve()` instead let out-of-window neighbours fill the candidate pool and
    either hid in-window matches or leaked results from outside the range (#1106).
    """
    stored = test_app.post(
        "/api/memories",
        json={"content": "Window probe about rivers", "store": "home"},
    )
    assert stored.status_code == 200, stored.text

    # "yesterday" excludes the row just written.
    outside = test_app.post(
        "/api/search/by-time",
        json={
            "query": "yesterday",
            "semantic_query": "rivers",
            "n_results": 10,
            "store": "home",
        },
    )
    assert outside.status_code == 200, outside.text
    assert outside.json()["results"] == []

    # "today" includes it.
    inside = test_app.post(
        "/api/search/by-time",
        json={
            "query": "today",
            "semantic_query": "rivers",
            "n_results": 10,
            "store": "home",
        },
    )
    assert inside.status_code == 200, inside.text
    contents = {result["memory"]["content"] for result in inside.json()["results"]}
    assert contents == {"Window probe about rivers"}


@pytest.mark.integration
def test_by_time_search_selects_candidates_with_the_window(test_app, initialized_storage, monkeypatch):
    """The time window must reach storage *during* candidate selection (#1106).

    An earlier revision pushed the window into `retrieve()`, which Milvus accepts
    but never reads: the search then post-filtered a windowless top-N, so
    out-of-window neighbours could fill the pool and the window came back empty.
    The route now selects through `recall()`, which every backend applies the
    window in. This test fails if that regresses to retrieve()+post-filter.
    """
    calls = {"recall": [], "retrieve": []}
    real_recall = initialized_storage.recall

    async def spy_recall(**kwargs):
        calls["recall"].append(kwargs)
        return await real_recall(**kwargs)

    async def spy_retrieve(**kwargs):
        calls["retrieve"].append(kwargs)
        return []

    monkeypatch.setattr(initialized_storage, "recall", spy_recall)
    monkeypatch.setattr(initialized_storage, "retrieve", spy_retrieve)

    stored = test_app.post(
        "/api/memories",
        json={"content": "Candidate selection probe", "store": "home"},
    )
    assert stored.status_code == 200, stored.text

    response = test_app.post(
        "/api/search/by-time",
        json={
            "query": "today",
            "semantic_query": "candidate selection",
            "n_results": 5,
            "store": "home",
        },
    )
    assert response.status_code == 200, response.text

    assert calls["recall"], "by-time search must select candidates through recall()"
    kwargs = calls["recall"][0]
    assert kwargs["start_timestamp"] is not None
    assert kwargs["end_timestamp"] is not None
    assert kwargs["store"] == "home"
    assert calls["retrieve"] == [], (
        "the window must be applied while selecting candidates, not after a "
        "windowless retrieve()"
    )
