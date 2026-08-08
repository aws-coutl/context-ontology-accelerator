# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AOSS NEXTGEN ``_source`` projection patch.

These assert on the request body the patched ``paginated_search`` sends, which is
the actual defect: without an explicit ``_source`` include, AOSS NEXTGEN omits the
``knn_vector`` field and the toolkit's ``_to_get_embedding_result`` raises
``KeyError: 'embedding'`` — swallowed upstream, collapsing the topic beam.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Load the patch module directly from its file rather than through
# ``coa_serve.lexical``, whose package __init__ imports graphrag_toolkit
# at module load. The module under test has no graphrag import-time dependency (all
# toolkit access is lazy), so a direct load keeps this test runnable without the
# heavyweight optional deps installed.
_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "coa_serve" / "lexical" / "graphrag_patches.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("_graphrag_patches_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeOsClient:
    """Records search bodies and replays canned pages."""

    def __init__(self, pages):
        self._pages = list(pages)
        self.bodies = []

    def search(self, index, body):
        self.bodies.append(body)
        hits = self._pages.pop(0) if self._pages else []
        return {"hits": {"hits": hits}}


class _FakeIndex:
    def __init__(self, client):
        self.client = types.SimpleNamespace(_os_client=client)

    def underlying_index_name(self):
        return "topic"


@pytest.fixture
def patched_module(monkeypatch):
    """Install a stub graphrag module tree, then apply the patch to it."""
    calls = []

    def original(self, query, page_size=10000, max_pages=None, ids_only=False):
        calls.append({"ids_only": ids_only, "page_size": page_size})
        return iter([])

    index_cls = type("OpenSearchIndex", (), {"paginated_search": original})
    ovi = types.ModuleType("graphrag_toolkit.lexical_graph.storage.vector.opensearch_vector_indexes")
    ovi.OpenSearchIndex = index_cls

    for name in (
        "graphrag_toolkit",
        "graphrag_toolkit.lexical_graph",
        "graphrag_toolkit.lexical_graph.storage",
        "graphrag_toolkit.lexical_graph.storage.vector",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "graphrag_toolkit.lexical_graph.storage.vector.opensearch_vector_indexes", ovi)

    graphrag_patches = _load_patch_module()
    graphrag_patches.patch_paginated_search_source()
    return index_cls, calls, graphrag_patches


def test_search_body_projects_embedding(patched_module):
    """The regression guard: 'embedding' must be explicitly requested in _source."""
    index_cls, _, _mod = patched_module
    client = _FakeOsClient(pages=[[]])
    index = _FakeIndex(client)

    list(index_cls.paginated_search(index, {"terms": {"id": ["t1"]}}))

    assert len(client.bodies) == 1
    source = client.bodies[0]["_source"]
    assert "embedding" in source["includes"], "knn_vector must be projected or the beam collapses"
    # The other fields _to_get_embedding_result subscripts.
    for field in ("id", "value", "metadata"):
        assert field in source["includes"]


def test_ids_only_delegates_and_keeps_source_suppressed(patched_module):
    """ids_only must keep upstream's ``_source: False`` behavior (no widening)."""
    index_cls, calls, _mod = patched_module
    client = _FakeOsClient(pages=[[]])
    index = _FakeIndex(client)

    list(index_cls.paginated_search(index, {"match_all": {}}, ids_only=True))

    assert calls == [{"ids_only": True, "page_size": 10000}]
    assert client.bodies == [], "ids_only must not issue its own widened search"


def test_paginates_via_search_after(patched_module):
    """Multi-page walk still threads ``search_after`` from the last hit's sort key."""
    index_cls, _, _mod = patched_module
    page1 = [{"_source": {"id": "a"}, "sort": ["a"]}]
    page2 = [{"_source": {"id": "b"}, "sort": ["b"]}]
    client = _FakeOsClient(pages=[page1, page2, []])
    index = _FakeIndex(client)

    pages = list(index_cls.paginated_search(index, {"match_all": {}}))

    assert pages == [page1, page2]
    assert "search_after" not in client.bodies[0]
    assert client.bodies[1]["search_after"] == ["a"]
    assert client.bodies[2]["search_after"] == ["b"]


def test_max_pages_stops_early(patched_module):
    index_cls, _, _mod = patched_module
    pages = [[{"_source": {}, "sort": [str(i)]}] for i in range(5)]
    client = _FakeOsClient(pages=pages)
    index = _FakeIndex(client)

    got = list(index_cls.paginated_search(index, {"match_all": {}}, max_pages=2))

    assert len(got) == 2


def test_patch_is_idempotent(patched_module):
    """Re-applying must not wrap the already-patched function again."""
    index_cls, _, mod = patched_module

    first = index_cls.paginated_search
    mod.patch_paginated_search_source()
    assert index_cls.paginated_search is first
