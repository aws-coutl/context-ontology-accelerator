# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared constants — to_graphrag_tenant_id() and validation helpers."""

import importlib
import re

import pytest
from coa_common.constants import (
    canonical_col,
    graphrag_chunk_index_name,
    ontology_artifact_s3_key,
    ontology_vector_index_name,
    sql_ident,
    to_graphrag_tenant_id,
    validate_id,
    validate_namespace_id,
    validate_namespace_name,
    validate_s3_prefix,
)

pytestmark = pytest.mark.unit


class TestToGraphragTenantId:
    """Verify the namespace_id → TenantId mapping.

    Strips hyphens from the UUID v4 namespace_id and truncates to 25 chars,
    satisfying the GraphRAG TenantId constraint (1–25 lowercase alphanumeric).
    All doc sources within a namespace share the same tenant ID.
    """

    def _expected(self, namespace_id: str) -> str:
        return namespace_id.replace("-", "")[:25]

    def test_basic_uuid(self):
        ns = "550e8400-e29b-41d4-a716-446655440000"
        result = to_graphrag_tenant_id(ns)
        assert result == "550e8400e29b41d4a71644665"

    def test_hyphens_stripped(self):
        ns = "550e8400-e29b-41d4-a716-446655440000"
        result = to_graphrag_tenant_id(ns)
        assert "-" not in result

    def test_always_25_chars_for_uuid(self):
        ns = "550e8400-e29b-41d4-a716-446655440000"
        assert len(to_graphrag_tenant_id(ns)) == 25

    def test_deterministic(self):
        """Same namespace always produces the same tenant ID."""
        ns = "550e8400-e29b-41d4-a716-446655440000"
        assert to_graphrag_tenant_id(ns) == to_graphrag_tenant_id(ns)

    def test_different_namespaces_different_tenant_ids(self):
        """Different namespaces produce different tenant IDs."""
        a = to_graphrag_tenant_id("550e8400-e29b-41d4-a716-446655440000")
        b = to_graphrag_tenant_id("f47ac10b-58cc-4372-a567-0e02b2c3d479")
        assert a != b

    def test_doc_source_id_ignored(self):
        """doc_source_id parameter is accepted but ignored — same namespace always
        produces the same tenant ID regardless of doc_source_id."""
        ns = "550e8400-e29b-41d4-a716-446655440000"
        assert to_graphrag_tenant_id(ns, "ds-1") == to_graphrag_tenant_id(ns, "ds-2")
        assert to_graphrag_tenant_id(ns, "ds-1") == to_graphrag_tenant_id(ns)

    def test_backward_compatible_two_arg_call(self):
        """Existing call sites passing doc_source_id still work."""
        ns = "550e8400-e29b-41d4-a716-446655440000"
        ds = "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        result = to_graphrag_tenant_id(ns, ds)
        assert result == self._expected(ns)
        assert len(result) == 25

    def test_valid_graphrag_tenant_id_format(self):
        """Result satisfies GraphRAG TenantId constraint: 1–25 lowercase alphanumeric."""
        ns = "550e8400-e29b-41d4-a716-446655440000"
        result = to_graphrag_tenant_id(ns)
        assert 1 <= len(result) <= 25
        assert re.fullmatch(r"[0-9a-f]{25}", result), f"Invalid tenant_id: {result}"

    @pytest.mark.parametrize(
        "ns",
        [
            "550e8400-e29b-41d4-a716-446655440000",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "00000000-0000-4000-8000-000000000000",
            "ffffffff-ffff-4fff-bfff-ffffffffffff",
        ],
    )
    def test_always_valid_for_uuid_v4(self, ns: str):
        result = to_graphrag_tenant_id(ns)
        assert len(result) == 25
        assert re.fullmatch(r"[0-9a-f]{25}", result)


class TestValidateId:
    @pytest.mark.parametrize("value", ["abc", "abc-123", "a_b", "ABC_123-x"])
    def test_valid_ids_pass(self, value: str):
        validate_id(value, "myId")  # should not raise

    @pytest.mark.parametrize("value", ["", "has space", "bad/slash", "dot.dot", "semi;colon"])
    def test_invalid_ids_raise(self, value: str):
        with pytest.raises(ValueError, match="Invalid myId"):
            validate_id(value, "myId")


class TestValidateNamespaceId:
    def test_valid_uuid_v4_passes(self):
        validate_namespace_id("550e8400-e29b-41d4-a716-446655440000")

    @pytest.mark.parametrize("value", ["", "not-a-uuid", "550e8400e29b41d4a716446655440000"])
    def test_invalid_raises(self, value: str):
        with pytest.raises(ValueError, match="Must be a UUID v4"):
            validate_namespace_id(value)


class TestValidateNamespaceName:
    @pytest.mark.parametrize("value", ["bird", "bird-benchmark", "a", "A1_b-2"])
    def test_valid_names_pass(self, value: str):
        validate_namespace_name(value)

    @pytest.mark.parametrize("value", ["", "-leading-hyphen", "_underscore", "has space", "x" * 129])
    def test_invalid_names_raise(self, value: str):
        with pytest.raises(ValueError, match="Invalid namespace"):
            validate_namespace_name(value)


class TestValidateS3Prefix:
    def test_empty_string_allowed(self):
        validate_s3_prefix("", "prefix")  # no raise

    @pytest.mark.parametrize("value", ["healthcare/", "documents/2024/", "a-b_c.d/"])
    def test_valid_relative_prefixes_pass(self, value: str):
        validate_s3_prefix(value, "prefix")

    @pytest.mark.parametrize("value", ["/leading", "../traversal", "s3://bucket/x", "has space"])
    def test_unsafe_prefixes_raise(self, value: str):
        with pytest.raises(ValueError, match="Invalid prefix"):
            validate_s3_prefix(value, "prefix")


class TestCanonicalCol:
    def test_lowercases_and_replaces_spaces(self):
        assert canonical_col("First Name") == "first_name"

    def test_strips_illegal_characters(self):
        assert canonical_col("aCL IgG (%)") == "acl_igg_"

    def test_digit_leading_gets_underscore_prefix(self):
        assert canonical_col("1st") == "_1st"

    def test_empty_result_defaults_to_col(self):
        assert canonical_col("()%") == "col"


class TestSqlIdent:
    def test_wraps_in_double_quotes(self):
        assert sql_ident("First Name") == '"First Name"'

    def test_escapes_embedded_double_quotes(self):
        assert sql_ident('a"b') == '"a""b"'

    def test_none_becomes_empty_quoted(self):
        assert sql_ident(None) == '""'  # type: ignore[arg-type]


class TestIndexNameHelpers:
    def test_graphrag_chunk_index_name(self):
        ns = "550e8400-e29b-41d4-a716-446655440000"
        assert graphrag_chunk_index_name(ns) == "chunk_550e8400e29b41d4a71644665"

    def test_ontology_vector_index_name_default_namespace_returns_bare_prefix(self):
        assert ontology_vector_index_name("coa-vectors", "default") == "coa-vectors"

    def test_ontology_vector_index_name_appends_namespace(self):
        assert ontology_vector_index_name("coa-vectors", "ns-1") == "coa-vectors-ns-1"


class TestOntologyArtifactS3Key:
    def test_builds_key(self):
        key = ontology_artifact_s3_key("pc-insurance", "latest", "examples.json")
        assert key == "ontologies/pc-insurance/latest/examples.json"

    def test_path_traversal_filename_raises(self):
        with pytest.raises(ValueError):
            ontology_artifact_s3_key("pc-insurance", "latest", "../secret")

    def test_invalid_namespace_raises(self):
        with pytest.raises(ValueError, match="Invalid namespace"):
            ontology_artifact_s3_key("bad namespace", "latest", "examples.json")


class TestBrandTokensIndependentOfResourcePrefix:
    """Data-plane identifiers must not track RESOURCE_PREFIX.

    CDK injects ``RESOURCE_PREFIX`` as ``{prefix}-{env}-`` (e.g. ``scl-dev-``)
    because its consumers build resource names from it. These tokens used to
    derive from it, so a deployment with ``resource_prefix=scl`` wrote IRIs under
    ``http://scl-dev.amazon.com/vocab/scl-dev#`` while readers used their own
    compiled-in default and silently matched nothing.
    """

    def test_graph_tokens_ignore_resource_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESOURCE_PREFIX", "scl-dev-")
        for var in ("BRAND", "GRAPH_BASE_URI", "URN_PREFIX", "VOCAB_PREFIX", "VOCAB_URI"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("DZ_TYPE_PREFIX", raising=False)
        monkeypatch.delenv("EVENT_SOURCE_PREFIX", raising=False)

        mod = importlib.reload(importlib.import_module("coa_common.constants"))
        try:
            assert mod.RESOURCE_PREFIX == "scl-dev"
            assert mod.BRAND == "coa"
            assert mod.GRAPH_BASE_URI == "http://coa.amazon.com"
            assert mod.VOCAB_URI == "http://coa.amazon.com/vocab/coa#"
            assert mod.URN_PREFIX == "coa"
            assert mod.EVENT_SOURCE_PREFIX == "coa"
            assert mod.DZ_TYPE_PREFIX == "Coa"
        finally:
            monkeypatch.undo()
            importlib.reload(mod)

    def test_graph_base_uri_override_flows_into_vocab_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GRAPH_BASE_URI", "http://example.internal")
        monkeypatch.delenv("VOCAB_URI", raising=False)

        mod = importlib.reload(importlib.import_module("coa_common.constants"))
        try:
            assert mod.VOCAB_URI == "http://example.internal/vocab/coa#"
        finally:
            monkeypatch.undo()
            importlib.reload(mod)
