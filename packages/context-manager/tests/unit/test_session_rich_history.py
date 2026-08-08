# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for session.py rich history features.

Tests ULID session ID generation, message envelope building/parsing,
and the get_rich_history method.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from coa_serve.session import (
    SessionManager,
    _build_message_envelope,
    _parse_assistant_content,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# SessionManager.generate_session_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateSessionId:
    def test_returns_non_empty_string(self):
        sid = SessionManager.generate_session_id()
        assert sid
        assert isinstance(sid, str)

    def test_returns_unique_ids(self):
        sid1 = SessionManager.generate_session_id()
        sid2 = SessionManager.generate_session_id()
        assert sid1 != sid2

    def test_returns_valid_ulid_length(self):
        sid = SessionManager.generate_session_id()
        assert len(sid) == 26  # ULID string is always 26 chars


# ---------------------------------------------------------------------------
# _build_message_envelope
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildMessageEnvelope:
    def test_basic_envelope(self):
        result = _build_message_envelope("Hello world", result_metadata={"tier": 2})
        obj = json.loads(result)
        assert obj["text"] == "Hello world"
        assert obj["tier"] == 2

    def test_full_envelope(self):
        result = _build_message_envelope(
            answer="Revenue is $4.2M",
            result_metadata={
                "tier": 1,
                "confidence": {"score": 0.95, "rationale": "Exact metric match"},
                "trace": [{"step": "metric_match", "status": "success", "durationMs": 50}],
                "queryUsed": "SELECT sum(revenue) FROM facts",
            },
        )
        obj = json.loads(result)
        assert obj["text"] == "Revenue is $4.2M"
        assert obj["tier"] == 1
        assert obj["confidence"]["score"] == 0.95
        assert obj["confidence"]["rationale"] == "Exact metric match"
        assert len(obj["trace"]) == 1
        assert obj["queryUsed"] == "SELECT sum(revenue) FROM facts"

    def test_truncates_long_trace_details(self):
        # Build trace with huge detail fields
        trace = [{"step": f"step_{i}", "status": "success", "durationMs": 100, "detail": "x" * 5000} for i in range(5)]
        result = _build_message_envelope("answer", result_metadata={"tier": 2, "trace": trace})
        assert len(result) <= 10_000
        obj = json.loads(result)
        # All steps preserved (only details truncated, not steps removed)
        assert len(obj["trace"]) == 5
        # At least some details were stripped to fit within limit
        details_remaining = sum(1 for s in obj["trace"] if "detail" in s)
        assert details_remaining < 5

    def test_preserves_step_status_duration_after_truncation(self):
        trace = [{"step": "big_step", "status": "success", "durationMs": 200, "detail": "y" * 9000}]
        result = _build_message_envelope("ans", result_metadata={"tier": 3, "trace": trace})
        obj = json.loads(result)
        assert obj["trace"][0]["step"] == "big_step"
        assert obj["trace"][0]["status"] == "success"
        assert obj["trace"][0]["durationMs"] == 200

    def test_no_metadata_produces_text_only(self):
        result = _build_message_envelope("just text")
        obj = json.loads(result)
        assert obj == {"text": "just text"}

    def test_new_fields_pass_through(self):
        """Any new field on QueryResult flows through without code changes."""
        result = _build_message_envelope(
            "ans",
            result_metadata={
                "tier": 2,
                "confidence": {"score": 0.8, "rationale": "good"},
                "sparqlGenerated": "SELECT ?x WHERE { ?x a :Foo }",
                "guardrailBlocked": False,
            },
        )
        obj = json.loads(result)
        assert obj["sparqlGenerated"] == "SELECT ?x WHERE { ?x a :Foo }"
        assert obj["guardrailBlocked"] is False

    def test_result_rows_included_in_envelope(self):
        """resultRows are preserved in the envelope for UI table restore."""
        rows = [
            {"claim_id": "CL-44201", "status": "Open", "amount": "$14,200"},
            {"claim_id": "CL-44318", "status": "Open", "amount": "$6,800"},
        ]
        result = _build_message_envelope(
            "",
            result_metadata={
                "tier": 1,
                "confidence": {"score": 0.92, "rationale": "Exact match"},
                "resultRows": rows,
                "queryUsed": "SELECT * FROM claims WHERE status = 'Open'",
                "trace": [{"step": "metric_match", "status": "success", "durationMs": 50}],
            },
        )
        obj = json.loads(result)
        assert obj["text"] == ""
        assert obj["resultRows"] == rows
        assert len(obj["resultRows"]) == 2
        assert obj["resultRows"][0]["claim_id"] == "CL-44201"


# ---------------------------------------------------------------------------
# _parse_assistant_content
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseAssistantContent:
    def test_plain_string(self):
        text, meta = _parse_assistant_content("Hello world")
        assert text == "Hello world"
        assert meta is None

    def test_envelope_with_metadata(self):
        envelope = json.dumps({"text": "The answer", "tier": 2, "confidence": {"score": 0.9, "rationale": "ok"}})
        text, meta = _parse_assistant_content(envelope)
        assert text == "The answer"
        assert meta["tier"] == 2
        assert meta["confidence"]["score"] == 0.9

    def test_envelope_text_only(self):
        envelope = json.dumps({"text": "Just text"})
        text, meta = _parse_assistant_content(envelope)
        assert text == "Just text"
        assert meta is None  # no other fields -> None

    def test_malformed_json(self):
        text, meta = _parse_assistant_content("{broken json")
        assert text == "{broken json"
        assert meta is None

    def test_json_without_text_key(self):
        # Valid JSON but not our envelope format
        text, meta = _parse_assistant_content('{"foo": "bar"}')
        assert text == '{"foo": "bar"}'
        assert meta is None

    def test_non_object_json(self):
        text, meta = _parse_assistant_content("[1, 2, 3]")
        assert text == "[1, 2, 3]"
        assert meta is None


# ---------------------------------------------------------------------------
# SessionManager.create_or_resume (envelope handling)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateOrResumeEnvelope:
    @pytest.mark.asyncio
    async def test_extracts_text_from_envelope(self):
        envelope = json.dumps({"text": "Answer", "tier": 2, "trace": []})
        mock_turns = [
            [
                {"role": "USER", "content": {"text": "What?"}},
                {"role": "ASSISTANT", "content": {"text": envelope}},
            ]
        ]
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.get_last_k_turns.return_value = mock_turns
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            sid, history = await mgr.create_or_resume("user1", "ns1")

            # Should extract only text for orchestrator
            assert history[1]["content"] == "Answer"
            assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_plain_string_passes_through(self):
        mock_turns = [
            [
                {"role": "USER", "content": {"text": "Hi"}},
                {"role": "ASSISTANT", "content": {"text": "Hello there"}},
            ]
        ]
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.get_last_k_turns.return_value = mock_turns
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            _, history = await mgr.create_or_resume("user1", "ns1")

            assert history[1]["content"] == "Hello there"


# ---------------------------------------------------------------------------
# SessionManager.get_rich_history
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetRichHistory:
    @pytest.mark.asyncio
    async def test_parses_envelope_metadata(self):
        envelope = json.dumps(
            {
                "text": "Revenue is $4.2M",
                "tier": 1,
                "confidence": {"score": 0.92, "rationale": "Exact match"},
                "trace": [{"step": "metric_match", "status": "success", "durationMs": 50}],
                "queryUsed": "SELECT sum(revenue)",
            }
        )
        mock_turns = [
            [
                {"role": "USER", "content": {"text": "What is revenue?"}},
                {"role": "ASSISTANT", "content": {"text": envelope}},
            ]
        ]
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.get_last_k_turns.return_value = mock_turns
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            history = await mgr.get_rich_history("user1", "session-abc")

            assert len(history) == 2
            user_msg = history[0]
            assistant_msg = history[1]

            assert user_msg["role"] == "user"
            assert user_msg["content"] == "What is revenue?"
            assert "metadata" not in user_msg

            assert assistant_msg["role"] == "assistant"
            assert assistant_msg["content"] == "Revenue is $4.2M"
            assert assistant_msg["metadata"]["tier"] == 1
            assert assistant_msg["metadata"]["confidence"]["score"] == 0.92
            assert len(assistant_msg["metadata"]["trace"]) == 1

    @pytest.mark.asyncio
    async def test_legacy_messages_no_metadata(self):
        mock_turns = [
            [
                {"role": "USER", "content": {"text": "Hi"}},
                {"role": "ASSISTANT", "content": {"text": "Hello!"}},
            ]
        ]
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.get_last_k_turns.return_value = mock_turns
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            history = await mgr.get_rich_history("user1", "session-abc")

            assert history[1]["content"] == "Hello!"
            assert "metadata" not in history[1]

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back(self):
        mock_turns = [
            [
                {"role": "USER", "content": {"text": "Q"}},
                {"role": "ASSISTANT", "content": {"text": "{bad json here"}},
            ]
        ]
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.get_last_k_turns.return_value = mock_turns
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            history = await mgr.get_rich_history("user1", "session-abc")

            assert history[1]["content"] == "{bad json here"
            assert "metadata" not in history[1]

    @pytest.mark.asyncio
    async def test_result_rows_preserved_in_rich_history(self):
        """resultRows stored in envelope are available in metadata on restore."""
        rows = [{"total_accounts": "47"}]
        envelope = json.dumps(
            {
                "text": "",
                "tier": 1,
                "confidence": {"score": 0.95, "rationale": "Metric match"},
                "resultRows": rows,
                "queryUsed": "SELECT count(*) FROM accounts",
                "trace": [],
            }
        )
        mock_turns = [
            [
                {"role": "USER", "content": {"text": "total accounts"}},
                {"role": "ASSISTANT", "content": {"text": envelope}},
            ]
        ]
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.get_last_k_turns.return_value = mock_turns
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            history = await mgr.get_rich_history("user1", "session-abc")

            assistant_msg = history[1]
            assert assistant_msg["content"] == ""
            assert assistant_msg["metadata"]["resultRows"] == rows
            assert assistant_msg["metadata"]["queryUsed"] == "SELECT count(*) FROM accounts"


# ---------------------------------------------------------------------------
# SessionManager.append_turn (with metadata)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAppendTurnWithMetadata:
    @pytest.mark.asyncio
    async def test_builds_envelope_when_metadata_provided(self):
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.add_turns = MagicMock()
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            await mgr.append_turn(
                user_id="user1",
                session_id="sess1",
                query="What is revenue?",
                answer_summary="$4.2M",
                result_metadata={"tier": 1, "confidence": {"score": 0.9, "rationale": "ok"}},
            )

            # Verify add_turns was called
            mock_session.add_turns.assert_called_once()
            call_args = mock_session.add_turns.call_args
            messages = call_args[0][0]
            assert messages[0].text == "What is revenue?"
            # Assistant message should be an envelope JSON
            assistant_text = messages[1].text
            obj = json.loads(assistant_text)
            assert obj["text"] == "$4.2M"
            assert obj["tier"] == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_plain_text_on_error(self):
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.add_turns = MagicMock()
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            # Pass un-serializable metadata to trigger fallback
            await mgr.append_turn(
                user_id="user1",
                session_id="sess1",
                query="Q",
                answer_summary="Plain answer",
                result_metadata={"tier": object()},  # not serializable
            )

            mock_session.add_turns.assert_called_once()
            messages = mock_session.add_turns.call_args[0][0]
            # Should fall back to plain text
            assert messages[1].text == "Plain answer"

    @pytest.mark.asyncio
    async def test_no_metadata_stores_plain_text(self):
        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.add_turns = MagicMock()
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            await mgr.append_turn(
                user_id="user1",
                session_id="sess1",
                query="Hi",
                answer_summary="Hello!",
            )

            messages = mock_session.add_turns.call_args[0][0]
            assert messages[1].text == "Hello!"


# ---------------------------------------------------------------------------
# SessionMetadataStore — namespace-scoped operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
class TestSessionMetadataNamespaceScoping:
    """Tests for namespace-aware metadata store methods."""

    @pytest.fixture
    def mock_table(self):
        with patch("coa_serve.session_metadata.boto3") as mock_boto:
            mock_resource = MagicMock()
            mock_table = MagicMock()
            mock_resource.Table.return_value = mock_table
            mock_boto.resource.return_value = mock_resource
            # Mock the exceptions for contextlib.suppress
            mock_resource.meta.client.exceptions.ConditionalCheckFailedException = type(
                "ConditionalCheckFailedException", (Exception,), {}
            )
            yield mock_table

    async def test_create_includes_namespace_and_composite_key(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        result = await store.create("user-1", namespace_id="ns-birds")

        assert result.namespace_id == "ns-birds"
        assert result.session_id  # ULID generated

        # Verify the DDB item was written with composite key
        put_call = mock_table.put_item.call_args
        item = put_call[1]["Item"]
        assert item["namespaceId"] == "ns-birds"
        assert item["nsLastActiveAt"].startswith("ns-birds#")

    async def test_create_with_title(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        result = await store.create("user-1", namespace_id="ns-birds", title="What are birds")

        assert result.title == "What are birds"
        item = mock_table.put_item.call_args[1]["Item"]
        assert item["title"] == "What are birds"

    async def test_create_truncates_long_title(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        long_title = "x" * 200
        result = await store.create("user-1", namespace_id="ns", title=long_title)

        assert len(result.title) == 100
        item = mock_table.put_item.call_args[1]["Item"]
        assert len(item["title"]) == 100

    async def test_get_recent_for_namespace_queries_gsi(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        mock_table.query.return_value = {
            "Items": [
                {
                    "userId": "user-1",
                    "sessionId": "01ABC",
                    "namespaceId": "ns-birds",
                    "nsLastActiveAt": "ns-birds#2026-06-24T10:00:00Z",
                    "createdAt": "2026-06-24T09:00:00Z",
                    "lastActiveAt": "2026-06-24T10:00:00Z",
                    "status": "active",
                    "messageCount": 5,
                    "ttl": 9999999,
                    "title": "Bird questions",
                }
            ]
        }

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        result = await store.get_recent_for_namespace("user-1", "ns-birds")

        assert result is not None
        assert result.session_id == "01ABC"
        assert result.title == "Bird questions"

        # Verify GSI query used begins_with
        query_kwargs = mock_table.query.call_args[1]
        assert query_kwargs["IndexName"] == "userId-nsLastActiveAt-index"
        assert ":nsp" in query_kwargs["ExpressionAttributeValues"]
        assert query_kwargs["ExpressionAttributeValues"][":nsp"] == "ns-birds#"

    async def test_get_recent_for_namespace_returns_none_when_empty(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        mock_table.query.return_value = {"Items": []}

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        result = await store.get_recent_for_namespace("user-1", "ns-empty")

        assert result is None

    async def test_update_activity_maintains_composite_key(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        await store.update_activity("user-1", "session-abc", namespace_id="ns-birds")

        update_kwargs = mock_table.update_item.call_args[1]
        assert ":nsc" in update_kwargs["ExpressionAttributeValues"]
        assert update_kwargs["ExpressionAttributeValues"][":nsc"].startswith("ns-birds#")

    async def test_list_for_namespace_paginated(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        mock_table.query.return_value = {
            "Items": [
                {
                    "userId": "user-1",
                    "sessionId": "01ABC",
                    "namespaceId": "ns-birds",
                    "nsLastActiveAt": "ns-birds#2026-06-24T10:00:00Z",
                    "createdAt": "2026-06-24T09:00:00Z",
                    "lastActiveAt": "2026-06-24T10:00:00Z",
                    "status": "active",
                    "messageCount": 5,
                    "ttl": 9999999,
                    "title": "Bird questions",
                }
            ],
            "LastEvaluatedKey": {"userId": "user-1", "nsLastActiveAt": "ns-birds#2026-06-24T10:00:00Z"},
        }

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        sessions, next_token = await store.list_for_namespace("user-1", "ns-birds", limit=1)

        assert len(sessions) == 1
        assert sessions[0].title == "Bird questions"
        assert next_token is not None

    async def test_list_for_namespace_empty(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        mock_table.query.return_value = {"Items": []}

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        sessions, next_token = await store.list_for_namespace("user-1", "ns-empty")

        assert sessions == []
        assert next_token is None

    async def test_validate_ownership_found(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        mock_table.get_item.return_value = {"Item": {"userId": "user-1"}}

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        result = await store.validate_ownership("user-1", "session-abc")

        assert result is True

    async def test_validate_ownership_not_found(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        mock_table.get_item.return_value = {}

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        result = await store.validate_ownership("user-1", "session-xyz")

        assert result is False

    async def test_delete_session(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        await store.delete("user-1", "session-abc")

        mock_table.delete_item.assert_called_once_with(Key={"userId": "user-1", "sessionId": "session-abc"})

    async def test_get_recent_returns_most_recent(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        mock_table.query.return_value = {
            "Items": [
                {
                    "userId": "user-1",
                    "sessionId": "01DEF",
                    "createdAt": "2026-06-24T08:00:00Z",
                    "lastActiveAt": "2026-06-24T12:00:00Z",
                    "status": "active",
                    "messageCount": 10,
                    "ttl": 9999999,
                }
            ]
        }

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        result = await store.get_recent("user-1")

        assert result is not None
        assert result.session_id == "01DEF"

    async def test_get_recent_returns_none_when_empty(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        mock_table.query.return_value = {"Items": []}

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        result = await store.get_recent("user-1")

        assert result is None

    async def test_update_title_idempotent(self, mock_table):
        from coa_serve.session_metadata import SessionMetadataStore

        store = SessionMetadataStore(table_name="test-table", region="us-east-1")
        await store.update_title("user-1", "session-abc", "My title")

        mock_table.update_item.assert_called_once()
        call_kwargs = mock_table.update_item.call_args[1]
        assert call_kwargs["ExpressionAttributeValues"][":t"] == "My title"


# ---------------------------------------------------------------------------
# SessionManager.get_rich_history_page — pagination edge cases
# ---------------------------------------------------------------------------


class TestGetRichHistoryPage:
    @pytest.mark.asyncio
    async def test_first_page_of_many(self):
        """First page returns last N messages with hasMore=True."""
        from unittest.mock import MagicMock, patch

        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            # 10 turns = 20 messages (user + assistant pairs)
            turns = [
                [
                    {"role": "user", "content": {"text": f"q{i}"}},
                    {"role": "assistant", "content": {"text": f"a{i}"}},
                ]
                for i in range(10)
            ]
            mock_session.get_last_k_turns.return_value = list(reversed(turns))
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            page, total, has_more = await mgr.get_rich_history_page("u1", "s1", limit=5, offset=0)

            assert total == 20
            assert len(page) == 5
            assert has_more is True
            # Last 5 messages should be from turns 8-9
            assert page[-1]["content"] == "a9"

    @pytest.mark.asyncio
    async def test_last_page(self):
        """Offset reaching the beginning returns remaining messages with hasMore=False."""
        from unittest.mock import MagicMock, patch

        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            turns = [
                [
                    {"role": "user", "content": {"text": f"q{i}"}},
                    {"role": "assistant", "content": {"text": f"a{i}"}},
                ]
                for i in range(3)
            ]
            mock_session.get_last_k_turns.return_value = list(reversed(turns))
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            # 6 total messages, offset=5 means skip last 5, take up to 5 from start
            page, total, has_more = await mgr.get_rich_history_page("u1", "s1", limit=5, offset=5)

            assert total == 6
            assert len(page) == 1  # Only 1 message remains
            assert has_more is False
            assert page[0]["content"] == "q0"

    @pytest.mark.asyncio
    async def test_empty_session(self):
        """Empty session returns empty page with hasMore=False."""
        from unittest.mock import MagicMock, patch

        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            mock_session.get_last_k_turns.return_value = []
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            page, total, has_more = await mgr.get_rich_history_page("u1", "s1", limit=5, offset=0)

            assert total == 0
            assert len(page) == 0
            assert has_more is False

    @pytest.mark.asyncio
    async def test_offset_beyond_total(self):
        """Offset larger than total returns empty page."""
        from unittest.mock import MagicMock, patch

        with patch("coa_serve.session.MemorySessionManager") as MockMgr:
            mock_session = MagicMock()
            turns = [
                [
                    {"role": "user", "content": {"text": "q"}},
                    {"role": "assistant", "content": {"text": "a"}},
                ]
            ]
            mock_session.get_last_k_turns.return_value = list(reversed(turns))
            MockMgr.return_value.create_memory_session.return_value = mock_session

            mgr = SessionManager(memory_id="mem-123")
            page, total, has_more = await mgr.get_rich_history_page("u1", "s1", limit=5, offset=100)

            assert total == 2
            assert len(page) == 0
            assert has_more is False
