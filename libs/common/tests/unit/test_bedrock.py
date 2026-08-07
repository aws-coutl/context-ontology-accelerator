# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the common BedrockClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from coa_common.bedrock import (
    BedrockClient,
    BedrockInvocationResult,
    GuardrailBlockedError,
    InputTooLargeError,
)

pytestmark = pytest.mark.unit


class TestBedrockClient:
    @patch("coa_common.bedrock.boto3")
    def test_invoke_parses_json_response(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '{"key": "value"}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1")
        result = client.invoke("system", "user")

        assert isinstance(result, BedrockInvocationResult)
        assert result.result == {"key": "value"}
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.latency_ms > 0
        mock_client.converse.assert_called_once_with(
            modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": [{"text": "user"}]}],
            system=[{"text": "system"}],
            inferenceConfig={"maxTokens": 4096},
        )

    @patch("coa_common.bedrock.boto3")
    def test_invoke_strips_markdown_fences(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '```json\n{"key": "value"}\n```'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1")
        result = client.invoke("system", "user")

        assert result.result == {"key": "value"}

    @patch("coa_common.bedrock.boto3")
    def test_invoke_raises_on_invalid_json(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "not json"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1")
        with pytest.raises(ValueError, match="Invalid Bedrock response format"):
            client.invoke("system", "user")

    @patch("coa_common.bedrock.boto3")
    def test_invoke_raises_on_empty_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "   "}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1")
        with pytest.raises(ValueError, match="Bedrock returned empty response"):
            client.invoke("system", "user")

    @patch("coa_common.bedrock.boto3")
    def test_invoke_raises_on_client_error(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        error_response = {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}}
        mock_client.exceptions.ClientError = type("ClientError", (Exception,), {})
        mock_client.converse.side_effect = mock_client.exceptions.ClientError(error_response, "Converse")

        client = BedrockClient(region="us-east-1")
        with pytest.raises(mock_client.exceptions.ClientError):
            client.invoke("system", "user")

    @patch("coa_common.bedrock.boto3")
    def test_custom_model_id(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '{"ok": true}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-west-2", model_id="custom-model")
        client.invoke("sys", "usr")

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["modelId"] == "custom-model"

    @patch("coa_common.bedrock.boto3")
    def test_custom_max_tokens(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '{"result": "ok"}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1")
        client.invoke("sys", "usr", max_tokens=8192)

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["inferenceConfig"] == {"maxTokens": 8192}

    @patch("coa_common.bedrock.boto3")
    def test_invoke_rejects_oversized_input(self, mock_boto3):
        """Prompts exceeding max_input_chars are rejected before any Bedrock call."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        client = BedrockClient(region="us-east-1", max_input_chars=100)
        with pytest.raises(InputTooLargeError):
            client.invoke("system", "x" * 200)

        mock_client.converse.assert_not_called()


class TestGuardrailIntegration:
    @patch("coa_common.bedrock.boto3")
    def test_guardrail_config_passed_when_id_set(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '{"ok": true}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1", guardrail_id="gr-123")
        client.invoke("sys", "usr")

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["guardrailConfig"] == {
            "guardrailIdentifier": "gr-123",
            "guardrailVersion": "DRAFT",
            "trace": "enabled",
        }

    @patch("coa_common.bedrock.boto3")
    def test_guardrail_config_not_passed_when_id_absent(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '{"ok": true}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1")
        client.invoke("sys", "usr")

        call_kwargs = mock_client.converse.call_args[1]
        assert "guardrailConfig" not in call_kwargs

    @patch("coa_common.bedrock.boto3")
    def test_guardrail_blocked_raises_error(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "I cannot help with that."}]}},
            "stopReason": "guardrail_intervened",
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "trace": {
                "guardrail": {
                    "outputAssessments": {
                        "gr-123": [
                            {"contentPolicy": {"filters": [{"action": "BLOCKED", "type": "VIOLENCE"}]}},
                        ]
                    }
                }
            },
        }

        client = BedrockClient(region="us-east-1", guardrail_id="gr-123")
        with pytest.raises(GuardrailBlockedError, match="gr-123"):
            client.invoke("sys", "usr")

    @patch("coa_common.bedrock.boto3")
    def test_guardrail_blocked_when_no_trace(self, mock_boto3):
        """Conservatively treat as blocked when trace is absent."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "blocked"}]}},
            "stopReason": "guardrail_intervened",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1", guardrail_id="gr-123")
        with pytest.raises(GuardrailBlockedError):
            client.invoke("sys", "usr")

    @patch("coa_common.bedrock.boto3")
    def test_guardrail_anonymize_does_not_raise(self, mock_boto3):
        """PII ANONYMIZE is not a block — response is usable with masked values."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '{"description": "Contact {EMAIL} for info"}'}]}},
            "stopReason": "guardrail_intervened",
            "usage": {"inputTokens": 10, "outputTokens": 5},
            "trace": {
                "guardrail": {
                    "outputAssessments": {
                        "gr-123": [
                            {
                                "sensitiveInformationPolicy": {
                                    "piiEntities": [{"action": "ANONYMIZED", "type": "EMAIL"}]
                                }
                            },
                        ]
                    }
                }
            },
        }

        client = BedrockClient(region="us-east-1", guardrail_id="gr-123")
        result = client.invoke("sys", "usr")
        assert result.result == {"description": "Contact {EMAIL} for info"}

    @patch("coa_common.bedrock.boto3")
    def test_custom_guardrail_version(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        mock_client.converse.return_value = {
            "output": {"message": {"content": [{"text": '{"ok": true}'}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        client = BedrockClient(region="us-east-1", guardrail_id="gr-456", guardrail_version="3")
        client.invoke("sys", "usr")

        call_kwargs = mock_client.converse.call_args[1]
        assert call_kwargs["guardrailConfig"]["guardrailVersion"] == "3"
