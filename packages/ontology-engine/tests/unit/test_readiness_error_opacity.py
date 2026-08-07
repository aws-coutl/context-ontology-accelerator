# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security tests for ``/readiness`` — CWE-209 stack-trace / detail exposure.

``readiness()`` probes Neptune, OpenSearch, and two Bedrock model paths. When a
probe raises, the exception text carries infrastructure internals: cluster
endpoints, role ARNs, botocore request ids, and occasionally credential
material embedded in a URL. This endpoint is reachable by any caller that can
reach the service, so the response must report a bare status and leave the
detail in the server-side log.

Run with: uv run pytest tests/unit/test_readiness_error_opacity.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

# A probe failure whose text is deliberately loaded with the kinds of internals
# that must never reach the HTTP response.
LEAKY_MESSAGE = (
    "Could not connect to https://coa-neptune-cluster.cluster-abc123."
    "us-east-1.neptune.amazonaws.com:8182 using role "
    "arn:aws:iam::123456789012:role/coa-ontology-engine-task "
    "(RequestId: 4b1f9c2e-77aa-4c31-9f0e-8d2a3e5b7c11)"
)

_SECRETS = [
    "coa-neptune-cluster",
    "neptune.amazonaws.com",
    "arn:aws:iam::123456789012",
    "123456789012",
    "RequestId",
    "4b1f9c2e",
    "8182",
    # Not exposed by the pre-fix code either (it formatted str(e), not a
    # traceback) — kept as a guard against a future switch to format_exc().
    "Traceback",
]


class _Boom:
    """Stand-in store whose health check always raises the leaky error."""

    def health_check(self):
        raise RuntimeError(LEAKY_MESSAGE)


@pytest.fixture()
def readiness_with_failing_probes(monkeypatch):
    """Import ``readiness`` and point every probe at a failing dependency."""
    from coa_ontology import main as main_mod

    monkeypatch.setattr(main_mod.app.state, "graph_store", _Boom(), raising=False)
    monkeypatch.setattr(main_mod.app.state, "vector_store", _Boom(), raising=False)
    monkeypatch.setattr(
        main_mod.app.state,
        "config",
        {"bedrock_embed_dimensions": 1024},
        raising=False,
    )
    return main_mod


class TestReadinessDoesNotLeakProbeDetail:
    def test_all_four_probes_report_bare_error(self, readiness_with_failing_probes):
        main_mod = readiness_with_failing_probes
        with (
            patch.object(main_mod, "_make_bedrock_client", side_effect=RuntimeError(LEAKY_MESSAGE)),
            patch.object(main_mod, "_make_description_llm_client", side_effect=RuntimeError(LEAKY_MESSAGE)),
        ):
            body = main_mod.readiness()

        assert body["status"] == "degraded"
        assert body["graph_store"] == "error"
        assert body["vector_store"] == "error"
        assert body["bedrock_embeddings"] == "error"
        assert body["description_llm"] == "error"

    @pytest.mark.parametrize("secret", _SECRETS)
    def test_no_infrastructure_detail_appears_anywhere_in_the_response(self, secret, readiness_with_failing_probes):
        main_mod = readiness_with_failing_probes
        with (
            patch.object(main_mod, "_make_bedrock_client", side_effect=RuntimeError(LEAKY_MESSAGE)),
            patch.object(main_mod, "_make_description_llm_client", side_effect=RuntimeError(LEAKY_MESSAGE)),
        ):
            body = main_mod.readiness()

        assert secret not in str(body)

    def test_probe_detail_is_still_logged_server_side(self, readiness_with_failing_probes, caplog):
        """The detail must remain diagnosable in CloudWatch, just not in the body."""
        main_mod = readiness_with_failing_probes
        # ``main.py`` calls ``logging.basicConfig(force=True)`` at import, which
        # strips existing root handlers. Attach caplog's handler directly to the
        # module logger so this test does not depend on whether that import
        # happened before or after pytest installed its capture handler.
        main_mod.logger.addHandler(caplog.handler)
        try:
            self._assert_probe_detail_logged(main_mod, caplog)
        finally:
            main_mod.logger.removeHandler(caplog.handler)

    @staticmethod
    def _assert_probe_detail_logged(main_mod, caplog):
        with (
            caplog.at_level("WARNING", logger="coa_ontology.main"),
            patch.object(main_mod, "_make_bedrock_client", side_effect=RuntimeError(LEAKY_MESSAGE)),
            patch.object(main_mod, "_make_description_llm_client", side_effect=RuntimeError(LEAKY_MESSAGE)),
        ):
            main_mod.readiness()

        assert "readiness_graph_store_probe_failed" in caplog.text
        assert "readiness_vector_store_probe_failed" in caplog.text
        assert "readiness_bedrock_probe_failed" in caplog.text
        assert "readiness_description_llm_probe_failed" in caplog.text
        # exc_info=True means the operator still gets the real cause.
        assert LEAKY_MESSAGE in caplog.text


class TestReadinessHealthyPath:
    def test_all_probes_ok_reports_ok(self, monkeypatch):
        from coa_ontology import main as main_mod

        class _Ok:
            def health_check(self):
                return {"status": "ok"}

        class _Embed:
            def embed_text(self, _text):
                return [0.0] * 1024

        class _Llm:
            def generate(self, prompt, max_tokens):  # noqa: ARG002
                return "pong"

        monkeypatch.setattr(main_mod.app.state, "graph_store", _Ok(), raising=False)
        monkeypatch.setattr(main_mod.app.state, "vector_store", _Ok(), raising=False)
        monkeypatch.setattr(main_mod.app.state, "config", {"bedrock_embed_dimensions": 1024}, raising=False)

        with (
            patch.object(main_mod, "_make_bedrock_client", return_value=_Embed()),
            patch.object(main_mod, "_make_description_llm_client", return_value=_Llm()),
        ):
            body = main_mod.readiness()

        assert body["status"] == "ok"
        assert body["bedrock_embeddings"] == "ok"
        assert body["description_llm"] == "ok"
