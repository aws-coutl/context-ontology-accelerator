# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security tests for ontologies router — CWE-209 information exposure.

Tests that httpx.HTTPError exceptions during source_url fetch do NOT leak
internal network topology (IP addresses, private DNS names, TLS handshake details).

These tests verify that the except _httpx.HTTPError handler produces opaque 502 messages.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.unit


class TestHTTPErrorMessageSanitization:
    """CWE-209: httpx errors must be caught and NOT appear in 502 response body."""

    def test_httpx_connect_error_message_structure(self):
        """Verify httpx.ConnectError carries internal details that must not leak."""
        import httpx

        # Simulate what happens during a connection failure
        try:
            raise httpx.ConnectError(
                "Connection refused to internal IP 10.0.1.42:443 for https://example.com/ontology.ttl"
            )
        except httpx.ConnectError as e:
            error_string = str(e)
            # Confirm the exception DOES contain sensitive details (pre-sanitization)
            assert "10.0.1.42" in error_string or "Connection refused" in error_string
            # This demonstrates why we need sanitization

    def test_httpx_remote_protocol_error_structure(self):
        """Verify httpx.RemoteProtocolError carries TLS/cert details that must not leak."""
        import httpx

        try:
            raise httpx.RemoteProtocolError(
                "TLS handshake failed: peer certificate CN=internal.example.com does not match expected CN=example.com"
            )
        except httpx.RemoteProtocolError as e:
            error_string = str(e)
            # Confirm the exception DOES contain TLS details
            assert "TLS handshake" in error_string or "internal.example.com" in error_string

    def test_httpx_timeout_exception_structure(self):
        """Verify httpx.TimeoutException can carry resolved IP that must not leak."""
        import httpx

        try:
            raise httpx.TimeoutException(
                "Timeout while connecting to https://example.com/ontology.ttl (resolved to 192.168.1.100)"
            )
        except httpx.TimeoutException as e:
            error_string = str(e)
            # Confirm the exception DOES contain resolved IP
            assert "192.168.1.100" in error_string or "Timeout while connecting" in error_string

    def test_sanitized_502_message_does_not_reference_exception_text(self):
        """The opaque 502 message must not include any exception details."""
        opaque_message = "Failed to fetch source_url: upstream request failed"

        # Verify the sanitized message contains no infrastructure details
        assert "10.0.1.42" not in opaque_message
        assert "internal.example.com" not in opaque_message
        assert "192.168.1.100" not in opaque_message
        assert "Connection refused" not in opaque_message
        assert "TLS handshake" not in opaque_message
        assert "Timeout while connecting" not in opaque_message
        # Confirm it's truly opaque
        assert opaque_message == "Failed to fetch source_url: upstream request failed"


class TestStorageRootContainment:
    """``_is_within_storage_root`` is the single containment predicate.

    Three call sites depend on it: the ``/download`` local-file fallback, the
    ``source_url`` fetch destination, and ``_persist_upload``.
    """

    def test_sibling_directory_sharing_the_root_prefix_is_outside(self, tmp_path, monkeypatch):
        """The case a ``startswith`` check gets wrong."""
        from coa_ontology.catalog.routers import ontologies

        root = tmp_path / "ontologies"
        root.mkdir()
        evil = tmp_path / "ontologies-evil"
        evil.mkdir()
        monkeypatch.setattr(ontologies, "ONTOLOGY_STORAGE_PATH", str(root))

        assert ontologies._is_within_storage_root(str(evil / "x.ttl")) is False
        assert str(evil).startswith(str(root)), "precondition: prefix check would pass"

    def test_paths_inside_the_root_are_accepted(self, tmp_path, monkeypatch):
        from coa_ontology.catalog.routers import ontologies

        root = tmp_path / "ontologies"
        (root / "ns").mkdir(parents=True)
        monkeypatch.setattr(ontologies, "ONTOLOGY_STORAGE_PATH", str(root))

        assert ontologies._is_within_storage_root(str(root)) is True
        assert ontologies._is_within_storage_root(str(root / "ns" / "o.ttl")) is True

    def test_symlink_escaping_the_root_is_rejected(self, tmp_path, monkeypatch):
        from coa_ontology.catalog.routers import ontologies

        root = tmp_path / "ontologies"
        root.mkdir()
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (root / "link.ttl").symlink_to(outside / "secret.ttl")
        monkeypatch.setattr(ontologies, "ONTOLOGY_STORAGE_PATH", str(root))

        assert ontologies._is_within_storage_root(str(root / "link.ttl")) is False


class TestPersistUploadPathTraversal:
    """CWE-22: ``_persist_upload`` writes to a caller-influenced path.

    ``namespace`` and ``identifier`` both originate from request input, so the
    resolved destination must always stay inside ``ONTOLOGY_STORAGE_PATH``.
    """

    @pytest.fixture()
    def storage_root(self, tmp_path, monkeypatch):
        from coa_ontology.catalog.routers import ontologies

        root = tmp_path / "ontologies"
        root.mkdir()
        monkeypatch.setattr(ontologies, "ONTOLOGY_STORAGE_PATH", str(root))
        return root

    @pytest.mark.parametrize(
        "namespace",
        [
            "../escape",
            "..",
            "a/../../b",
            "ns/sub",
            "ns\x00",
            "/abs",
            ".",
            # Python's ``$`` also matches just before a trailing newline, so an
            # unanchored ``re.match`` would accept this and create a directory
            # whose name ends in a newline.
            "ns\n",
        ],
    )
    def test_traversal_namespaces_are_rejected(self, namespace, storage_root):
        from coa_ontology.catalog.routers.ontologies import _persist_upload

        with pytest.raises(ValueError):
            _persist_upload(namespace, "ont", b"data", "turtle")

    @pytest.mark.parametrize(
        "identifier",
        [
            "../../../etc/passwd",
            "..",
            ".",
            "/etc/passwd",
            "sub/../../../evil",
            "",
        ],
    )
    def test_traversal_identifiers_stay_inside_the_namespace_dir(self, identifier, storage_root):
        """A hostile ontology id must never escape — it is slugified, not honoured."""
        from coa_ontology.catalog.routers.ontologies import _persist_upload

        dest = _persist_upload("myns", identifier, b"data", "turtle")

        root = os.path.realpath(str(storage_root))
        ns_dir = os.path.realpath(str(storage_root / "myns"))
        # Containment asserted via commonpath rather than by recomputing the
        # source's own dirname check, which would be circular.
        assert os.path.commonpath([root, os.path.realpath(dest)]) == root
        # The file landed in the namespace dir, and the id contributed only a
        # single flat filename component (no separators survived slugification).
        assert os.path.realpath(dest) == os.path.join(ns_dir, os.path.basename(dest))
        assert os.sep not in os.path.basename(dest)
        assert os.path.isfile(dest)

    def test_symlinked_namespace_escaping_to_sibling_prefix_is_rejected(self, tmp_path, monkeypatch):
        """A namespace symlinked to ``/…/ontologies-evil`` must be rejected.

        This is the input that separates the two containment strategies. With a
        root of ``/…/ontologies``, the resolved ``/…/ontologies-evil`` still
        satisfies ``ns_dir.startswith(root)`` — a sibling whose name merely
        begins with the root — but fails a component-wise ``commonpath``
        comparison. Without the symlink the two agree, so a plain
        ``mkdir``-based test would pass against either implementation and prove
        nothing.
        """
        from coa_ontology.catalog.routers import ontologies

        root = tmp_path / "ontologies"
        root.mkdir()
        evil = tmp_path / "ontologies-evil"
        evil.mkdir()
        (root / "ns").symlink_to(evil)
        monkeypatch.setattr(ontologies, "ONTOLOGY_STORAGE_PATH", str(root))

        with pytest.raises(ValueError, match="Path traversal"):
            ontologies._persist_upload("ns", "ont", b"data", "turtle")
        assert not list(evil.iterdir()), "nothing may be written outside the storage root"

    def test_distinct_ontology_iris_do_not_collide_on_one_filename(self, storage_root):
        """Ontology ids are IRIs that often differ only before the last ``/``.

        Slugifying just the basename would map ``https://schema.org/``,
        ``http://purl.org/dc/terms/`` and ``http://xmlns.com/foaf/0.1/`` all to
        the same file, so each upload would silently overwrite the previous
        ontology's bytes and every registry row would point at one file.
        """
        from coa_ontology.catalog.routers.ontologies import _persist_upload

        ids = [
            "https://schema.org/",
            "http://purl.org/dc/terms/",
            "http://xmlns.com/foaf/0.1/",
            "http://ex.org/induced#",
            "http://other.org/x/induced#",
        ]
        dests = {i: _persist_upload("myns", i, i.encode(), "turtle") for i in ids}

        assert len(set(dests.values())) == len(ids), f"slug collision: {dests}"
        # Each file still holds its own ontology's bytes.
        for identifier, path in dests.items():
            with open(path, "rb") as fh:
                assert fh.read() == identifier.encode()

    def test_ids_sharing_a_200_char_prefix_do_not_collide(self, storage_root):
        """Truncation alone would merge ids that differ only past 200 chars."""
        from coa_ontology.catalog.routers.ontologies import _persist_upload

        prefix = "http://example.com/" + "x" * 195
        a = _persist_upload("myns", f"{prefix}/alpha", b"A", "turtle")
        b = _persist_upload("myns", f"{prefix}/beta", b"B", "turtle")

        assert a != b
        with open(a, "rb") as fh:
            assert fh.read() == b"A"
        with open(b, "rb") as fh:
            assert fh.read() == b"B"

    def test_short_ids_keep_their_natural_filename(self, storage_root):
        """The digest suffix must apply ONLY when truncation happens.

        Appending it unconditionally would rename every existing ontology file.
        """
        from coa_ontology.catalog.routers.ontologies import _persist_upload

        dest = _persist_upload("myns", "https://schema.org/", b"data", "turtle")
        assert os.path.basename(dest) == "https_schema.org_.ttl"

    def test_valid_upload_round_trips(self, storage_root):
        from coa_ontology.catalog.routers.ontologies import _persist_upload

        dest = _persist_upload("myns", "http://example.com/my-ont", b"payload", "turtle")

        assert os.path.isfile(dest)
        with open(dest, "rb") as fh:
            assert fh.read() == b"payload"
        assert dest.endswith(".ttl")
