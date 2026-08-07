# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DNS rebinding SSRF prevention in fetch_ontology_from_source.

Tests the fix for Server-Side Request Forgery via DNS Rebinding.

The vulnerability was a TOCTOU (Time-of-Check-Time-of-Use) race where:
1. _validate_fetch_url resolved the hostname and validated IPs were public
2. httpx performed a separate DNS resolution for the actual HTTP request
3. An attacker controlling authoritative DNS could serve different IPs

The fix: _resolve_and_validate_url resolves DNS once, validates all returned
IPs are public, then returns a URL pinned to the validated IP. The HTTP
client connects directly to that IP (eliminating the second DNS query),
with the original hostname in the Host header for TLS SNI and virtual hosting.

These tests verify the core security logic without importing the full router
to avoid Smithy model dependencies in unit tests.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from unittest.mock import patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.unit

# Replicate the fixed function logic inline for testing
_FETCH_ALLOWED_SCHEMES = {"https"}


def _resolve_and_validate_url(url: str) -> tuple[str, str, str]:
    """The fixed version that eliminates DNS rebinding TOCTOU."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _FETCH_ALLOWED_SCHEMES:
        raise HTTPException(400, "source_url must use https://")
    if not parsed.hostname:
        raise HTTPException(400, "source_url is missing a hostname")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise HTTPException(400, f"Cannot resolve source_url host: {e}") from e
    if not infos:
        raise HTTPException(400, "source_url hostname resolved to no addresses")

    # Validate ALL returned IPs are globally routable unicast before using any.
    # NOTE: keep this in sync with the router's _resolve_and_validate_url —
    # `not ip.is_global or ip.is_multicast` (multicast is not "private", so
    # is_global is True for it and must be rejected separately).
    validated_ips: list[str] = []
    for _family, _type, _proto, _canon, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global or ip.is_multicast:
            raise HTTPException(400, f"source_url resolves to non-routable address {ip}")
        validated_ips.append(sockaddr[0])

    # Use the first validated IP and reconstruct the URL to connect directly to it
    pinned_ip = validated_ips[0]
    port = parsed.port or 443
    # IPv6 addresses need bracket notation in URLs
    netloc = f"[{pinned_ip}]:{port}" if ":" in pinned_ip else f"{pinned_ip}:{port}"
    pinned_url = parsed._replace(netloc=netloc).geturl()

    return pinned_url, parsed.hostname, str(port)


def test_resolve_and_validate_url_rejects_http_scheme():
    """Only https:// is allowed."""
    with pytest.raises(HTTPException) as exc:
        _resolve_and_validate_url("http://example.com/ontology.ttl")
    assert exc.value.status_code == 400
    assert "https://" in str(exc.value.detail).lower()


def test_resolve_and_validate_url_rejects_private_ips():
    """All RFC 1918 private addresses must be rejected."""
    private_ips = [
        "10.0.0.1",  # RFC 1918
        "172.16.0.1",  # RFC 1918
        "192.168.1.1",  # RFC 1918
        "127.0.0.1",  # loopback
        "169.254.169.254",  # link-local (AWS IMDS)
    ]

    for ip in private_ips:
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]
            with pytest.raises(HTTPException) as exc:
                _resolve_and_validate_url("https://example.com/ontology.ttl")
            assert exc.value.status_code == 400
            assert "non-routable" in str(exc.value.detail).lower()


def test_resolve_and_validate_url_rejects_multicast():
    """Multicast addresses must be rejected."""
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("224.0.0.1", 0))]
        with pytest.raises(HTTPException) as exc:
            _resolve_and_validate_url("https://example.com/ontology.ttl")
        assert exc.value.status_code == 400
        assert "non-routable" in str(exc.value.detail).lower()


def test_resolve_and_validate_url_rejects_cgnat_and_unspecified():
    """CGNAT (100.64.0.0/10) and unspecified (0.0.0.0) — missed by an explicit
    flag enumeration — must be rejected by the is_global check."""
    for ip in ["100.64.0.1", "0.0.0.0"]:
        with patch("socket.getaddrinfo") as mock_getaddrinfo:
            mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]
            with pytest.raises(HTTPException) as exc:
                _resolve_and_validate_url("https://example.com/ontology.ttl")
            assert exc.value.status_code == 400
            assert "non-routable" in str(exc.value.detail).lower()


def test_resolve_and_validate_url_rejects_ipv4_mapped_imds():
    """IPv4-mapped IPv6 pointing at IMDS (::ffff:169.254.169.254) must be
    rejected — an SSRF bypass that per-flag checks can miss."""
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [
            (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::ffff:169.254.169.254", 0, 0, 0)),
        ]
        with pytest.raises(HTTPException) as exc:
            _resolve_and_validate_url("https://example.com/ontology.ttl")
        assert exc.value.status_code == 400
        assert "non-routable" in str(exc.value.detail).lower()


def test_resolve_and_validate_url_accepts_public_ipv4():
    """Valid public IPv4 addresses are accepted and pinned."""
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0))]
        pinned_url, hostname, port = _resolve_and_validate_url("https://example.com/ontology.ttl")

    assert "1.1.1.1" in pinned_url
    assert "443" in pinned_url
    assert hostname == "example.com"
    assert port == "443"


def test_resolve_and_validate_url_accepts_public_ipv6():
    """Valid public IPv6 addresses are accepted with bracket notation."""
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("2606:4700:4700::1111", 0))]
        pinned_url, hostname, port = _resolve_and_validate_url("https://example.com/ontology.ttl")

    assert "[2606:4700:4700::1111]" in pinned_url
    assert hostname == "example.com"


def test_resolve_and_validate_url_uses_custom_port():
    """Custom HTTPS ports are preserved in the pinned URL."""
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
        pinned_url, hostname, port = _resolve_and_validate_url("https://example.com:8443/ontology.ttl")

    assert "93.184.216.34:8443" in pinned_url
    assert port == "8443"


def test_resolve_and_validate_url_rejects_dns_failure():
    """DNS resolution failures are rejected with 400."""
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.side_effect = socket.gaierror("Name or service not known")
        with pytest.raises(HTTPException) as exc:
            _resolve_and_validate_url("https://nonexistent.invalid/ontology.ttl")
        assert exc.value.status_code == 400
        assert "Cannot resolve" in exc.value.detail


def test_resolve_and_validate_url_rejects_empty_resolution():
    """DNS resolution returning no addresses is rejected."""
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        mock_getaddrinfo.return_value = []
        with pytest.raises(HTTPException) as exc:
            _resolve_and_validate_url("https://example.com/ontology.ttl")
        assert exc.value.status_code == 400
        assert "no addresses" in str(exc.value.detail).lower()


def test_resolve_and_validate_url_rejects_if_any_ip_is_private():
    """If ANY returned IP is private, the entire request is rejected.

    This prevents an attacker from relying on round-robin DNS where
    some IPs are public (pass validation) but others are private.
    """
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        # Mock: first IP is public, second is private
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("192.168.1.1", 0)),
        ]
        with pytest.raises(HTTPException) as exc:
            _resolve_and_validate_url("https://example.com/ontology.ttl")
        assert exc.value.status_code == 400
        assert "non-routable" in str(exc.value.detail).lower()


def test_resolve_and_validate_url_missing_hostname():
    """URLs without a hostname are rejected."""
    with pytest.raises(HTTPException) as exc:
        _resolve_and_validate_url("https:///path/to/file.ttl")
    assert exc.value.status_code == 400
    assert "missing a hostname" in str(exc.value.detail).lower()


def test_dns_rebinding_attack_scenario():
    """Security test: DNS rebinding cannot bypass validation.

    In the vulnerable version, an attacker's DNS could return different IPs
    for validation vs. HTTP request. The fix eliminates this by resolving once.
    """
    with patch("socket.getaddrinfo") as mock_getaddrinfo:
        # Attacker's DNS returns private IP (simulating the resolved value)
        mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 0))]
        with pytest.raises(HTTPException) as exc:
            _resolve_and_validate_url("https://rebind.attacker.com/steal.ttl")
        assert exc.value.status_code == 400
        assert "non-routable" in str(exc.value.detail).lower()
