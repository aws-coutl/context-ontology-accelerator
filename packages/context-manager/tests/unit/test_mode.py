# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mode/composition contract.

The load-bearing property is the *no-op* one: ``resolve_mode`` must reproduce the
existing ``_is_agentic_engaged`` precedence exactly, so adopting it cannot move a
benchmark number. ``test_matches_legacy_is_agentic_engaged`` asserts that against
a transcription of the original logic over the full input cross-product.
"""

from __future__ import annotations

import itertools

import pytest
from coa_serve.clients.sources_registry import SourceComposition
from coa_serve.mode import Composition, Mode, ResolutionPlan, build_plan, resolve_mode

pytestmark = pytest.mark.unit


def _legacy_is_agentic_engaged(request_mode, *, agentic_retriever_exists, tier3_agentic_default):
    """Transcription of Orchestrator._is_agentic_engaged (pre-mode-module)."""
    if not agentic_retriever_exists:
        return False
    if request_mode == "agentic":
        return True
    if request_mode == "standard":
        return False
    return request_mode is None and tier3_agentic_default


@pytest.mark.parametrize(
    ("request_mode", "agentic_available", "deployment_agentic"),
    itertools.product(["agentic", "standard", None], [True, False], [True, False]),
)
def test_matches_legacy_is_agentic_engaged(request_mode, agentic_available, deployment_agentic):
    """resolve_mode must be a behavior-preserving replacement for the old predicate."""
    mode, _ = resolve_mode(
        request_mode,
        deployment_default=Mode.AGENTIC if deployment_agentic else Mode.STANDARD,
        agentic_available=agentic_available,
    )
    expected = _legacy_is_agentic_engaged(
        request_mode,
        agentic_retriever_exists=agentic_available,
        tier3_agentic_default=deployment_agentic,
    )
    assert (mode is Mode.AGENTIC) == expected


def test_request_overrides_deployment_default_both_directions():
    agentic, src = resolve_mode("agentic", deployment_default=Mode.STANDARD, agentic_available=True)
    assert (agentic, src) == (Mode.AGENTIC, "request")

    standard, src = resolve_mode("standard", deployment_default=Mode.AGENTIC, agentic_available=True)
    assert (standard, src) == (Mode.STANDARD, "request")


def test_absent_request_uses_deployment_default():
    mode, src = resolve_mode(None, deployment_default=Mode.AGENTIC, agentic_available=True)
    assert (mode, src) == (Mode.AGENTIC, "deployment")


def test_unrecognized_request_mode_falls_through_without_raising():
    mode, src = resolve_mode("turbo", deployment_default=Mode.STANDARD, agentic_available=True)
    assert (mode, src) == (Mode.STANDARD, "deployment")


def test_agentic_unavailable_forces_standard_even_when_requested():
    """Cannot run an engine that was never built."""
    mode, _ = resolve_mode("agentic", deployment_default=Mode.AGENTIC, agentic_available=False)
    assert mode is Mode.STANDARD


def test_auto_never_leaks_to_callers():
    """AUTO must collapse to a concrete mode; dispatch never sees it."""
    mode, src = resolve_mode("auto", deployment_default=Mode.STANDARD, agentic_available=True)
    assert mode is not Mode.AUTO
    assert src == "auto"


def test_auto_falls_back_to_standard_when_agentic_unavailable():
    mode, _ = resolve_mode("auto", deployment_default=Mode.STANDARD, agentic_available=False)
    assert mode is Mode.STANDARD


@pytest.mark.parametrize(
    ("structured", "unstructured", "expected"),
    [
        (True, False, Composition.STRUCTURED),
        (False, True, Composition.UNSTRUCTURED),
        (True, True, Composition.MIXED),
        (False, False, Composition.MIXED),
    ],
)
def test_composition_mapping(structured, unstructured, expected):
    got = Composition.from_source_composition(
        SourceComposition(has_structured_source=structured, has_unstructured_source=unstructured)
    )
    assert got is expected


def test_unknown_composition_fails_open_to_mixed():
    """A registry miss must not withhold any capability."""
    got = Composition.from_source_composition(SourceComposition.unknown_composition())
    assert got is Composition.MIXED


def test_build_plan_is_frozen_and_carries_provenance():
    plan = build_plan(
        "agentic",
        SourceComposition(has_structured_source=False, has_unstructured_source=True),
        deployment_default=Mode.STANDARD,
        agentic_available=True,
    )
    assert plan == ResolutionPlan(mode=Mode.AGENTIC, composition=Composition.UNSTRUCTURED, mode_source="request")
    with pytest.raises(AttributeError):
        plan.mode = Mode.STANDARD


def test_mode_values_are_wire_stable():
    """These strings are the public API (options.mode); pinning guards a rename."""
    assert (Mode.STANDARD, Mode.AGENTIC, Mode.AUTO) == ("standard", "agentic", "auto")
