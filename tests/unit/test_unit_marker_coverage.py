# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard against unit tests that CI silently never runs.

Every package test target filters with `-m unit`. A test that carries no `unit`
marker is therefore *deselected* rather than failed, so the suite reports green
while the test never executes. This has already happened twice: 18 whole files
plus 22 individual tests were dead in CI.

`--strict-markers` does not catch this — it only rejects *unknown* markers, not
*absent* ones. So this test asserts the convention directly: within any path a
`-m unit` target collects, every test must be reachable under that filter.

The scanned paths are derived from the `project.json` test targets themselves,
so a newly added package is covered without editing this test.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent

# Paths whose tests are legitimately excluded from `-m unit`.
EXCLUDED_PARTS = ("/integ/", "/cdk/", "/load/")
# Markers that mean "deliberately not a unit test".
NON_UNIT_MARKERS = {"integ", "slow", "security"}
# A test carrying any of these is accounted for: either `-m unit` selects it,
# or it is explicitly declared as something other than a unit test.
SELECTED_BY_UNIT = {"unit"} | NON_UNIT_MARKERS


def _project_files() -> list[Path]:
    """Every project.json in the repo, excluding node_modules."""
    return [p for p in REPO_ROOT.glob("**/project.json") if "node_modules" not in p.parts]


def _unit_filtered_projects() -> list[Path]:
    """project.json files whose raw text shows a `-m unit` test filter.

    Deliberately a text check, not a parse: it must still spot a file that
    `json` cannot load, so an unparseable project can be reported rather than
    silently skipped (that is how ontology-engine — the largest package — went
    unscanned in the first place).
    """
    return [p for p in _project_files() if "-m unit" in p.read_text()]


def _unit_filtered_test_paths() -> dict[Path, Path]:
    """Map project.json -> the path its `-m unit` pytest target collects.

    Raises on a project.json that cannot be parsed, so a malformed file fails
    the suite instead of quietly dropping a package from the scan.
    """
    paths: dict[Path, Path] = {}
    for project_file in _unit_filtered_projects():
        try:
            config = json.loads(project_file.read_text())
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"{project_file.relative_to(REPO_ROOT)} is not valid JSON "
                f"({exc}). Nx tolerates trailing commas, but tooling that parses "
                f"it strictly — including this guard — silently skips the package."
            ) from exc
        command = config.get("targets", {}).get("test", {}).get("command", "")
        if "-m unit" not in command:
            continue
        # Grab the path argument that follows `pytest`.
        match = re.search(r"pytest\s+(\S+)", command)
        if match:
            paths[project_file] = REPO_ROOT / match.group(1)
    return paths


def _marker_names(decorators: list[ast.expr]) -> set[str]:
    """Marker names from a decorator list (`@pytest.mark.X`, incl. calls)."""
    names: set[str] = set()
    for dec in decorators:
        node = dec.func if isinstance(dec, ast.Call) else dec
        # pytest.mark.X -> Attribute(attr='X', value=Attribute(attr='mark'))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
            names.add(node.attr)
    return names


def _module_level_markers(tree: ast.Module) -> set[str]:
    """Markers from a module-level `pytestmark = ...` assignment."""
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        values = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
        names |= _marker_names(list(values))
    return names


def _unreachable_tests(path: Path) -> list[str]:
    """Test names in `path` that `-m unit` would deselect."""
    tree = ast.parse(path.read_text())
    module_markers = _module_level_markers(tree)
    if module_markers & SELECTED_BY_UNIT:
        return []  # covered module-wide, or deliberately not unit

    unreachable: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test_"):
                continue
            if _marker_names(node.decorator_list) & SELECTED_BY_UNIT:
                continue
            unreachable.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("Test"):
                continue
            if _marker_names(node.decorator_list) & SELECTED_BY_UNIT:
                continue
            # A class without its own marker: check each test method.
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name.startswith("test_")
                    and not (_marker_names(item.decorator_list) & SELECTED_BY_UNIT)
                ):
                    unreachable.append(f"{node.name}::{item.name}")
    return unreachable


@pytest.mark.unit
class TestUnitMarkerCoverage:
    """Every collected unit test must be selectable by `-m unit`."""

    def test_unit_filtered_targets_exist(self) -> None:
        """Sanity: we actually found the `-m unit` test targets to scan."""
        assert _unit_filtered_test_paths(), (
            "No project.json test target using '-m unit' was found — this guard would silently scan nothing."
        )

    def test_every_unit_filtered_project_is_scanned(self) -> None:
        """No `-m unit` package may drop out of the scan.

        Guards the guard: a project.json that is unparseable, or whose test
        command this cannot read, must fail loudly rather than leave a package
        silently unchecked.
        """
        expected = {p.relative_to(REPO_ROOT) for p in _unit_filtered_projects()}
        scanned = {p.relative_to(REPO_ROOT) for p in _unit_filtered_test_paths()}
        missed = expected - scanned
        assert not missed, (
            "These project.json files filter tests with `-m unit` but were not "
            "scanned, so unmarked tests in them would go unnoticed: "
            f"{sorted(str(m) for m in missed)}"
        )

    def test_no_test_is_silently_deselected(self) -> None:
        """No test file may lack a marker that `-m unit` selects."""
        offenders: dict[str, list[str]] = {}
        for base in _unit_filtered_test_paths().values():
            if not base.exists():
                continue
            for test_file in base.glob("**/test_*.py"):
                posix = test_file.as_posix()
                if any(part in posix for part in EXCLUDED_PARTS):
                    continue
                missing = _unreachable_tests(test_file)
                if missing:
                    offenders[str(test_file.relative_to(REPO_ROOT))] = missing

        assert not offenders, (
            "These tests carry no `unit` marker, so `-m unit` deselects them and "
            "CI never runs them. Add `pytestmark = pytest.mark.unit` at module "
            "level (or mark them `integ`/`slow` if they are not unit tests):\n"
            + "\n".join(
                f"  {path}: {', '.join(names[:5])}" + (f" (+{len(names) - 5} more)" if len(names) > 5 else "")
                for path, names in sorted(offenders.items())
            )
        )
