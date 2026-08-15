"""
C8-4 focused tests: end-to-end acceptance harness.

Runs the real runtime pipeline (src.acceptance.run_acceptance) and asserts the
complete user journey is verified deterministically. Also checks the
`lifesearch acceptance` CLI command and JSON report wiring.

The harness itself never downloads a model. When no semantic model is
installed, the semantic scenario is reported as BLOCKED (skipped) rather than
silently passed; that does not fail the suite.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

from src.acceptance import run_acceptance


def test_acceptance_full_suite():
    """Run the entire end-to-end acceptance harness and assert it passes."""
    report = run_acceptance(verbose=False)

    # No scenario may be in a failed state.
    failed = [s for s in report["scenarios"] if s["status"] == "failed"]
    assert not failed, (
        "Acceptance scenarios failed:\n"
        + "\n".join(f"  {s['id']} {s['name']}: {s['details']}" for s in failed)
    )

    # Overall acceptance must pass (no failures).
    assert report["passed"] is True
    # A synthetic dataset must have been exercised.
    assert report["total"] >= 11
    # Evidence/confidence scenarios must have actually run (not failed).
    ids = {s["id"] for s in report["scenarios"]}
    for required in ("A", "B", "C", "E", "F", "G", "H", "I", "J", "K"):
        assert required in ids, f"scenario {required} missing from report"


def test_acceptance_scenario_details():
    """Sanity-check that key scenarios produced real, non-empty evidence."""
    report = run_acceptance(verbose=False)
    by_id = {s["id"]: s for s in report["scenarios"]}

    # G: runtime evidence must be present and classified.
    g = by_id["G"]
    assert g["status"] == "passed"
    assert g["extra"]["evidence_count"] >= 1
    assert g["extra"]["snippet"] is True

    # H: confidence classification must be deterministic and valid.
    h = by_id["H"]
    assert h["status"] == "passed"
    assert set(h["extra"]["classes"]) <= {"FACT", "INFERENCE", "GUESS"}

    # K: model readiness must clearly report missing model + no download.
    k = by_id["K"]
    assert k["status"] == "passed"
    assert k["extra"]["downloaded"] is False

    # D (semantic) is either passed (model present) or skipped (blocked),
    # never failed when the model is absent.
    d = by_id["D"]
    assert d["status"] in ("passed", "skipped")


def test_acceptance_cli_command():
    """The `lifesearch acceptance` CLI command runs and writes a JSON report."""
    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "acceptance.json")
        env = os.environ.copy()
        env["LIFESEARCH_DB"] = os.path.join(tmp, "ls.db")
        result = subprocess.run(
            [sys.executable, "-m", "src.cli.main", "acceptance", "--json", json_path],
            capture_output=True,
            text=True,
            env=env,
        )
        # The command must complete without crashing and produce a report file.
        assert result.returncode in (0, 1), result.stderr
        assert os.path.exists(json_path), result.stdout + result.stderr
        with open(json_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        assert "scenarios" in report
        assert "passed" in report
        # When the semantic model is absent the suite is allowed to pass with
        # the semantic scenario skipped; it must never claim semantic passed
        # while actually blocked.
        if report.get("semantic_blocked_by_missing_model"):
            d = next((s for s in report["scenarios"] if s["id"] == "D"), None)
            assert d is not None and d["status"] == "skipped"
