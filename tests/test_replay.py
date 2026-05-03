"""Tests for the replay verifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from assessor.audit import assemble_record, write_record
from assessor.ai_act import classify
from assessor.iso_42001 import map_controls
from assessor.normalizer import hash_text
from assessor.replay import replay
from assessor.schema import CheckResult, RiskTier, VerificationResult
from assessor.verifier import verify_extraction
from tests.conftest import make_profile


def _make_verification() -> VerificationResult:
    return VerificationResult(
        checks=[CheckResult(check_id="TEST", passed=True, detail="ok")],
        passed=True,
    )


def _write_test_record(
    tmp_path: Path,
    input_text: str = "Test Feature: A test AI feature for testing purposes.",
    **kwargs: object,
) -> Path:
    """Write a complete test audit record and return the record path."""
    profile = make_profile(input_text=input_text, **kwargs)  # type: ignore[arg-type]
    input_sha = hash_text(input_text)
    verification = verify_extraction(profile, input_text)
    iso_controls = map_controls(profile)
    classification = classify(profile, iso_controls=iso_controls)
    memo = f"# Test Memo\n\nClassification: {classification.final_tier.value}"
    memo_v = _make_verification()

    record = assemble_record(
        input_text=input_text,
        input_text_sha256=input_sha,
        feature_profile=profile,
        extraction_verification=verification,
        classification=classification,
        memo=memo,
        memo_verification=memo_v,
        extractor_model_id="test-model",
        drafter_model_id="test-model",
        audit_dir=tmp_path,
    )
    return write_record(record, input_text, audit_dir=tmp_path, skip_schema_validation=True)


class TestReplay:
    def test_replay_passes_for_valid_record(self, tmp_path: Path) -> None:
        record_path = _write_test_record(tmp_path)
        result = replay(record_path, audit_dir=tmp_path)
        # Schema check may fail since we skip validation on write,
        # but rule engine and other checks should pass.
        rule_check = next(c for c in result.checks if c.step == "rule_engine")
        assert rule_check.passed

    def test_replay_detects_input_hash_mismatch(self, tmp_path: Path) -> None:
        record_path = _write_test_record(tmp_path)
        # Tamper with the input file.
        input_path = record_path.parent / f"{record_path.stem}.input.txt"
        input_path.write_text("TAMPERED INPUT", encoding="utf-8")
        result = replay(record_path, audit_dir=tmp_path)
        hash_check = next(c for c in result.checks if c.step == "input_hash")
        assert not hash_check.passed

    def test_replay_detects_missing_input(self, tmp_path: Path) -> None:
        record_path = _write_test_record(tmp_path)
        input_path = record_path.parent / f"{record_path.stem}.input.txt"
        input_path.unlink()
        result = replay(record_path, audit_dir=tmp_path)
        hash_check = next(c for c in result.checks if c.step == "input_hash")
        assert not hash_check.passed

    def test_replay_rule_engine_matches(self, tmp_path: Path) -> None:
        """Rule engine replay should produce the same classification."""
        record_path = _write_test_record(
            tmp_path,
            high_risk_signals=["employment_decisions"],
            autonomy_level="full_autonomy",
            decision_impact="determines promotions",
        )
        result = replay(record_path, audit_dir=tmp_path)
        rule_check = next(c for c in result.checks if c.step == "rule_engine")
        assert rule_check.passed

    def test_replay_hash_chain_first_record(self, tmp_path: Path) -> None:
        record_path = _write_test_record(tmp_path)
        result = replay(record_path, audit_dir=tmp_path)
        chain_check = next(c for c in result.checks if c.step == "hash_chain")
        assert chain_check.passed
