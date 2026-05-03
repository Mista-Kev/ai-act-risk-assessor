"""Tests for audit record assembly, storage, hash chain, and replay."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from assessor.ai_act import classify
from assessor.audit import assemble_record, list_records, write_record
from assessor.iso_42001 import map_controls
from assessor.normalizer import hash_text
from assessor.schema import (
    AuditRecord,
    CheckResult,
    Classification,
    RiskTier,
    VerificationResult,
)
from assessor.verifier import verify_extraction
from tests.conftest import make_profile


def _make_verification() -> VerificationResult:
    return VerificationResult(
        checks=[
            CheckResult(check_id="TEST_CHECK", passed=True, detail="ok"),
        ],
        passed=True,
    )


def _run_pipeline(
    input_text: str = "Test Feature: A test AI feature for testing purposes.",
    **profile_kwargs: object,
) -> tuple[str, str, object, VerificationResult, Classification, str, VerificationResult]:
    """Run the deterministic parts of the pipeline for testing."""
    profile = make_profile(input_text=input_text, **profile_kwargs)  # type: ignore[arg-type]
    input_sha = hash_text(input_text)
    verification = verify_extraction(profile, input_text)
    iso_controls = map_controls(profile, tier=None)
    classification = classify(profile, iso_controls=iso_controls)
    iso_controls = map_controls(profile, tier=classification.final_tier)
    classification = classify(profile, iso_controls=iso_controls)
    memo = f"# Test Memo\n\nClassification: {classification.final_tier.value}"
    memo_verification = _make_verification()
    return input_text, input_sha, profile, verification, classification, memo, memo_verification


class TestAuditRecordAssembly:
    def test_assemble_creates_valid_record(self, tmp_path: Path) -> None:
        input_text, input_sha, profile, verification, classification, memo, memo_v = _run_pipeline()
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
        assert record.assessment_id
        assert record.input_text_sha256 == input_sha
        assert record.classification.final_tier == RiskTier.MINIMAL

    def test_first_record_has_null_previous(self, tmp_path: Path) -> None:
        input_text, input_sha, profile, verification, classification, memo, memo_v = _run_pipeline()
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
        assert record.previous_record_sha256 is None


class TestCanonicalJson:
    def test_canonical_json_deterministic(self, tmp_path: Path) -> None:
        input_text, input_sha, profile, verification, classification, memo, memo_v = _run_pipeline()
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
        json1 = record.canonical_json()
        json2 = record.canonical_json()
        assert json1 == json2

    def test_canonical_json_sorted_keys(self, tmp_path: Path) -> None:
        input_text, input_sha, profile, verification, classification, memo, memo_v = _run_pipeline()
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
        parsed = json.loads(record.canonical_json())
        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_sha256_deterministic(self, tmp_path: Path) -> None:
        input_text, input_sha, profile, verification, classification, memo, memo_v = _run_pipeline()
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
        assert record.canonical_sha256() == record.canonical_sha256()


class TestAuditStorage:
    def test_write_creates_files(self, tmp_path: Path) -> None:
        input_text, input_sha, profile, verification, classification, memo, memo_v = _run_pipeline()
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
        record_path = write_record(record, input_text, audit_dir=tmp_path, skip_schema_validation=True)
        assert record_path.exists()
        assert (record_path.parent / f"{record.assessment_id}.input.txt").exists()
        assert (record_path.parent / f"{record.assessment_id}.memo.md").exists()

    def test_hash_chain_updates(self, tmp_path: Path) -> None:
        # Write first record.
        input_text, input_sha, profile, verification, classification, memo, memo_v = _run_pipeline()
        record1 = assemble_record(
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
        write_record(record1, input_text, audit_dir=tmp_path, skip_schema_validation=True)

        # Write second record — should have previous hash.
        record2 = assemble_record(
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
        assert record2.previous_record_sha256 is not None
        assert record2.previous_record_sha256 == record1.canonical_sha256()

    def test_list_records(self, tmp_path: Path) -> None:
        input_text, input_sha, profile, verification, classification, memo, memo_v = _run_pipeline()
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
        write_record(record, input_text, audit_dir=tmp_path, skip_schema_validation=True)
        records = list_records(tmp_path)
        assert len(records) == 1
        assert records[0]["assessment_id"] == record.assessment_id
