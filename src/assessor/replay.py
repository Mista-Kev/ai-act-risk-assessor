"""Deterministic replay verifier for audit records.

Replays ONLY the deterministic components of the pipeline:
  1. JSON Schema validation
  2. Input text hash verification
  3. Source span re-verification
  4. Rule engine re-execution (deterministic classification)
  5. Citation grounding re-check
  6. Hash chain link verification

Does NOT re-run LLMs — those aren't deterministic across model versions.
Only deterministic components are replayed.

Note on rule engine versioning: the replay uses the CURRENT rule engine.
If the rule engine version has changed since the assessment, the replay
will detect the mismatch and report it. A production system would need
version-pinned rule engine snapshots — noted as an extension point.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from assessor.ai_act import RULE_ENGINE_VERSION, classify
from assessor.iso_42001 import map_controls
from assessor.normalizer import hash_text
from assessor.schema import (
    AuditRecord,
    FeatureProfile,
    ReplayCheckResult,
    ReplayResult,
)
from assessor.verifier import verify_extraction, verify_memo


def _load_record(record_path: Path) -> tuple[dict[str, object], AuditRecord]:
    """Load and parse an audit record from disk.

    Returns both the raw dict (for schema validation) and the parsed model.
    """
    raw_text = record_path.read_text(encoding="utf-8")
    raw_data = json.loads(raw_text)
    record = AuditRecord.model_validate(raw_data)
    return raw_data, record


def _check_schema(raw_data: dict[str, object], schema_path: Path) -> ReplayCheckResult:
    """Validate the record against the JSON Schema."""
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.validate(instance=raw_data, schema=schema)
        return ReplayCheckResult(
            step="schema_validation",
            passed=True,
            detail="Record validates against audit_record.schema.json.",
        )
    except jsonschema.ValidationError as e:
        return ReplayCheckResult(
            step="schema_validation",
            passed=False,
            detail=f"Schema validation failed: {e.message}",
        )


def _check_input_hash(record: AuditRecord, record_path: Path) -> ReplayCheckResult:
    """Verify input_text_sha256 matches the stored input file."""
    input_path = record_path.parent / f"{record.assessment_id}.input.txt"
    if not input_path.exists():
        return ReplayCheckResult(
            step="input_hash",
            passed=False,
            detail=f"Input file not found: {input_path}",
        )

    stored_text = input_path.read_text(encoding="utf-8")
    computed_hash = hash_text(stored_text)

    if computed_hash == record.input_text_sha256:
        return ReplayCheckResult(
            step="input_hash",
            passed=True,
            detail="Input text SHA-256 matches stored file.",
        )
    return ReplayCheckResult(
        step="input_hash",
        passed=False,
        detail="Input text SHA-256 does NOT match stored file.",
        expected=record.input_text_sha256,
        actual=computed_hash,
    )


def _check_span_verification(
    record: AuditRecord, record_path: Path
) -> ReplayCheckResult:
    """Re-run source span checks against the stored input text."""
    input_path = record_path.parent / f"{record.assessment_id}.input.txt"
    if not input_path.exists():
        return ReplayCheckResult(
            step="span_verification",
            passed=False,
            detail=f"Input file not found: {input_path}",
        )

    input_text = input_path.read_text(encoding="utf-8")
    verification = verify_extraction(record.feature_profile, input_text)

    span_checks = [
        c for c in verification.checks if c.check_id.startswith("SPAN_")
    ]
    passed_count = sum(1 for c in span_checks if c.passed)
    total_count = len(span_checks)

    if verification.passed:
        return ReplayCheckResult(
            step="span_verification",
            passed=True,
            detail=f"Span verification: {passed_count}/{total_count} passed.",
        )
    failed = [c for c in span_checks if not c.passed and c.severity == "error"]
    return ReplayCheckResult(
        step="span_verification",
        passed=False,
        detail=(
            f"Span verification: {passed_count}/{total_count} passed. "
            f"Failures: {'; '.join(c.detail for c in failed[:3])}"
        ),
    )


def _check_rule_engine(record: AuditRecord) -> ReplayCheckResult:
    """Re-run the rule engine and compare classification.

    If the rule engine version has changed since the assessment,
    report the mismatch but still attempt comparison.
    """
    recorded_version = record.provenance.rule_engine_version
    version_match = recorded_version == RULE_ENGINE_VERSION

    # Re-run the rule engine against the stored feature profile.
    iso_controls = map_controls(
        record.feature_profile,
        tier=record.classification.final_tier,
    )
    replayed = classify(record.feature_profile, iso_controls=iso_controls)

    tier_match = replayed.final_tier == record.classification.final_tier
    rule_count_match = (
        len(replayed.triggered_rules) == len(record.classification.triggered_rules)
    )

    if tier_match and rule_count_match and version_match:
        return ReplayCheckResult(
            step="rule_engine",
            passed=True,
            detail=(
                f"Classification matches: {replayed.final_tier.value.upper()}. "
                f"Rule engine v{RULE_ENGINE_VERSION}."
            ),
        )

    details: list[str] = []
    if not version_match:
        details.append(
            f"Rule engine version mismatch: recorded={recorded_version}, "
            f"current={RULE_ENGINE_VERSION}"
        )
    if not tier_match:
        details.append(
            f"Tier mismatch: recorded={record.classification.final_tier.value}, "
            f"replayed={replayed.final_tier.value}"
        )
    if not rule_count_match:
        details.append(
            f"Triggered rule count mismatch: "
            f"recorded={len(record.classification.triggered_rules)}, "
            f"replayed={len(replayed.triggered_rules)}"
        )

    return ReplayCheckResult(
        step="rule_engine",
        passed=tier_match and rule_count_match,
        detail="; ".join(details),
        expected=record.classification.final_tier.value,
        actual=replayed.final_tier.value,
    )


def _check_citation_grounding(record: AuditRecord) -> ReplayCheckResult:
    """Re-run citation grounding on the stored memo."""
    verification = verify_memo(record.memo, record.classification)

    if verification.passed:
        citation_checks = [
            c for c in verification.checks if c.check_id.startswith("CITATION_")
        ]
        return ReplayCheckResult(
            step="citation_grounding",
            passed=True,
            detail=f"Citation grounding: {len(citation_checks)} citations verified.",
        )

    errors = [c for c in verification.checks if not c.passed and c.severity == "error"]
    return ReplayCheckResult(
        step="citation_grounding",
        passed=False,
        detail=f"Citation grounding failed: {'; '.join(c.detail for c in errors[:3])}",
    )


def _check_hash_chain(
    record: AuditRecord,
    record_path: Path,
    audit_dir: Path,
) -> ReplayCheckResult:
    """Verify the hash chain link to the previous record."""
    if record.previous_record_sha256 is None:
        return ReplayCheckResult(
            step="hash_chain",
            passed=True,
            detail="First record in chain (previous_record_sha256 is null).",
        )

    # Find the previous record by searching for it.
    # We need to check that some record in the audit dir has this hash.
    for json_file in sorted(audit_dir.rglob("*.json")):
        if json_file.name == ".chain" or json_file == record_path:
            continue
        try:
            raw_text = json_file.read_text(encoding="utf-8")
            prev_data = json.loads(raw_text)
            prev_record = AuditRecord.model_validate(prev_data)
            if prev_record.canonical_sha256() == record.previous_record_sha256:
                return ReplayCheckResult(
                    step="hash_chain",
                    passed=True,
                    detail=(
                        f"Hash chain verified: previous record "
                        f"{prev_record.assessment_id} matches."
                    ),
                )
        except (json.JSONDecodeError, Exception):
            continue

    return ReplayCheckResult(
        step="hash_chain",
        passed=False,
        detail=(
            f"Hash chain broken: no record found with hash "
            f"{record.previous_record_sha256[:16]}..."
        ),
        expected=record.previous_record_sha256,
        actual="not found",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "audit_record.schema.json"


def replay(
    record_path: Path,
    audit_dir: Path = Path("audit"),
) -> ReplayResult:
    """Replay all deterministic checks against a stored audit record.

    Args:
        record_path: Path to the audit record JSON file.
        audit_dir: Root audit directory for hash chain verification.

    Returns:
        ReplayResult with per-check pass/fail and overall verdict.
    """
    raw_data, record = _load_record(record_path)

    checks: list[ReplayCheckResult] = [
        _check_schema(raw_data, _SCHEMA_PATH),
        _check_input_hash(record, record_path),
        _check_span_verification(record, record_path),
        _check_rule_engine(record),
        _check_citation_grounding(record),
        _check_hash_chain(record, record_path, audit_dir),
    ]

    return ReplayResult(
        assessment_id=record.assessment_id,
        checks=checks,
        overall_passed=all(c.passed for c in checks),
        replayed_at=datetime.now(timezone.utc),
    )
