"""Audit record assembly, storage, and hash chain management.

Storage layout:
    audit/
      YYYY-MM-DD/
        <assessment_id>.json          # the AuditRecord
        <assessment_id>.input.txt     # verbatim original input
        <assessment_id>.memo.md       # drafted memo

Hash chain: each new record's previous_record_sha256 is the SHA-256 of
the most recent record's canonical JSON. First record has None.
audit/.chain stores the latest hash for fast lookup.

Canonicalization: sorted keys, no whitespace, UTF-8.
See AuditRecord.canonical_json() in schema.py.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from assessor.ai_act import RULE_ENGINE_VERSION
from assessor.drafter import prompt_hash as drafter_prompt_hash
from assessor.extractor import prompt_hash as extractor_prompt_hash
from assessor.schema import (
    SCHEMA_VERSION,
    AuditRecord,
    Classification,
    FeatureProfile,
    Provenance,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AUDIT_DIR = Path("audit")
_CHAIN_FILE = ".chain"


def _get_git_sha() -> str:
    """Get current git commit SHA, or empty string if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


def _read_chain_hash(audit_dir: Path) -> str | None:
    """Read the latest record hash from the chain file."""
    chain_path = audit_dir / _CHAIN_FILE
    if chain_path.exists():
        content = chain_path.read_text(encoding="utf-8").strip()
        return content if content else None
    return None


def _write_chain_hash(audit_dir: Path, sha256: str) -> None:
    """Write the latest record hash to the chain file."""
    chain_path = audit_dir / _CHAIN_FILE
    chain_path.write_text(sha256 + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON Schema validation
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "audit_record.schema.json"


def _load_json_schema() -> dict[str, object]:
    """Load the audit record JSON Schema."""
    return dict(json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")))


def validate_against_schema(record: AuditRecord) -> None:
    """Validate an AuditRecord against the JSON Schema.

    Raises jsonschema.ValidationError if the record is invalid.
    Records that fail validation are bugs — fail loud, do not write.
    """
    schema = _load_json_schema()
    data = json.loads(record.canonical_json())
    jsonschema.validate(instance=data, schema=schema)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def assemble_record(
    input_text: str,
    input_text_sha256: str,
    feature_profile: FeatureProfile,
    extraction_verification: VerificationResult,
    classification: Classification,
    memo: str,
    memo_verification: VerificationResult,
    extractor_model_id: str,
    drafter_model_id: str,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> AuditRecord:
    """Assemble a complete AuditRecord from pipeline outputs.

    Args:
        input_text: Original normalized input text.
        input_text_sha256: SHA-256 of the input text.
        feature_profile: Extracted feature profile.
        extraction_verification: Verification of extraction.
        classification: Classification result.
        memo: Drafted assessment memo.
        memo_verification: Verification of the memo.
        extractor_model_id: Ollama model used for extraction.
        drafter_model_id: Ollama model used for drafting.
        audit_dir: Root audit directory for chain lookup.

    Returns:
        A complete, immutable AuditRecord ready for validation and storage.
    """
    previous_hash = _read_chain_hash(audit_dir)

    provenance = Provenance(
        rule_engine_version=RULE_ENGINE_VERSION,
        schema_version=SCHEMA_VERSION,
        git_sha=_get_git_sha(),
        extractor_model_id=extractor_model_id,
        drafter_model_id=drafter_model_id,
        extractor_prompt_hash=extractor_prompt_hash(),
        drafter_prompt_hash=drafter_prompt_hash(),
    )

    return AuditRecord(
        assessment_id=str(uuid.uuid4()),
        timestamp_utc=datetime.now(timezone.utc),
        input_text_sha256=input_text_sha256,
        input_text_length=len(input_text),
        feature_profile=feature_profile,
        extraction_verification=extraction_verification,
        classification=classification,
        memo=memo,
        memo_verification=memo_verification,
        provenance=provenance,
        previous_record_sha256=previous_hash,
    )


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def write_record(
    record: AuditRecord,
    input_text: str,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    skip_schema_validation: bool = False,
) -> Path:
    """Validate and write an AuditRecord to disk.

    Creates the date-partitioned directory, writes the JSON record,
    the verbatim input text, and the memo. Updates the hash chain.

    Args:
        record: The assembled audit record.
        input_text: Original input text (stored alongside the record).
        audit_dir: Root audit directory.
        skip_schema_validation: Skip JSON Schema validation (for testing).

    Returns:
        Path to the written JSON record.

    Raises:
        jsonschema.ValidationError: If the record fails schema validation.
    """
    if not skip_schema_validation:
        validate_against_schema(record)

    # Date-partitioned directory.
    date_str = record.timestamp_utc.strftime("%Y-%m-%d")
    day_dir = audit_dir / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    aid = record.assessment_id

    # Write the canonical JSON record.
    record_path = day_dir / f"{aid}.json"
    record_path.write_text(record.canonical_json(), encoding="utf-8")

    # Write verbatim input text.
    input_path = day_dir / f"{aid}.input.txt"
    input_path.write_text(input_text, encoding="utf-8")

    # Write the drafted memo.
    memo_path = day_dir / f"{aid}.memo.md"
    memo_path.write_text(record.memo, encoding="utf-8")

    # Update hash chain.
    _write_chain_hash(audit_dir, record.canonical_sha256())

    return record_path


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def list_records(audit_dir: Path = DEFAULT_AUDIT_DIR) -> list[dict[str, str]]:
    """List all audit records with basic metadata.

    Returns:
        List of dicts with keys: path, assessment_id, timestamp, tier, review_status.
    """
    records: list[dict[str, str]] = []

    if not audit_dir.exists():
        return records

    for json_file in sorted(audit_dir.rglob("*.json")):
        if json_file.name == ".chain":
            continue
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            records.append({
                "path": str(json_file),
                "assessment_id": data.get("assessment_id", "unknown"),
                "timestamp": data.get("timestamp_utc", "unknown"),
                "tier": data.get("classification", {}).get("final_tier", "unknown"),
                "requires_review": str(
                    data.get("classification", {}).get("requires_human_review", False)
                ),
            })
        except (json.JSONDecodeError, KeyError):
            records.append({
                "path": str(json_file),
                "assessment_id": "CORRUPT",
                "timestamp": "unknown",
                "tier": "unknown",
                "requires_review": "unknown",
            })

    return records
