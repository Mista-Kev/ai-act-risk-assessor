"""Generate the JSON Schema artifact from the AuditRecord Pydantic model.

This module exists so the JSON Schema stays in sync with the Pydantic model.
Run as a script to regenerate: python -m assessor.audit_schema
"""

from __future__ import annotations

import json
from pathlib import Path

from assessor.schema import AuditRecord

_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "audit_record.schema.json"


def generate_schema() -> dict[str, object]:
    """Generate JSON Schema from the AuditRecord Pydantic model."""
    schema = AuditRecord.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "AI Act Risk Assessor — Audit Record"
    schema["description"] = (
        "Immutable, hash-chained audit record for a single AI Act risk assessment. "
        "Validated before writing. Canonical JSON serialization used for hashing."
    )
    return schema


def write_schema(path: Path = _SCHEMA_PATH) -> Path:
    """Write the JSON Schema to disk."""
    schema = generate_schema()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    out = write_schema()
    print(f"Schema written to {out}")
