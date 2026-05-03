"""Input normalization and hashing.

Handles the first stage of the pipeline: reading input text, normalizing
whitespace, and computing the SHA-256 hash that anchors the audit record.

The normalizer does NOT alter semantic content — it only ensures consistent
whitespace handling so that span verification works reliably across platforms.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize input text for consistent processing.

    - Strips leading/trailing whitespace.
    - Normalizes line endings to \\n.
    - Does NOT collapse internal whitespace (spans must match verbatim).
    """
    text = text.strip()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def hash_text(text: str) -> str:
    """Compute SHA-256 hash of text (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_input(path: Path) -> str:
    """Read and normalize an input text file."""
    raw = path.read_text(encoding="utf-8")
    return normalize_text(raw)


def read_form(path: Path) -> dict[str, object]:
    """Read a structured form JSON file."""
    return dict(json.loads(path.read_text(encoding="utf-8")))


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of a file's contents (UTF-8 text)."""
    return hash_text(path.read_text(encoding="utf-8"))
