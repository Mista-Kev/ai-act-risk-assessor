"""LLM-based structured extraction via Instructor + Ollama.

The extractor's job is ONLY to populate a FeatureProfile. It does not
classify. It does not decide. The prompt explicitly forbids reasoning
about risk tiers.

Uses Instructor's JSON mode with Ollama's OpenAI-compatible API to enforce
the FeatureProfile schema at decoding time. Temperature=0 and a fixed seed
for deterministic output in the PoC.

v2 enhancement: self-consistency (N>1 runs, majority vote per field).
The insertion point is marked below but not implemented.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import instructor
import jinja2
from openai import OpenAI

from assessor.schema import (
    ANNEX_III_VOCAB,
    ARTICLE_5_VOCAB,
    FeatureProfile,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_EXTRACTOR_MODEL = "gemma4:e4b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

_PROMPT_DIR = Path(__file__).parent / "prompts"
_EXTRACTOR_TEMPLATE_PATH = _PROMPT_DIR / "extractor.jinja2"


def _load_template() -> jinja2.Template:
    """Load the extractor prompt template."""
    return jinja2.Template(
        _EXTRACTOR_TEMPLATE_PATH.read_text(encoding="utf-8"),
        undefined=jinja2.StrictUndefined,
    )


def prompt_hash() -> str:
    """SHA-256 of the extractor prompt template, for audit provenance."""
    content = _EXTRACTOR_TEMPLATE_PATH.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _render_prompt(input_text: str, form_summary: str | None = None) -> str:
    """Render the extraction prompt with the input text and vocabulary."""
    template = _load_template()
    return template.render(
        input_text=input_text,
        form_summary=form_summary or "",
        article5_signals=sorted(ARTICLE_5_VOCAB.keys()),
        annex3_signals=sorted(ANNEX_III_VOCAB.keys()),
    )


def _make_client(model: str, base_url: str) -> instructor.Instructor:
    """Create an Instructor-patched OpenAI client for Ollama."""
    openai_client = OpenAI(
        base_url=base_url,
        api_key="ollama",  # Ollama doesn't require a real key.
    )
    return instructor.from_openai(openai_client, mode=instructor.Mode.JSON)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(
    input_text: str,
    form_summary: str | None = None,
    model: str = DEFAULT_EXTRACTOR_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.0,
    seed: int = 42,
    max_retries: int = 2,
) -> FeatureProfile:
    """Extract a FeatureProfile from free-text input using a local LLM.

    Args:
        input_text: The normalized input text describing the AI feature.
        form_summary: Optional stringified form data to supplement extraction.
        model: Ollama model tag.
        base_url: Ollama API base URL.
        temperature: LLM temperature. 0 for deterministic PoC output.
        seed: Random seed for reproducibility.
        max_retries: Instructor retry count on validation failure.

    Returns:
        A validated FeatureProfile with provenance-tracked fields.

    Raises:
        instructor.exceptions.InstructorRetryException: If extraction fails
            after max_retries attempts.
    """
    client = _make_client(model, base_url)
    rendered_prompt = _render_prompt(input_text, form_summary)

    # --- v2 enhancement point: self-consistency ---
    # For N>1 runs, loop here with different seeds, collect N profiles,
    # then merge via majority vote per field. For PoC, N=1.

    profile: FeatureProfile = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a structured data extractor. "
                    "Return ONLY the requested JSON structure. "
                    "Do not add commentary or explanation."
                ),
            },
            {"role": "user", "content": rendered_prompt},
        ],
        response_model=FeatureProfile,
        temperature=temperature,
        seed=seed,
        max_retries=max_retries,
        # Disable the model's chain-of-thought. Extraction is a mechanical
        # schema-fill; the reasoning trace roughly triples latency for no gain
        # and the deterministic verifier audits the output regardless.
        reasoning_effort="none",
    )

    return profile
