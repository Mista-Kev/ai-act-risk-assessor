"""LLM-based assessment memo drafting via Instructor + Ollama.

The drafter receives ONLY the validated Classification — never the original
input text. This prevents drift back into unclassified territory.
Every claim must cite a triggered provision or ISO control via [Fn] markers.

Temperature 0.3 for prose readability. This is a deliberate tradeoff:
higher temperature improves natural language quality but risks hallucinated
claims. The post-drafting citation verifier catches any ungrounded claims,
so the tradeoff is acceptable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jinja2
from openai import OpenAI

from assessor.schema import Classification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DRAFTER_MODEL = "gemma4:e4b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

_PROMPT_DIR = Path(__file__).parent / "prompts"
_DRAFTER_TEMPLATE_PATH = _PROMPT_DIR / "drafter.jinja2"


def _load_template() -> jinja2.Template:
    """Load the drafter prompt template."""
    return jinja2.Template(
        _DRAFTER_TEMPLATE_PATH.read_text(encoding="utf-8"),
        undefined=jinja2.StrictUndefined,
    )


def prompt_hash() -> str:
    """SHA-256 of the drafter prompt template, for audit provenance."""
    content = _DRAFTER_TEMPLATE_PATH.read_text(encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _render_prompt(classification: Classification) -> str:
    """Render the drafter prompt with the classification data."""
    template = _load_template()
    return template.render(
        feature_name=classification.feature_name,
        feature_description=classification.feature_description,
        domain=classification.domain,
        final_tier=classification.final_tier.value.upper(),
        downgrade=classification.downgrade,
        requires_human_review=classification.requires_human_review,
        human_review_reasons=classification.human_review_reasons,
        triggered_rules=classification.triggered_rules,
        obligations=classification.obligations,
        iso_controls=classification.iso_controls,
        citations=classification.triggered_citations,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def draft(
    classification: Classification,
    model: str = DEFAULT_DRAFTER_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.3,
    seed: int = 42,
) -> str:
    """Draft an assessment memo from a validated Classification.

    The drafter does not see the original input text. It works only from
    the structured classification data, ensuring every claim is grounded
    in deterministic rule engine output.

    Args:
        classification: The complete classification result.
        model: Ollama model tag for drafting.
        base_url: Ollama API base URL.
        temperature: Higher than extraction (0.3) for prose quality.
        seed: Random seed for reproducibility.

    Returns:
        Assessment memo as markdown text.
    """
    client = OpenAI(
        base_url=base_url,
        api_key="ollama",
    )

    rendered_prompt = _render_prompt(classification)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compliance memo drafter. Write professional, "
                    "concise assessment memos. Every factual claim MUST include "
                    "a citation marker [Fn] or [Art. X]. Do not invent provisions "
                    "that are not listed in the prompt."
                ),
            },
            {"role": "user", "content": rendered_prompt},
        ],
        temperature=temperature,
        seed=seed,
        # Disable chain-of-thought: the memo is generated from already-decided
        # classification data, so the reasoning trace only adds latency.
        reasoning_effort="none",
    )

    content = response.choices[0].message.content
    if content is None:
        return ""
    return content.strip()
