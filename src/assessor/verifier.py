"""Verification checks — the deterministic guards between pipeline stages.

Two verification stages:

Post-extraction (verify_extraction):
  - Span check: every source_span is a literal substring of input_text.
  - Coverage: required fields populated with confidence != unclear.
  - Signal vocabulary: all signals are valid vocabulary tokens.
  - Contradiction check: structured form values vs. extracted values (if form provided).

Post-drafting (verify_memo):
  - Citation markers [F1], [F2]... and [Art.X] parsed from memo text.
  - Each [Fn] must resolve to a triggered citation or ISO control.
  - Unresolved citations flag the memo as ungrounded.

Verifier failures escalate to human review — they do not fail outright.
The goal is "agent prepares, human decides."
"""

from __future__ import annotations

import re

from assessor.schema import (
    ANNEX_III_SIGNALS,
    ARTICLE_5_SIGNALS,
    CheckResult,
    Classification,
    ConfidenceLevel,
    ExtractedField,
    FeatureProfile,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Post-extraction verification
# ---------------------------------------------------------------------------

# Fields that must be populated with explicit confidence for a complete extraction.
_REQUIRED_FIELDS: list[str] = [
    "feature_name",
    "description",
    "domain",
    "autonomy_level",
    "decision_impact",
]


def _check_span(
    field_name: str,
    field: ExtractedField[object],
    input_text: str,
) -> CheckResult:
    """Verify a source_span is a literal substring of the input text."""
    if not field.source_span:
        # No span provided — this is valid if confidence is not explicit.
        if field.confidence == ConfidenceLevel.EXPLICIT:
            return CheckResult(
                check_id=f"SPAN_PRESENT_{field_name}",
                passed=False,
                detail=(
                    f"Field '{field_name}' has confidence=explicit but empty source_span. "
                    f"Explicit confidence requires a verbatim span."
                ),
                severity="warning",
            )
        return CheckResult(
            check_id=f"SPAN_PRESENT_{field_name}",
            passed=True,
            detail=f"Field '{field_name}' has no span (confidence={field.confidence.value}).",
            severity="info",
        )

    # Span provided — must be a literal substring.
    if field.source_span in input_text:
        return CheckResult(
            check_id=f"SPAN_FOUND_{field_name}",
            passed=True,
            detail=f"Span for '{field_name}' found in input text.",
        )
    else:
        return CheckResult(
            check_id=f"SPAN_FOUND_{field_name}",
            passed=False,
            detail=(
                f"Span for '{field_name}' NOT found in input text. "
                f"Span: '{field.source_span[:80]}...'"
            ),
            severity="error",
        )


def _check_coverage(field_name: str, field: ExtractedField[object]) -> CheckResult:
    """Check that required fields are populated with explicit confidence."""
    if field.confidence == ConfidenceLevel.EXPLICIT:
        return CheckResult(
            check_id=f"COVERAGE_{field_name}",
            passed=True,
            detail=f"Required field '{field_name}' is explicitly populated.",
        )
    return CheckResult(
        check_id=f"COVERAGE_{field_name}",
        passed=False,
        detail=(
            f"Required field '{field_name}' has confidence={field.confidence.value}. "
            f"Expected explicit for required fields."
        ),
        severity="warning",
    )


def _check_signal_vocab(
    signals: list[str],
    valid_signals: frozenset[str],
    vocab_name: str,
) -> list[CheckResult]:
    """Verify all signal tokens are in the controlled vocabulary."""
    results: list[CheckResult] = []
    for signal in signals:
        if signal in valid_signals:
            results.append(
                CheckResult(
                    check_id=f"SIGNAL_VALID_{signal}",
                    passed=True,
                    detail=f"Signal '{signal}' is a valid {vocab_name} token.",
                    severity="info",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_id=f"SIGNAL_INVALID_{signal}",
                    passed=False,
                    detail=(
                        f"Signal '{signal}' is NOT a valid {vocab_name} token. "
                        f"The extractor produced an out-of-vocabulary signal."
                    ),
                    severity="error",
                )
            )
    return results


def _check_contradictions(
    profile: FeatureProfile,
    form_data: dict[str, object] | None,
) -> list[CheckResult]:
    """Check for contradictions between structured form data and extraction.

    If no form_data is provided, this check is skipped.
    """
    if form_data is None:
        return []

    results: list[CheckResult] = []
    # Check each form field against the corresponding extraction.
    for field_name, form_value in form_data.items():
        extracted_field = getattr(profile, field_name, None)
        if extracted_field is None or not isinstance(extracted_field, ExtractedField):
            continue

        extracted_value = extracted_field.value
        # Normalize for comparison.
        form_str = str(form_value).lower().strip()
        extracted_str = str(extracted_value).lower().strip()

        if form_str == extracted_str:
            results.append(
                CheckResult(
                    check_id=f"CONTRADICTION_{field_name}",
                    passed=True,
                    detail=f"Form value matches extraction for '{field_name}'.",
                    severity="info",
                )
            )
        else:
            results.append(
                CheckResult(
                    check_id=f"CONTRADICTION_{field_name}",
                    passed=False,
                    detail=(
                        f"Contradiction in '{field_name}': "
                        f"form='{form_str}', extracted='{extracted_str}'"
                    ),
                    severity="error",
                )
            )
    return results


def verify_extraction(
    profile: FeatureProfile,
    input_text: str,
    form_data: dict[str, object] | None = None,
) -> VerificationResult:
    """Run all post-extraction verification checks.

    Args:
        profile: The LLM-extracted feature profile.
        input_text: The original input text (for span verification).
        form_data: Optional structured form input (for contradiction checks).

    Returns:
        VerificationResult with all checks, pass/fail, and escalation info.
    """
    checks: list[CheckResult] = []
    escalation_reasons: list[str] = []

    # --- Span checks on all ExtractedField instances ---
    for field_name in FeatureProfile.model_fields:
        field_value = getattr(profile, field_name)
        if isinstance(field_value, ExtractedField):
            checks.append(_check_span(field_name, field_value, input_text))

    # --- Coverage checks on required fields ---
    for field_name in _REQUIRED_FIELDS:
        field_value = getattr(profile, field_name, None)
        if isinstance(field_value, ExtractedField):
            checks.append(_check_coverage(field_name, field_value))

    # --- Signal vocabulary checks ---
    checks.extend(
        _check_signal_vocab(
            profile.prohibited_signals.value,
            ARTICLE_5_SIGNALS,
            "ARTICLE_5_VOCAB",
        )
    )
    checks.extend(
        _check_signal_vocab(
            profile.high_risk_signals.value,
            ANNEX_III_SIGNALS,
            "ANNEX_III_VOCAB",
        )
    )

    # --- Contradiction checks ---
    checks.extend(_check_contradictions(profile, form_data))

    # --- Evaluate overall pass/fail ---
    errors = [c for c in checks if not c.passed and c.severity == "error"]
    warnings = [c for c in checks if not c.passed and c.severity == "warning"]
    passed = len(errors) == 0

    if errors:
        escalation_reasons.append(
            f"{len(errors)} error(s) in extraction verification — human review required"
        )
    if warnings:
        escalation_reasons.append(
            f"{len(warnings)} warning(s) in extraction verification"
        )

    return VerificationResult(
        checks=checks,
        passed=passed,
        escalate=len(escalation_reasons) > 0,
        escalation_reasons=escalation_reasons,
    )


# ---------------------------------------------------------------------------
# Post-drafting verification (citation grounding)
# ---------------------------------------------------------------------------

# Matches [F1], [F2], ..., [F99]
_FN_PATTERN = re.compile(r"\[F(\d+)\]")
# Matches [Art. 5(1)(a)], [Annex III, 4(b)], etc.
_ART_PATTERN = re.compile(r"\[(Art\.\s*\d+[^]]*|Annex\s+III[^]]*)\]")


def verify_memo(
    memo: str,
    classification: Classification,
) -> VerificationResult:
    """Run post-drafting citation grounding checks on the assessment memo.

    Parses [Fn] footnote markers and [Art.X] / [Annex III, X] references
    from the memo text and verifies each resolves to a triggered citation
    or ISO control in the classification.

    Args:
        memo: The drafted assessment memo (markdown text).
        classification: The classification result with triggered_citations and iso_controls.

    Returns:
        VerificationResult for the memo.
    """
    checks: list[CheckResult] = []
    escalation_reasons: list[str] = []

    # Build lookup sets for resolution.
    # Footnotes [Fn] map 1-indexed to triggered_citations + iso_controls.
    all_citable = list(classification.triggered_citations) + [
        c for c in classification.iso_controls
    ]
    max_fn = len(all_citable)

    # --- Check [Fn] markers ---
    fn_matches = _FN_PATTERN.findall(memo)
    fn_indices = {int(m) for m in fn_matches}

    if not fn_matches and classification.triggered_citations:
        checks.append(
            CheckResult(
                check_id="CITATION_PRESENT",
                passed=False,
                detail="No [Fn] citations found in memo but classification has triggered citations.",
                severity="warning",
            )
        )

    for idx in sorted(fn_indices):
        if 1 <= idx <= max_fn:
            checks.append(
                CheckResult(
                    check_id=f"CITATION_F{idx}",
                    passed=True,
                    detail=f"[F{idx}] resolves to a triggered citation or ISO control.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    check_id=f"CITATION_F{idx}",
                    passed=False,
                    detail=(
                        f"[F{idx}] does not resolve — only {max_fn} citable items available."
                    ),
                    severity="error",
                )
            )

    # --- Check [Art.X] / [Annex III, X] markers ---
    art_matches = _ART_PATTERN.findall(memo)
    citation_refs = {c.article_ref for c in classification.triggered_citations}

    for art_ref in art_matches:
        # Normalize whitespace for matching.
        normalized = re.sub(r"\s+", " ", art_ref.strip())
        if any(normalized in ref or ref in normalized for ref in citation_refs):
            checks.append(
                CheckResult(
                    check_id=f"ART_REF_{normalized}",
                    passed=True,
                    detail=f"[{normalized}] matches a triggered citation.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    check_id=f"ART_REF_{normalized}",
                    passed=False,
                    detail=f"[{normalized}] does not match any triggered citation.",
                    severity="warning",
                )
            )

    # --- Evaluate overall ---
    errors = [c for c in checks if not c.passed and c.severity == "error"]
    passed = len(errors) == 0

    if errors:
        escalation_reasons.append(
            f"{len(errors)} unresolved citation(s) — memo is ungrounded"
        )

    return VerificationResult(
        checks=checks,
        passed=passed,
        escalate=not passed,
        escalation_reasons=escalation_reasons,
    )
