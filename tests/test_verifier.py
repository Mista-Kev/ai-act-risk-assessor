"""Tests for the verifier — span checks, coverage, signal vocab, citation grounding."""

from __future__ import annotations

import pytest

from assessor.schema import (
    Citation,
    Classification,
    ConfidenceLevel,
    ExtractedField,
    ISOControl,
    RiskTier,
    RuleEval,
)
from assessor.verifier import verify_extraction, verify_memo
from tests.conftest import make_profile


class TestSpanVerification:
    """Source span checks against input text."""

    def test_span_found(self) -> None:
        input_text = "This is a Test Feature for credit scoring purposes."
        profile = make_profile(
            feature_name="Test Feature",
            input_text=input_text,
        )
        result = verify_extraction(profile, input_text)
        span_checks = [c for c in result.checks if "SPAN_FOUND" in c.check_id]
        # At least the feature_name span should be found.
        assert any(c.passed for c in span_checks)

    def test_span_not_found_flagged(self) -> None:
        input_text = "Some completely different text."
        profile = make_profile(
            feature_name="Credit Scorer",
            input_text="Credit Scorer is a feature.",
        )
        # The profile has spans referencing "Credit Scorer" but input_text doesn't have it.
        result = verify_extraction(profile, input_text)
        span_errors = [
            c for c in result.checks
            if "SPAN_FOUND" in c.check_id and not c.passed
        ]
        assert len(span_errors) > 0

    def test_empty_span_not_error(self) -> None:
        """Empty spans with non-explicit confidence should not be errors."""
        input_text = "Test Feature: A test AI feature."
        profile = make_profile(input_text=input_text)
        result = verify_extraction(profile, input_text)
        # Fields with empty spans and inferred confidence should pass.
        info_checks = [c for c in result.checks if c.severity == "info"]
        assert len(info_checks) > 0


class TestCoverageVerification:
    """Required field coverage checks."""

    def test_required_fields_coverage(self) -> None:
        input_text = "Test Feature: A test AI feature."
        profile = make_profile(input_text=input_text)
        result = verify_extraction(profile, input_text)
        coverage_checks = [c for c in result.checks if "COVERAGE" in c.check_id]
        assert len(coverage_checks) > 0


class TestSignalVocab:
    """Signal vocabulary validation."""

    def test_valid_prohibited_signal(self) -> None:
        input_text = "Test"
        profile = make_profile(prohibited_signals=["social_scoring"], input_text=input_text)
        result = verify_extraction(profile, input_text)
        signal_checks = [c for c in result.checks if "SIGNAL_VALID" in c.check_id]
        assert any(c.passed and "social_scoring" in c.detail for c in signal_checks)

    def test_invalid_signal_flagged(self) -> None:
        input_text = "Test"
        profile = make_profile(prohibited_signals=["made_up_signal"], input_text=input_text)
        result = verify_extraction(profile, input_text)
        invalid_checks = [c for c in result.checks if "SIGNAL_INVALID" in c.check_id]
        assert len(invalid_checks) == 1
        assert not invalid_checks[0].passed

    def test_valid_annex3_signal(self) -> None:
        input_text = "Test"
        profile = make_profile(high_risk_signals=["employment_decisions"], input_text=input_text)
        result = verify_extraction(profile, input_text)
        signal_checks = [c for c in result.checks if "SIGNAL_VALID" in c.check_id]
        assert any("employment_decisions" in c.detail for c in signal_checks)


class TestContradictionCheck:
    """Structured form vs extraction contradiction checks."""

    def test_no_contradiction(self) -> None:
        input_text = "Test"
        profile = make_profile(domain="finance", input_text=input_text)
        form_data = {"domain": "finance"}
        result = verify_extraction(profile, input_text, form_data=form_data)
        contra_checks = [c for c in result.checks if "CONTRADICTION" in c.check_id]
        assert all(c.passed for c in contra_checks)

    def test_contradiction_flagged(self) -> None:
        input_text = "Test"
        profile = make_profile(domain="finance", input_text=input_text)
        form_data = {"domain": "healthcare"}
        result = verify_extraction(profile, input_text, form_data=form_data)
        contra_checks = [c for c in result.checks if "CONTRADICTION" in c.check_id]
        assert any(not c.passed for c in contra_checks)


class TestEscalation:
    """Verification failures should escalate, not fail outright."""

    def test_errors_trigger_escalation(self) -> None:
        input_text = "Some text."
        profile = make_profile(
            prohibited_signals=["not_a_real_signal"],
            input_text=input_text,
        )
        result = verify_extraction(profile, input_text)
        assert result.escalate
        assert len(result.escalation_reasons) > 0

    def test_clean_profile_no_escalation(self) -> None:
        input_text = "Test Feature: A test AI feature."
        profile = make_profile(input_text=input_text)
        result = verify_extraction(profile, input_text)
        assert result.passed


class TestMemoVerification:
    """Citation grounding checks for the drafted memo."""

    def _make_classification(self) -> Classification:
        return Classification(
            final_tier=RiskTier.HIGH,
            triggered_rules=[
                RuleEval(
                    rule_id="ANNEX3_4B_EMPLOYMENT_DECISIONS",
                    matched=True,
                    tier=RiskTier.HIGH,
                    article_ref="Annex III, 4(b)",
                    verbatim_provision="AI for employment decisions",
                    triggered_signals=["employment_decisions"],
                ),
            ],
            all_rules_evaluated=[],
            triggered_citations=[
                Citation(
                    article_ref="Annex III, 4(b)",
                    verbatim_text="AI for employment decisions",
                ),
            ],
            obligations=["Risk management required"],
            iso_controls=[
                ISOControl(
                    control_id="A.5.2",
                    title="AI Risk Assessment",
                    description="...",
                    applicability="High-risk system",
                ),
            ],
            requires_human_review=True,
        )

    def test_valid_citations_pass(self) -> None:
        classification = self._make_classification()
        memo = "This system is classified HIGH [F1] and requires risk assessment [F2]."
        result = verify_memo(memo, classification)
        assert result.passed

    def test_invalid_citation_index(self) -> None:
        classification = self._make_classification()
        memo = "This system is classified HIGH [F1] [F99]."
        result = verify_memo(memo, classification)
        errors = [c for c in result.checks if not c.passed and c.severity == "error"]
        assert len(errors) > 0

    def test_art_reference_matches(self) -> None:
        classification = self._make_classification()
        memo = "Per [Annex III, 4(b)], this system is high-risk."
        result = verify_memo(memo, classification)
        art_checks = [c for c in result.checks if "ART_REF" in c.check_id]
        assert any(c.passed for c in art_checks)

    def test_no_citations_warns(self) -> None:
        classification = self._make_classification()
        memo = "This system is high risk."
        result = verify_memo(memo, classification)
        warns = [c for c in result.checks if not c.passed and c.severity == "warning"]
        assert len(warns) > 0
