"""Tests for the deterministic rule engine — every Annex III branch, every Article 5 signal."""

from __future__ import annotations

import pytest

from assessor.ai_act import RULE_ENGINE_VERSION, classify
from assessor.schema import (
    ANNEX_III_VOCAB,
    ARTICLE_5_VOCAB,
    AutonomyLevel,
    RiskTier,
)
from tests.conftest import make_profile


class TestArticle5Prohibited:
    """Each Article 5 signal should trigger PROHIBITED classification."""

    @pytest.mark.parametrize("signal", sorted(ARTICLE_5_VOCAB.keys()))
    def test_each_signal_triggers_prohibited(self, signal: str) -> None:
        profile = make_profile(prohibited_signals=[signal])
        result = classify(profile)
        assert result.final_tier == RiskTier.PROHIBITED, (
            f"Signal '{signal}' should trigger PROHIBITED, got {result.final_tier}"
        )

    @pytest.mark.parametrize("signal", sorted(ARTICLE_5_VOCAB.keys()))
    def test_each_signal_has_citation(self, signal: str) -> None:
        profile = make_profile(prohibited_signals=[signal])
        result = classify(profile)
        article_ref = ARTICLE_5_VOCAB[signal][0]
        refs = {c.article_ref for c in result.triggered_citations}
        assert article_ref in refs, f"Expected citation {article_ref} for signal {signal}"

    def test_social_scoring_specific(self) -> None:
        profile = make_profile(prohibited_signals=["social_scoring"])
        result = classify(profile)
        assert result.final_tier == RiskTier.PROHIBITED
        assert any(r.rule_id == "ART5_C_SOCIAL_SCORING" for r in result.triggered_rules)
        assert result.requires_human_review

    def test_multiple_prohibited_signals(self) -> None:
        profile = make_profile(
            prohibited_signals=["social_scoring", "subliminal_technique"]
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.PROHIBITED
        assert len(result.triggered_rules) >= 2

    def test_prohibited_trumps_high(self) -> None:
        """PROHIBITED should take precedence over HIGH."""
        profile = make_profile(
            prohibited_signals=["social_scoring"],
            high_risk_signals=["employment_decisions"],
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.PROHIBITED

    def test_prohibited_obligations(self) -> None:
        profile = make_profile(prohibited_signals=["social_scoring"])
        result = classify(profile)
        assert any("prohibited" in o.lower() for o in result.obligations)


class TestAnnexIIIHighRisk:
    """Each Annex III signal should trigger HIGH classification."""

    @pytest.mark.parametrize("signal", sorted(ANNEX_III_VOCAB.keys()))
    def test_each_signal_triggers_high(self, signal: str) -> None:
        profile = make_profile(
            high_risk_signals=[signal],
            autonomy_level=AutonomyLevel.FULL_AUTONOMY,
            decision_impact="autonomous decision with legal consequences",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.HIGH, (
            f"Signal '{signal}' should trigger HIGH, got {result.final_tier}"
        )

    @pytest.mark.parametrize("signal", sorted(ANNEX_III_VOCAB.keys()))
    def test_each_signal_has_citation(self, signal: str) -> None:
        profile = make_profile(
            high_risk_signals=[signal],
            autonomy_level=AutonomyLevel.FULL_AUTONOMY,
            decision_impact="autonomous decision",
        )
        result = classify(profile)
        article_ref = ANNEX_III_VOCAB[signal][0]
        refs = {c.article_ref for c in result.triggered_citations}
        assert article_ref in refs, f"Expected citation {article_ref} for signal {signal}"

    def test_employment_decisions_high(self) -> None:
        profile = make_profile(
            high_risk_signals=["employment_decisions"],
            autonomy_level=AutonomyLevel.FULL_AUTONOMY,
            decision_impact="determines promotion and termination",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.HIGH
        assert any(r.rule_id == "ANNEX3_4B_EMPLOYMENT_DECISIONS" for r in result.triggered_rules)

    def test_credit_scoring_high(self) -> None:
        profile = make_profile(
            high_risk_signals=["creditworthiness_assessment"],
            autonomy_level=AutonomyLevel.FULL_AUTONOMY,
            decision_impact="determines loan approval",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.HIGH
        assert any(r.rule_id == "ANNEX3_5A_CREDIT" for r in result.triggered_rules)

    def test_multiple_high_risk_signals(self) -> None:
        profile = make_profile(
            high_risk_signals=["employment_decisions", "creditworthiness_assessment"],
            autonomy_level=AutonomyLevel.FULL_AUTONOMY,
            decision_impact="autonomous decisions",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.HIGH
        assert len(result.triggered_rules) >= 2

    def test_high_risk_obligations(self) -> None:
        profile = make_profile(
            high_risk_signals=["employment_decisions"],
            autonomy_level=AutonomyLevel.FULL_AUTONOMY,
            decision_impact="determines employment outcomes",
        )
        result = classify(profile)
        assert any("Art. 9" in o for o in result.obligations)
        assert any("Art. 14" in o for o in result.obligations)


class TestArticle6_3Exception:
    """Article 6(3) exception should downgrade HIGH to LIMITED under narrow conditions."""

    def test_downgrade_when_all_conditions_met(self) -> None:
        profile = make_profile(
            high_risk_signals=["employment_decisions"],
            autonomy_level=AutonomyLevel.ADVISORY_ONLY,
            decision_impact="informational recommendation for managers",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.LIMITED
        assert result.downgrade is not None
        assert result.downgrade.original_tier == RiskTier.HIGH
        assert result.downgrade.downgraded_tier == RiskTier.LIMITED
        assert result.requires_human_review

    def test_no_downgrade_full_autonomy(self) -> None:
        """FULL_AUTONOMY fails the assistive autonomy condition."""
        profile = make_profile(
            high_risk_signals=["employment_decisions"],
            autonomy_level=AutonomyLevel.FULL_AUTONOMY,
            decision_impact="informational recommendation",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.HIGH
        assert result.downgrade is None

    def test_no_downgrade_consequential_impact(self) -> None:
        """Non-informational impact fails the informational condition."""
        profile = make_profile(
            high_risk_signals=["employment_decisions"],
            autonomy_level=AutonomyLevel.ADVISORY_ONLY,
            decision_impact="determines employee termination directly",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.HIGH
        assert result.downgrade is None

    def test_no_downgrade_for_prohibited(self) -> None:
        """PROHIBITED should never be downgraded."""
        profile = make_profile(
            prohibited_signals=["social_scoring"],
            autonomy_level=AutonomyLevel.ADVISORY_ONLY,
            decision_impact="informational recommendation",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.PROHIBITED
        assert result.downgrade is None

    def test_downgrade_human_in_loop(self) -> None:
        """HUMAN_IN_THE_LOOP also qualifies for assistive autonomy."""
        profile = make_profile(
            high_risk_signals=["creditworthiness_assessment"],
            autonomy_level=AutonomyLevel.HUMAN_IN_THE_LOOP,
            decision_impact="advisory credit assessment recommendation",
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.LIMITED
        assert result.downgrade is not None


class TestArticle50Limited:
    """GenAI features should trigger LIMITED classification."""

    def test_content_generation_limited(self) -> None:
        profile = make_profile(generates_content=True)
        result = classify(profile)
        assert result.final_tier == RiskTier.LIMITED

    def test_human_interaction_limited(self) -> None:
        profile = make_profile(interacts_with_humans=True)
        result = classify(profile)
        assert result.final_tier == RiskTier.LIMITED

    def test_deepfake_generation_limited(self) -> None:
        profile = make_profile(generates_deepfakes=True)
        result = classify(profile)
        assert result.final_tier == RiskTier.LIMITED

    def test_genai_plus_content_limited(self) -> None:
        profile = make_profile(
            generates_content=True,
            interacts_with_humans=True,
        )
        result = classify(profile)
        assert result.final_tier == RiskTier.LIMITED

    def test_limited_obligations(self) -> None:
        profile = make_profile(generates_content=True)
        result = classify(profile)
        assert any("transparency" in o.lower() or "Art. 50" in o for o in result.obligations)


class TestMinimalDefault:
    """Features with no signals should default to MINIMAL."""

    def test_no_signals_minimal(self) -> None:
        profile = make_profile()
        result = classify(profile)
        assert result.final_tier == RiskTier.MINIMAL

    def test_minimal_no_human_review(self) -> None:
        profile = make_profile()
        result = classify(profile)
        assert not result.requires_human_review

    def test_minimal_obligations(self) -> None:
        profile = make_profile()
        result = classify(profile)
        assert any("voluntary" in o.lower() or "no mandatory" in o.lower() for o in result.obligations)


class TestAuditCompleteness:
    """All rules should emit a RuleEval regardless of match status."""

    def test_all_rules_evaluated_count(self) -> None:
        """Total rules evaluated should match Art.5 + Annex III + Art.50 rules."""
        profile = make_profile()
        result = classify(profile)
        expected_count = len(ARTICLE_5_VOCAB) + len(ANNEX_III_VOCAB) + 3  # 3 Art.50 rules
        assert len(result.all_rules_evaluated) == expected_count

    def test_non_matching_rules_included(self) -> None:
        profile = make_profile()
        result = classify(profile)
        non_matched = [r for r in result.all_rules_evaluated if not r.matched]
        assert len(non_matched) > 0

    def test_all_rules_have_provision_text(self) -> None:
        profile = make_profile()
        result = classify(profile)
        for rule_eval in result.all_rules_evaluated:
            assert rule_eval.verbatim_provision, f"Rule {rule_eval.rule_id} missing provision text"

    def test_rule_engine_version_exists(self) -> None:
        assert RULE_ENGINE_VERSION
        assert "." in RULE_ENGINE_VERSION  # semver-ish
