"""Deterministic EU AI Act rule engine.

Cascade order matters and is load-bearing:
  1. Article 5 — prohibited practices. Short-circuits if matched.
  2. Annex III + Article 6 — high-risk. Evaluates all categories.
  3. Article 6(3) — exception check. Can downgrade HIGH → LIMITED,
     but always escalates to human review.
  4. Article 50 — GenAI transparency obligations → LIMITED.
  5. Default → MINIMAL.

Every rule emits a RuleEval whether it matched or not.
The rule engine never calls an LLM. Classification is purely symbolic.

Bump RULE_ENGINE_VERSION on any logic change.
"""

from __future__ import annotations

from assessor.schema import (
    ANNEX_III_VOCAB,
    ARTICLE_5_VOCAB,
    AutonomyLevel,
    Citation,
    Classification,
    DowngradeInfo,
    FeatureProfile,
    ISOControl,
    RiskTier,
    RuleEval,
)

# Bump on ANY logic change. Combined with git SHA at runtime,
# this is the determinism anchor for replay verification.
RULE_ENGINE_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Tier severity ordering for resolution
# ---------------------------------------------------------------------------

_TIER_SEVERITY: dict[RiskTier, int] = {
    RiskTier.PROHIBITED: 4,
    RiskTier.HIGH: 3,
    RiskTier.LIMITED: 2,
    RiskTier.MINIMAL: 1,
}


def _highest_tier(tiers: list[RiskTier]) -> RiskTier:
    """Return the most severe tier from a list. Defaults to MINIMAL."""
    if not tiers:
        return RiskTier.MINIMAL
    return max(tiers, key=lambda t: _TIER_SEVERITY[t])


# ---------------------------------------------------------------------------
# Article 5 — Prohibited practices
# ---------------------------------------------------------------------------

# Map from signal token to rule_id.
_ART5_RULES: dict[str, str] = {
    "subliminal_technique": "ART5_A_SUBLIMINAL",
    "manipulative_technique": "ART5_A_MANIPULATIVE",
    "deceptive_technique": "ART5_A_DECEPTIVE",
    "exploit_age_vulnerability": "ART5_B_AGE",
    "exploit_disability_vulnerability": "ART5_B_DISABILITY",
    "exploit_socioeconomic_vulnerability": "ART5_B_SOCIOECONOMIC",
    "social_scoring": "ART5_C_SOCIAL_SCORING",
    "predictive_policing_profiling": "ART5_D_PREDICTIVE_POLICING",
    "untargeted_facial_scraping": "ART5_E_FACIAL_SCRAPING",
    "emotion_recognition_workplace": "ART5_F_EMOTION_WORKPLACE",
    "emotion_recognition_education": "ART5_F_EMOTION_EDUCATION",
    "biometric_categorization_race": "ART5_G_BIOMETRIC_RACE",
    "biometric_categorization_politics": "ART5_G_BIOMETRIC_POLITICS",
    "biometric_categorization_union": "ART5_G_BIOMETRIC_UNION",
    "biometric_categorization_religion": "ART5_G_BIOMETRIC_RELIGION",
    "biometric_categorization_sex_life": "ART5_G_BIOMETRIC_SEX_LIFE",
    "realtime_remote_biometric_id": "ART5_H_REALTIME_BIOMETRIC",
}

# Map from signal token to rule_id.
_ANNEX3_RULES: dict[str, str] = {
    "remote_biometric_identification": "ANNEX3_1A_BIOMETRIC_ID",
    "biometric_categorization": "ANNEX3_1B_BIOMETRIC_CAT",
    "emotion_recognition": "ANNEX3_1C_EMOTION",
    "critical_infrastructure_safety": "ANNEX3_2A_CRITICAL_INFRA",
    "education_access_admission": "ANNEX3_3A_EDU_ACCESS",
    "education_outcome_evaluation": "ANNEX3_3B_EDU_OUTCOME",
    "education_level_assessment": "ANNEX3_3C_EDU_LEVEL",
    "education_cheating_detection": "ANNEX3_3D_EDU_CHEATING",
    "employment_recruitment": "ANNEX3_4A_EMPLOYMENT_RECRUIT",
    "employment_decisions": "ANNEX3_4B_EMPLOYMENT_DECISIONS",
    "creditworthiness_assessment": "ANNEX3_5A_CREDIT",
    "insurance_risk_assessment": "ANNEX3_5B_INSURANCE",
    "public_benefits_eligibility": "ANNEX3_5C_PUBLIC_BENEFITS",
    "emergency_dispatch": "ANNEX3_5D_EMERGENCY",
    "law_enforcement_risk_victim": "ANNEX3_6A_LE_VICTIM",
    "law_enforcement_polygraph": "ANNEX3_6B_LE_POLYGRAPH",
    "law_enforcement_evidence_reliability": "ANNEX3_6C_LE_EVIDENCE",
    "law_enforcement_risk_offending": "ANNEX3_6D_LE_OFFENDING",
    "law_enforcement_profiling": "ANNEX3_6E_LE_PROFILING",
    "migration_polygraph": "ANNEX3_7A_MIG_POLYGRAPH",
    "migration_risk_assessment": "ANNEX3_7B_MIG_RISK",
    "migration_application_evaluation": "ANNEX3_7C_MIG_APPLICATION",
    "migration_person_detection": "ANNEX3_7D_MIG_DETECTION",
    "justice_research_application": "ANNEX3_8A_JUSTICE",
    "democratic_process_influence": "ANNEX3_8B_DEMOCRACY",
}


def _evaluate_article5(profile: FeatureProfile) -> list[RuleEval]:
    """Evaluate all Article 5 prohibited-practice rules.

    Returns one RuleEval per rule, regardless of whether it matched.
    """
    extracted_signals = set(profile.prohibited_signals.value)
    results: list[RuleEval] = []

    for signal, rule_id in _ART5_RULES.items():
        article_ref, verbatim = ARTICLE_5_VOCAB[signal]
        matched = signal in extracted_signals
        results.append(
            RuleEval(
                rule_id=rule_id,
                matched=matched,
                tier=RiskTier.PROHIBITED,
                article_ref=article_ref,
                verbatim_provision=verbatim,
                triggered_signals=[signal] if matched else [],
                rationale=(
                    f"Signal '{signal}' present in extracted prohibited_signals"
                    if matched
                    else f"Signal '{signal}' not present in extracted prohibited_signals"
                ),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Annex III + Article 6 — High-risk
# ---------------------------------------------------------------------------


def _evaluate_annex3(profile: FeatureProfile) -> list[RuleEval]:
    """Evaluate all Annex III high-risk rules.

    Returns one RuleEval per rule, regardless of whether it matched.
    """
    extracted_signals = set(profile.high_risk_signals.value)
    results: list[RuleEval] = []

    for signal, rule_id in _ANNEX3_RULES.items():
        article_ref, verbatim = ANNEX_III_VOCAB[signal]
        matched = signal in extracted_signals
        results.append(
            RuleEval(
                rule_id=rule_id,
                matched=matched,
                tier=RiskTier.HIGH,
                article_ref=article_ref,
                verbatim_provision=verbatim,
                triggered_signals=[signal] if matched else [],
                rationale=(
                    f"Signal '{signal}' present in extracted high_risk_signals"
                    if matched
                    else f"Signal '{signal}' not present in extracted high_risk_signals"
                ),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Article 6(3) — Exception: downgrade HIGH → LIMITED
# ---------------------------------------------------------------------------
# Narrow conditions (all three must hold):
#   1. Assistive autonomy — the AI only assists, does not decide autonomously.
#   2. Informational decision impact — the AI's output is informational,
#      not directly actionable in a consequential way.
#   3. Not replacing human decision — a human makes the final decision.
#
# When all three are met, HIGH downgrades to LIMITED but ALWAYS
# escalates to human review.
# ---------------------------------------------------------------------------

_ART6_3_ASSISTIVE_AUTONOMY_LEVELS: frozenset[AutonomyLevel] = frozenset({
    AutonomyLevel.ADVISORY_ONLY,
    AutonomyLevel.HUMAN_IN_THE_LOOP,
})

# Keywords in decision_impact that suggest informational-only output.
_INFORMATIONAL_KEYWORDS: frozenset[str] = frozenset({
    "informational",
    "advisory",
    "recommendation",
    "suggests",
    "suggests only",
    "non-binding",
    "supplementary",
    "assistive",
    "preparatory",
})


def _check_article6_3_exception(
    profile: FeatureProfile,
    has_high_risk: bool,
) -> DowngradeInfo | None:
    """Check whether Article 6(3) exception applies to downgrade HIGH → LIMITED.

    Returns DowngradeInfo if all three conditions are met, None otherwise.
    Only evaluated when the profile has triggered at least one HIGH rule.
    """
    if not has_high_risk:
        return None

    conditions_met: list[str] = []

    # Condition 1: Assistive autonomy
    if profile.autonomy_level.value in _ART6_3_ASSISTIVE_AUTONOMY_LEVELS:
        conditions_met.append(
            f"assistive_autonomy: autonomy_level={profile.autonomy_level.value.value}"
        )

    # Condition 2: Informational decision impact
    impact_lower = profile.decision_impact.value.lower()
    if any(kw in impact_lower for kw in _INFORMATIONAL_KEYWORDS):
        conditions_met.append(
            "informational_impact: decision_impact contains informational keywords"
        )

    # Condition 3: Not replacing human decision (same as condition 1 for PoC,
    # but explicitly checked — in production this would be a separate field)
    if profile.autonomy_level.value != AutonomyLevel.FULL_AUTONOMY:
        conditions_met.append(
            "human_decides: autonomy_level is not FULL_AUTONOMY"
        )

    # All three conditions must be met
    if len(conditions_met) == 3:
        return DowngradeInfo(
            original_tier=RiskTier.HIGH,
            downgraded_tier=RiskTier.LIMITED,
            article_ref="Art. 6(3)",
            conditions_met=conditions_met,
            rationale=(
                "All three Article 6(3) exception conditions are met: "
                "assistive autonomy, informational decision impact, "
                "and human retains final decision authority. "
                "Downgrade from HIGH to LIMITED with mandatory human review."
            ),
            requires_human_review=True,
        )

    return None


# ---------------------------------------------------------------------------
# Article 50 — GenAI transparency obligations → LIMITED
# ---------------------------------------------------------------------------

_ART50_GENAI_RULE = "ART50_GENAI_TRANSPARENCY"
_ART50_DEEPFAKE_RULE = "ART50_DEEPFAKE_TRANSPARENCY"
_ART50_INTERACTION_RULE = "ART50_HUMAN_INTERACTION"


def _evaluate_article50(profile: FeatureProfile) -> list[RuleEval]:
    """Evaluate Article 50 transparency-obligation rules.

    These apply to GenAI systems that generate content, produce deepfakes,
    or interact directly with natural persons.
    """
    results: list[RuleEval] = []

    # GenAI content generation
    generates = profile.generates_content.value
    results.append(
        RuleEval(
            rule_id=_ART50_GENAI_RULE,
            matched=generates,
            tier=RiskTier.LIMITED,
            article_ref="Art. 50(1)",
            verbatim_provision=(
                "Providers shall ensure that AI systems intended to interact "
                "directly with natural persons are designed and developed in "
                "such a way that natural persons are informed that they are "
                "interacting with an AI system, unless this is obvious"
            ),
            triggered_signals=["generates_content"] if generates else [],
            rationale=(
                "Feature generates content — Art. 50 transparency obligation applies"
                if generates
                else "Feature does not generate content"
            ),
        )
    )

    # Deepfake generation
    deepfakes = profile.generates_deepfakes.value
    results.append(
        RuleEval(
            rule_id=_ART50_DEEPFAKE_RULE,
            matched=deepfakes,
            tier=RiskTier.LIMITED,
            article_ref="Art. 50(4)",
            verbatim_provision=(
                "Deployers of an AI system that generates or manipulates image, "
                "audio or video content constituting a deep fake, shall disclose "
                "that the content has been artificially generated or manipulated"
            ),
            triggered_signals=["generates_deepfakes"] if deepfakes else [],
            rationale=(
                "Feature generates deepfakes — Art. 50(4) disclosure obligation applies"
                if deepfakes
                else "Feature does not generate deepfakes"
            ),
        )
    )

    # Direct human interaction
    interacts = profile.interacts_with_humans.value
    results.append(
        RuleEval(
            rule_id=_ART50_INTERACTION_RULE,
            matched=interacts,
            tier=RiskTier.LIMITED,
            article_ref="Art. 50(1)",
            verbatim_provision=(
                "Providers shall ensure that AI systems intended to interact "
                "directly with natural persons are designed and developed in "
                "such a way that the natural person is informed that they are "
                "interacting with an AI system"
            ),
            triggered_signals=["interacts_with_humans"] if interacts else [],
            rationale=(
                "Feature interacts with humans — Art. 50(1) transparency obligation applies"
                if interacts
                else "Feature does not interact directly with humans"
            ),
        )
    )

    return results


# ---------------------------------------------------------------------------
# Obligation mapping
# ---------------------------------------------------------------------------

_OBLIGATIONS: dict[RiskTier, list[str]] = {
    RiskTier.PROHIBITED: [
        "System is prohibited under Art. 5 and must NOT be placed on the market or put into service",
        "No derogations available for this prohibition category",
    ],
    RiskTier.HIGH: [
        "Risk management system required (Art. 9)",
        "Data governance and data quality requirements (Art. 10)",
        "Technical documentation required (Art. 11)",
        "Record-keeping / automatic logging required (Art. 12)",
        "Transparency and information to deployers required (Art. 13)",
        "Human oversight measures required (Art. 14)",
        "Accuracy, robustness, and cybersecurity requirements (Art. 15)",
        "Quality management system required (Art. 17)",
        "Conformity assessment required before placing on market (Art. 43)",
        "EU declaration of conformity required (Art. 47)",
        "CE marking required (Art. 48)",
        "Registration in EU database required (Art. 49)",
        "Post-market monitoring required (Art. 72)",
        "Serious incident reporting required (Art. 73)",
    ],
    RiskTier.LIMITED: [
        "Transparency obligation: users must be informed they are interacting with AI (Art. 50)",
        "Content generated by AI must be labelled as such (Art. 50)",
        "If deepfake: must disclose artificial generation/manipulation (Art. 50(4))",
    ],
    RiskTier.MINIMAL: [
        "No mandatory requirements under the AI Act",
        "Voluntary codes of conduct encouraged (Art. 95)",
    ],
}


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------


def classify(
    profile: FeatureProfile,
    iso_controls: list[ISOControl] | None = None,
) -> Classification:
    """Run the deterministic rule engine against an extracted FeatureProfile.

    Args:
        profile: The LLM-extracted feature profile (read-only).
        iso_controls: Pre-computed ISO 42001 controls. If None, an empty
            list is used (caller is expected to run iso_42001.map_controls
            separately and pass the result).

    Returns:
        A complete Classification with every rule evaluated, citations,
        obligations, and ISO control mappings.
    """
    all_evals: list[RuleEval] = []
    human_review_reasons: list[str] = []

    # --- Phase 1: Article 5 (prohibited) — evaluated first, short-circuits ---
    art5_evals = _evaluate_article5(profile)
    all_evals.extend(art5_evals)

    # --- Phase 2: Annex III + Article 6 (high-risk) ---
    annex3_evals = _evaluate_annex3(profile)
    all_evals.extend(annex3_evals)

    # --- Phase 3: Article 50 (limited / transparency) ---
    art50_evals = _evaluate_article50(profile)
    all_evals.extend(art50_evals)

    # --- Collect matched rules and determine raw tier ---
    triggered = [e for e in all_evals if e.matched]
    triggered_tiers = [e.tier for e in triggered]
    raw_tier = _highest_tier(triggered_tiers)

    # --- Phase 4: Article 6(3) exception check ---
    # Only if raw tier is HIGH (not PROHIBITED — prohibited cannot be downgraded).
    downgrade: DowngradeInfo | None = None
    if raw_tier == RiskTier.HIGH:
        downgrade = _check_article6_3_exception(
            profile, has_high_risk=True
        )
        if downgrade is not None:
            human_review_reasons.append(
                "Art. 6(3) exception applied — downgrade from HIGH to LIMITED "
                "requires mandatory human review"
            )

    # --- Resolve final tier ---
    final_tier = raw_tier
    if downgrade is not None:
        final_tier = downgrade.downgraded_tier

    # --- Build citations from triggered rules ---
    citations: list[Citation] = []
    seen_refs: set[str] = set()
    for rule in triggered:
        if rule.article_ref not in seen_refs:
            citations.append(
                Citation(
                    article_ref=rule.article_ref,
                    verbatim_text=rule.verbatim_provision,
                )
            )
            seen_refs.add(rule.article_ref)

    # --- Determine obligations ---
    obligations = list(_OBLIGATIONS.get(final_tier, []))

    # --- Human review triggers ---
    if final_tier == RiskTier.PROHIBITED:
        human_review_reasons.append(
            "PROHIBITED classification — requires immediate human review"
        )
    if final_tier == RiskTier.HIGH:
        human_review_reasons.append(
            "HIGH-risk classification — human oversight required per Art. 14"
        )
    requires_human_review = len(human_review_reasons) > 0

    return Classification(
        final_tier=final_tier,
        triggered_rules=triggered,
        all_rules_evaluated=all_evals,
        triggered_citations=citations,
        obligations=obligations,
        iso_controls=iso_controls or [],
        requires_human_review=requires_human_review,
        human_review_reasons=human_review_reasons,
        downgrade=downgrade,
        feature_name=profile.feature_name.value,
        feature_description=profile.description.value,
        domain=profile.domain.value,
    )
