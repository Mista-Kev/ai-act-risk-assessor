"""Pydantic v2 models, enums, and controlled vocabularies for the AI Act Risk Assessor.

Every data shape in the pipeline is defined here. LLM extraction targets,
rule engine outputs, audit records, and verification results all share
these types. Immutability (frozen=True) is enforced on all models that
flow through the pipeline to guarantee audit integrity.

Design decision — Instructor over Outlines:
  Instructor integrates with Ollama's OpenAI-compatible API, has first-class
  Pydantic v2 support, and enforces schema compliance through JSON mode +
  validation retries. Outlines would require direct model access (transformers
  backend), defeating the purpose of Ollama. For a PoC with temperature=0,
  Instructor's API-level constraint is sufficient. Production could layer
  Outlines for token-level enforcement — the schema stays identical.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfidenceLevel(str, Enum):
    """How the extractor arrived at this value.

    Example::

        >>> ConfidenceLevel.EXPLICIT
        <ConfidenceLevel.EXPLICIT: 'explicit'>
    """

    EXPLICIT = "explicit"
    INFERRED = "inferred"
    UNCLEAR = "unclear"


class RiskTier(str, Enum):
    """EU AI Act risk classification tiers, ordered by severity.

    Example::

        >>> RiskTier.HIGH
        <RiskTier.HIGH: 'high'>
    """

    PROHIBITED = "prohibited"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


class AutonomyLevel(str, Enum):
    """Degree of human oversight in AI-assisted decisions.

    Example::

        >>> AutonomyLevel.ADVISORY_ONLY
        <AutonomyLevel.ADVISORY_ONLY: 'advisory_only'>
    """

    FULL_AUTONOMY = "full_autonomy"
    HUMAN_ON_THE_LOOP = "human_on_the_loop"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"
    ADVISORY_ONLY = "advisory_only"


class DataSensitivity(str, Enum):
    """Categories of data sensitivity relevant to AI Act classification.

    Example::

        >>> DataSensitivity.BIOMETRIC
        <DataSensitivity.BIOMETRIC: 'biometric'>
    """

    BIOMETRIC = "biometric"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"
    ANONYMIZED = "anonymized"
    AGGREGATED = "aggregated"
    PUBLIC = "public"


class DeploymentScope(str, Enum):
    """Whether the system operates in public or private contexts.

    Example::

        >>> DeploymentScope.ONLINE_ONLY
        <DeploymentScope.ONLINE_ONLY: 'online_only'>
    """

    PUBLIC_SPACE = "public_space"
    PRIVATE_SPACE = "private_space"
    MIXED = "mixed"
    ONLINE_ONLY = "online_only"


# ---------------------------------------------------------------------------
# Controlled Vocabularies
# ---------------------------------------------------------------------------
# Each key is a signal token the extractor may emit.
# Value is (article_ref, verbatim_provision_text).
# The rule engine uses keys for membership tests; values populate audit records.
# These are the ONLY signals the extractor is allowed to produce.
# ---------------------------------------------------------------------------


ARTICLE_5_VOCAB: dict[str, tuple[str, str]] = {
    # (a) Subliminal / manipulative / deceptive techniques
    "subliminal_technique": (
        "Art. 5(1)(a)",
        "AI system that deploys subliminal techniques beyond a person's "
        "consciousness to materially distort behaviour causing significant harm",
    ),
    "manipulative_technique": (
        "Art. 5(1)(a)",
        "AI system that deploys purposefully manipulative techniques to "
        "materially distort behaviour causing significant harm",
    ),
    "deceptive_technique": (
        "Art. 5(1)(a)",
        "AI system that deploys deceptive techniques materially distorting "
        "behaviour and causing significant harm",
    ),
    # (b) Exploiting vulnerabilities
    "exploit_age_vulnerability": (
        "Art. 5(1)(b)",
        "AI system that exploits vulnerabilities of a person due to their age "
        "to materially distort behaviour causing significant harm",
    ),
    "exploit_disability_vulnerability": (
        "Art. 5(1)(b)",
        "AI system that exploits vulnerabilities of a person due to disability "
        "to materially distort behaviour causing significant harm",
    ),
    "exploit_socioeconomic_vulnerability": (
        "Art. 5(1)(b)",
        "AI system that exploits vulnerabilities of a person due to social or "
        "economic situation to materially distort behaviour causing significant harm",
    ),
    # (c) Social scoring
    "social_scoring": (
        "Art. 5(1)(c)",
        "AI system that evaluates or classifies natural persons based on their "
        "social behaviour or personal characteristics, leading to detrimental "
        "treatment in unrelated contexts or disproportionate to the gravity "
        "of the social behaviour",
    ),
    # (d) Predictive policing based solely on profiling
    "predictive_policing_profiling": (
        "Art. 5(1)(d)",
        "AI system that makes risk assessments of natural persons to assess or "
        "predict the risk of criminal offence solely based on profiling or "
        "assessment of personality traits and characteristics",
    ),
    # (e) Untargeted facial image scraping
    "untargeted_facial_scraping": (
        "Art. 5(1)(e)",
        "AI system that creates or expands facial recognition databases through "
        "untargeted scraping of facial images from the internet or CCTV footage",
    ),
    # (f) Emotion recognition in workplace/education
    "emotion_recognition_workplace": (
        "Art. 5(1)(f)",
        "AI system that infers emotions of a natural person in the workplace "
        "except where intended for medical or safety reasons",
    ),
    "emotion_recognition_education": (
        "Art. 5(1)(f)",
        "AI system that infers emotions of a natural person in educational "
        "institutions except where intended for medical or safety reasons",
    ),
    # (g) Biometric categorisation inferring sensitive attributes
    "biometric_categorization_race": (
        "Art. 5(1)(g)",
        "AI system that categorises natural persons individually based on "
        "biometric data to deduce or infer their race",
    ),
    "biometric_categorization_politics": (
        "Art. 5(1)(g)",
        "AI system that categorises natural persons individually based on "
        "biometric data to deduce or infer their political opinions",
    ),
    "biometric_categorization_union": (
        "Art. 5(1)(g)",
        "AI system that categorises natural persons individually based on "
        "biometric data to deduce or infer trade union membership",
    ),
    "biometric_categorization_religion": (
        "Art. 5(1)(g)",
        "AI system that categorises natural persons individually based on "
        "biometric data to deduce or infer religious or philosophical beliefs",
    ),
    "biometric_categorization_sex_life": (
        "Art. 5(1)(g)",
        "AI system that categorises natural persons individually based on "
        "biometric data to deduce or infer sex life or sexual orientation",
    ),
    # (h) Real-time remote biometric identification in public spaces
    "realtime_remote_biometric_id": (
        "Art. 5(1)(h)",
        "Real-time remote biometric identification system in publicly accessible "
        "spaces for the purpose of law enforcement except where strictly necessary "
        "for targeted search, prevention of threats, or criminal investigation "
        "under judicial authorisation",
    ),
}

ANNEX_III_VOCAB: dict[str, tuple[str, str]] = {
    # 1. Biometrics
    "remote_biometric_identification": (
        "Annex III, 1(a)",
        "AI systems intended to be used for remote biometric identification "
        "of natural persons, not including verification authenticating identity",
    ),
    "biometric_categorization": (
        "Annex III, 1(b)",
        "AI systems intended to be used for biometric categorisation according "
        "to sensitive or protected attributes or characteristics",
    ),
    "emotion_recognition": (
        "Annex III, 1(c)",
        "AI systems intended to be used for emotion recognition",
    ),
    # 2. Critical infrastructure
    "critical_infrastructure_safety": (
        "Annex III, 2(a)",
        "AI systems intended to be used as safety components in the management "
        "and operation of critical digital infrastructure, road traffic, or "
        "supply of water, gas, heating and electricity",
    ),
    # 3. Education and vocational training
    "education_access_admission": (
        "Annex III, 3(a)",
        "AI systems intended to be used for determining access or admission "
        "or assigning persons to educational and vocational training institutions",
    ),
    "education_outcome_evaluation": (
        "Annex III, 3(b)",
        "AI systems intended to be used for evaluating learning outcomes, "
        "including when used to steer the learning process",
    ),
    "education_level_assessment": (
        "Annex III, 3(c)",
        "AI systems intended to be used for assessing the appropriate level "
        "of education that an individual will receive or be able to access",
    ),
    "education_cheating_detection": (
        "Annex III, 3(d)",
        "AI systems intended to be used for monitoring and detecting prohibited "
        "behaviour of students during tests",
    ),
    # 4. Employment, workers management, self-employment
    "employment_recruitment": (
        "Annex III, 4(a)",
        "AI systems intended to be used for recruitment or selection of natural "
        "persons, in particular for targeted job advertisements, screening or "
        "filtering applications, and evaluating candidates",
    ),
    "employment_decisions": (
        "Annex III, 4(b)",
        "AI systems intended to be used for making decisions affecting terms of "
        "work-related relationships, promotion, termination, task allocation based "
        "on individual behaviour, or monitoring and evaluating performance and "
        "behaviour of persons in such relationships",
    ),
    # 5. Access to essential private and public services
    "creditworthiness_assessment": (
        "Annex III, 5(a)",
        "AI systems intended to be used to evaluate the creditworthiness of "
        "natural persons or establish their credit score, except for detecting "
        "financial fraud",
    ),
    "insurance_risk_assessment": (
        "Annex III, 5(b)",
        "AI systems intended to be used for risk assessment and pricing in "
        "relation to natural persons in the case of life and health insurance",
    ),
    "public_benefits_eligibility": (
        "Annex III, 5(c)",
        "AI systems intended to be used to evaluate eligibility of natural "
        "persons for public assistance benefits and services, to grant, reduce, "
        "revoke or reclaim such benefits and services",
    ),
    "emergency_dispatch": (
        "Annex III, 5(d)",
        "AI systems intended to be used to evaluate and classify emergency calls "
        "or to dispatch or establish priority for emergency first response services",
    ),
    # 6. Law enforcement
    "law_enforcement_risk_victim": (
        "Annex III, 6(a)",
        "AI systems intended to be used to assess the risk of a natural person "
        "becoming the victim of criminal offences",
    ),
    "law_enforcement_polygraph": (
        "Annex III, 6(b)",
        "AI systems intended to be used as polygraphs or similar tools to detect "
        "the deception of a natural person",
    ),
    "law_enforcement_evidence_reliability": (
        "Annex III, 6(c)",
        "AI systems intended to be used to evaluate the reliability of evidence "
        "in the course of investigation or prosecution of criminal offences",
    ),
    "law_enforcement_risk_offending": (
        "Annex III, 6(d)",
        "AI systems intended to be used for assessing the risk of a natural "
        "person for offending or re-offending, not solely based on profiling",
    ),
    "law_enforcement_profiling": (
        "Annex III, 6(e)",
        "AI systems intended to be used for profiling of natural persons in the "
        "course of detection, investigation or prosecution of criminal offences",
    ),
    # 7. Migration, asylum, border control
    "migration_polygraph": (
        "Annex III, 7(a)",
        "AI systems intended to be used as polygraphs or similar tools to detect "
        "the deception of natural persons in the context of migration, asylum "
        "and border control management",
    ),
    "migration_risk_assessment": (
        "Annex III, 7(b)",
        "AI systems intended to be used to assess risks including security, "
        "irregular migration or health risks posed by a natural person intending "
        "to enter or having entered the territory of a Member State",
    ),
    "migration_application_evaluation": (
        "Annex III, 7(c)",
        "AI systems intended to be used to assist competent public authorities "
        "for the examination of applications for asylum, visa and residence permits "
        "and associated complaints regarding the eligibility of the applicants",
    ),
    "migration_person_detection": (
        "Annex III, 7(d)",
        "AI systems intended to be used for detecting, recognising or identifying "
        "natural persons in the context of migration, asylum and border control "
        "management, with the exception of verification of travel documents",
    ),
    # 8. Administration of justice and democratic processes
    "justice_research_application": (
        "Annex III, 8(a)",
        "AI systems intended to be used by a judicial authority or on their behalf "
        "to assist in researching and interpreting facts and the law and in "
        "applying the law to a concrete set of facts",
    ),
    "democratic_process_influence": (
        "Annex III, 8(b)",
        "AI systems intended to be used for influencing the outcome of an election "
        "or referendum or the voting behaviour of natural persons in the exercise "
        "of their vote, excluding AI systems whose output does not directly "
        "interact with natural persons",
    ),
}

# Frozen sets for fast membership testing in the rule engine.
ARTICLE_5_SIGNALS: frozenset[str] = frozenset(ARTICLE_5_VOCAB.keys())
ANNEX_III_SIGNALS: frozenset[str] = frozenset(ANNEX_III_VOCAB.keys())


# ---------------------------------------------------------------------------
# Generic extraction wrapper
# ---------------------------------------------------------------------------

T = TypeVar("T")


class ExtractedField(BaseModel, Generic[T]):
    """Wrapper for every LLM-extracted field, tracking provenance.

    The LLM fills these; the verifier audits them. If ``source_span`` is
    empty, ``confidence`` must be ``unclear`` — enforced by the verifier,
    not by Pydantic, to keep the LLM's JSON schema simple.

    Example::

        >>> ExtractedField[str](value="credit scoring", source_span="credit scoring engine", confidence=ConfidenceLevel.EXPLICIT)
        ExtractedField(value='credit scoring', ...)
    """

    model_config = ConfigDict(frozen=True)

    value: T = Field(description="The extracted value.")
    source_span: str = Field(
        default="",
        description=(
            "Verbatim substring from the input text that supports this value. "
            "Empty string if the value was inferred rather than directly quoted."
        ),
    )
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.UNCLEAR,
        description="How the extractor arrived at this value.",
    )


# ---------------------------------------------------------------------------
# Feature Profile (extraction target)
# ---------------------------------------------------------------------------


class FeatureProfile(BaseModel):
    """Complete structured extraction from a natural-language AI feature description.

    This is the ONLY model the extractor LLM populates.
    The rule engine and all downstream modules consume it read-only.
    The extractor does NOT classify — it extracts.

    Example::

        >>> profile = FeatureProfile(
        ...     feature_name=ExtractedField[str](value="CV Screener", source_span="CV Screener", confidence=ConfidenceLevel.EXPLICIT),
        ...     # ... remaining fields ...
        ... )
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ---
    feature_name: ExtractedField[str] = Field(
        description="Short name of the AI feature being assessed."
    )
    description: ExtractedField[str] = Field(
        description="One-paragraph summary of what the AI feature does."
    )
    domain: ExtractedField[str] = Field(
        description=(
            "Application domain, e.g. 'healthcare', 'finance', 'education', "
            "'law_enforcement', 'hr_recruitment', 'social_media'."
        ),
    )

    # --- Affected parties ---
    affected_subjects: ExtractedField[list[str]] = Field(
        description=(
            "Categories of natural persons affected by the AI system's outputs, "
            "e.g. ['job_applicants', 'patients', 'students', 'citizens']."
        ),
    )
    operators: ExtractedField[list[str]] = Field(
        description=(
            "Entities deploying or operating the AI system, "
            "e.g. ['employer', 'hospital', 'government_agency']."
        ),
    )

    # --- Autonomy and impact ---
    autonomy_level: ExtractedField[AutonomyLevel] = Field(
        description="Degree of human oversight in the AI-assisted decision.",
    )
    decision_impact: ExtractedField[str] = Field(
        description=(
            "What decision or outcome the AI influences and its potential "
            "consequence on affected subjects."
        ),
    )

    # --- Data characteristics ---
    data_types: ExtractedField[list[DataSensitivity]] = Field(
        description="Categories of data the AI system processes.",
    )
    uses_biometric_data: ExtractedField[bool] = Field(
        description="Whether the system processes biometric data (face, voice, gait, etc.).",
    )
    uses_personal_data: ExtractedField[bool] = Field(
        description="Whether the system processes personal data under GDPR.",
    )

    # --- Deployment context ---
    deployment_scope: ExtractedField[DeploymentScope] = Field(
        description="Whether the system operates in public, private, or mixed spaces.",
    )
    sector: ExtractedField[str] = Field(
        description=(
            "Sector of deployment, e.g. 'public_sector', 'private_sector', "
            "'healthcare', 'finance', 'education', 'law_enforcement'."
        ),
    )

    # --- AI Act signals (critical for rule engine) ---
    prohibited_signals: ExtractedField[list[str]] = Field(
        description=(
            "Signal tokens from ARTICLE_5_VOCAB that apply to this feature. "
            "Empty list if none apply. Valid tokens: "
            + ", ".join(sorted(ARTICLE_5_VOCAB.keys()))
        ),
    )
    high_risk_signals: ExtractedField[list[str]] = Field(
        description=(
            "Signal tokens from ANNEX_III_VOCAB that apply to this feature. "
            "Empty list if none apply. Valid tokens: "
            + ", ".join(sorted(ANNEX_III_VOCAB.keys()))
        ),
    )

    # --- GenAI indicators (for Limited-risk / Art. 50 transparency) ---
    generates_content: ExtractedField[bool] = Field(
        description="Whether the system generates text, images, audio, video, or code.",
    )
    interacts_with_humans: ExtractedField[bool] = Field(
        description=(
            "Whether the system interacts directly with natural persons "
            "(chatbot, voice assistant, etc.)."
        ),
    )
    generates_deepfakes: ExtractedField[bool] = Field(
        description=(
            "Whether the system generates or manipulates content that could be "
            "mistaken for authentic (deepfakes)."
        ),
    )


# ---------------------------------------------------------------------------
# Rule evaluation and citations
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A precise reference to an EU AI Act or ISO 42001 provision.

    Example::

        >>> Citation(article_ref="Art. 5(1)(c)", verbatim_text="AI system that evaluates...")
    """

    model_config = ConfigDict(frozen=True)

    article_ref: str = Field(
        description="Article or Annex reference, e.g. 'Art. 5(1)(a)' or 'Annex III, 4(b)'.",
    )
    verbatim_text: str = Field(
        description="Verbatim text of the provision at the time of assessment.",
    )


class RuleEval(BaseModel):
    """Result of evaluating one deterministic rule against a FeatureProfile.

    Produced only by the rule engine, never by an LLM. Every rule emits one
    of these whether it matched or not, so the audit trail records the entire
    decision path.

    Example::

        >>> RuleEval(rule_id="ART5_C_SOCIAL_SCORING", matched=True, tier=RiskTier.PROHIBITED,
        ...          article_ref="Art. 5(1)(c)", verbatim_provision="...", triggered_signals=["social_scoring"])
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(
        description="Unique rule identifier, e.g. 'ART5_A_SUBLIMINAL'.",
    )
    matched: bool = Field(
        description="Whether this rule's conditions were satisfied.",
    )
    tier: RiskTier = Field(
        description="The risk tier this rule asserts when matched.",
    )
    article_ref: str = Field(
        description="EU AI Act article or annex reference.",
    )
    verbatim_provision: str = Field(
        description="Verbatim provision text at assessment time.",
    )
    triggered_signals: list[str] = Field(
        default_factory=list,
        description="Signal tokens from the feature profile that caused this match.",
    )
    rationale: str = Field(
        default="",
        description="Brief explanation of why this rule matched or did not.",
    )


# ---------------------------------------------------------------------------
# ISO 42001 controls
# ---------------------------------------------------------------------------


class ISOControl(BaseModel):
    """A single ISO/IEC 42001:2023 Annex A control mapped to a classification.

    Example::

        >>> ISOControl(control_id="A.5.2", title="AI Risk Assessment",
        ...            description="...", applicability="High-risk system per Art. 9")
    """

    model_config = ConfigDict(frozen=True)

    control_id: str = Field(description="ISO 42001 control ID, e.g. 'A.5.2'.")
    title: str = Field(description="Control title.")
    description: str = Field(description="What the control requires.")
    applicability: str = Field(description="Why this control applies to the assessed feature.")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class DowngradeInfo(BaseModel):
    """Records when Article 6(3) exception downgrades HIGH to LIMITED.

    Example::

        >>> DowngradeInfo(original_tier=RiskTier.HIGH, downgraded_tier=RiskTier.LIMITED,
        ...               article_ref="Art. 6(3)", conditions_met=["assistive_autonomy", ...])
    """

    model_config = ConfigDict(frozen=True)

    original_tier: RiskTier
    downgraded_tier: RiskTier
    article_ref: str = Field(
        default="Art. 6(3)",
        description="Article reference for the exception.",
    )
    conditions_met: list[str] = Field(
        description="Which Article 6(3) conditions were satisfied.",
    )
    rationale: str = Field(
        description="Why the downgrade was applied.",
    )
    requires_human_review: bool = Field(
        default=True,
        description="Art. 6(3) downgrades always escalate to human review.",
    )


class Classification(BaseModel):
    """Complete classification result from the rule engine + ISO mapper.

    Assembled by the pipeline orchestrator. The drafter sees only this,
    never the original input text.

    Example::

        >>> Classification(final_tier=RiskTier.HIGH, triggered_rules=[...],
        ...                all_rules_evaluated=[...], triggered_citations=[...],
        ...                obligations=[...], iso_controls=[...], requires_human_review=True)
    """

    model_config = ConfigDict(frozen=True)

    final_tier: RiskTier = Field(
        description="The highest risk tier triggered by any matched rule.",
    )
    triggered_rules: list[RuleEval] = Field(
        description="Rules that matched (matched=True), ordered by tier severity.",
    )
    all_rules_evaluated: list[RuleEval] = Field(
        description="Every rule evaluated, including non-matches, for full audit trail.",
    )
    triggered_citations: list[Citation] = Field(
        description="EU AI Act citations from triggered rules.",
    )
    obligations: list[str] = Field(
        description=(
            "Compliance obligations arising from this classification, "
            "e.g. 'Conformity assessment required (Art. 43)'."
        ),
    )
    iso_controls: list[ISOControl] = Field(
        description="ISO 42001 Annex A controls applicable to this feature.",
    )
    requires_human_review: bool = Field(
        description="Whether this classification requires mandatory human review.",
    )
    human_review_reasons: list[str] = Field(
        default_factory=list,
        description="Reasons why human review is required.",
    )
    downgrade: DowngradeInfo | None = Field(
        default=None,
        description="Present only if Art. 6(3) exception was applied.",
    )
    feature_name: str = Field(
        default="",
        description="Feature name carried forward for the drafter's context.",
    )
    feature_description: str = Field(
        default="",
        description="Feature description carried forward for the drafter's context.",
    )
    domain: str = Field(
        default="",
        description="Domain carried forward for the drafter's context.",
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class CheckResult(BaseModel):
    """Result of a single verification check.

    Example::

        >>> CheckResult(check_id="SPAN_EXISTS", passed=True, detail="Span found in input text.")
    """

    model_config = ConfigDict(frozen=True)

    check_id: str = Field(description="Unique check identifier.")
    passed: bool
    detail: str = Field(description="What was checked and the result.")
    severity: Literal["error", "warning", "info"] = Field(default="error")


class VerificationResult(BaseModel):
    """Aggregated verification result for an extraction or drafted memo.

    Example::

        >>> result = VerificationResult(checks=[...], passed=True, escalate=False)
    """

    model_config = ConfigDict(frozen=True)

    checks: list[CheckResult] = Field(description="All individual check results.")
    passed: bool = Field(description="True only if all error-severity checks passed.")
    escalate: bool = Field(
        default=False,
        description="True if issues require human intervention.",
    )
    escalation_reasons: list[str] = Field(
        default_factory=list,
        description="Reasons for escalation.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def error_count(self) -> int:
        """Number of failed error-severity checks."""
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def warning_count(self) -> int:
        """Number of failed warning-severity checks."""
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")


# ---------------------------------------------------------------------------
# Provenance and Audit Record
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Immutable provenance metadata for reproducibility.

    Example::

        >>> Provenance(rule_engine_version="0.1.0", schema_version="0.1.0",
        ...            extractor_model_id="nemotron-3-nano:4b", drafter_model_id="gemma4:e4b",
        ...            extractor_prompt_hash="abc123...", drafter_prompt_hash="def456...")
    """

    model_config = ConfigDict(frozen=True)

    rule_engine_version: str = Field(
        description="Semantic version of the rule engine.",
    )
    schema_version: str = Field(
        description="Semantic version of the schema.",
    )
    git_sha: str = Field(
        default="",
        description="Git commit SHA at assessment time. Empty if not in a git repo.",
    )
    extractor_model_id: str = Field(
        description="Ollama model tag used for extraction.",
    )
    drafter_model_id: str = Field(
        description="Ollama model tag used for drafting.",
    )
    extractor_prompt_hash: str = Field(
        description="SHA-256 of the extractor prompt template.",
    )
    drafter_prompt_hash: str = Field(
        description="SHA-256 of the drafter prompt template.",
    )


class AuditRecord(BaseModel):
    """Complete, immutable, hash-chained audit record for one assessment.

    Serialized to JSON and stored to disk. The hash chain allows tamper
    detection across sequential assessments.

    Canonicalization for hashing: ``model_dump_json()`` with default
    Pydantic v2 serialization (sorted keys via json.dumps, no extra
    whitespace, UTF-8). This is documented here so replay is deterministic.

    Example::

        >>> record = AuditRecord(assessment_id="...", timestamp_utc=datetime.utcnow(), ...)
        >>> sha = record.canonical_sha256()
    """

    model_config = ConfigDict(frozen=True)

    assessment_id: str = Field(
        description="UUID identifying this assessment.",
    )
    timestamp_utc: datetime = Field(
        description="UTC timestamp when the assessment completed.",
    )
    input_text_sha256: str = Field(
        description="SHA-256 hash of the original input text.",
    )
    input_text_length: int = Field(
        description="Character length of the original input text.",
    )
    feature_profile: FeatureProfile = Field(
        description="The extracted feature profile.",
    )
    extraction_verification: VerificationResult = Field(
        description="Verification of the extraction step.",
    )
    classification: Classification = Field(
        description="The complete classification result.",
    )
    memo: str = Field(
        description="LLM-drafted assessment memo in markdown.",
    )
    memo_verification: VerificationResult = Field(
        description="Verification of the drafted memo (citation checks).",
    )
    provenance: Provenance = Field(
        description="Reproducibility metadata.",
    )
    previous_record_sha256: str | None = Field(
        default=None,
        description=(
            "SHA-256 of the previous AuditRecord's canonical JSON. "
            "None for the first record in the chain."
        ),
    )

    def canonical_json(self) -> str:
        """Canonical JSON serialization for hashing.

        Sort keys, no extra whitespace, UTF-8. This is the determinism
        anchor for the hash chain and replay verification.
        """
        data = self.model_dump(mode="json")
        return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def canonical_sha256(self) -> str:
        """SHA-256 of the canonical JSON serialization."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class ReplayCheckResult(BaseModel):
    """Result of re-running one deterministic check against a stored audit record.

    Example::

        >>> ReplayCheckResult(step="rule_engine", passed=True, detail="Classification matches.")
    """

    model_config = ConfigDict(frozen=True)

    step: str = Field(description="Pipeline step being replayed.")
    passed: bool
    detail: str
    expected: str = Field(default="", description="Expected value (for diff display).")
    actual: str = Field(default="", description="Actual value (for diff display).")


class ReplayResult(BaseModel):
    """Overall result of replaying an audit record.

    Example::

        >>> ReplayResult(assessment_id="abc123", checks=[...], overall_passed=True, ...)
    """

    model_config = ConfigDict(frozen=True)

    assessment_id: str
    checks: list[ReplayCheckResult]
    overall_passed: bool = Field(
        description="True only if all replay checks passed.",
    )
    replayed_at: datetime


# ---------------------------------------------------------------------------
# Schema version constant
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "0.1.0"
