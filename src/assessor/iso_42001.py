"""ISO/IEC 42001:2023 Annex A control mapping.

Maps AI Act risk tiers and feature attributes to applicable ISO 42001 controls.
This is a separate module from the AI Act rule engine — the two concerns must
not be conflated. The rule engine determines the risk tier; this module determines
which management system controls apply given that tier and the feature's characteristics.

The mapping is static and deterministic. No LLM calls.
"""

from __future__ import annotations

from assessor.schema import (
    ANNEX_III_SIGNALS,
    Classification,
    FeatureProfile,
    ISOControl,
    RiskTier,
)

# ---------------------------------------------------------------------------
# ISO 42001 Annex A control definitions
# ---------------------------------------------------------------------------
# Subset most relevant to AI Act compliance. Each tuple:
#   (control_id, title, description)
# ---------------------------------------------------------------------------

_CONTROLS: dict[str, tuple[str, str]] = {
    # A.2 — AI Policy
    "A.2.2": (
        "AI Policy",
        "Establish and maintain an AI policy aligned with the organization's objectives, "
        "addressing responsible AI development, deployment and use.",
    ),
    "A.2.3": (
        "Responsible AI Topics",
        "Address responsible AI topics such as fairness, transparency, accountability, "
        "safety and privacy within the AI management system.",
    ),
    # A.3 — Internal Organization
    "A.3.2": (
        "Roles and Responsibilities",
        "Define and assign roles and responsibilities for AI system development, "
        "deployment, monitoring and decommissioning.",
    ),
    "A.3.3": (
        "Reporting Concerns",
        "Establish mechanisms for reporting AI-related concerns, incidents and "
        "potential harms within and outside the organization.",
    ),
    # A.4 — Resources
    "A.4.4": (
        "Awareness of AI Policy",
        "Ensure that persons working under the organization's control are aware "
        "of the AI policy and their contribution to the AI management system.",
    ),
    "A.4.5": (
        "Competence of Personnel",
        "Determine and ensure necessary competence of persons doing work under "
        "the organization's control that affects AI system performance.",
    ),
    # A.5 — Assessing Impacts of AI Systems
    "A.5.2": (
        "AI System Risk Assessment",
        "Identify, analyse and evaluate risks associated with AI systems, "
        "considering impacts on individuals, groups and society.",
    ),
    "A.5.3": (
        "AI System Impact Assessment",
        "Assess the potential impacts of AI systems on individuals, groups, "
        "society and the environment before deployment.",
    ),
    "A.5.4": (
        "Documentation of Risk and Impact Assessments",
        "Document and maintain records of AI risk and impact assessments "
        "with sufficient detail for review and audit.",
    ),
    # A.6 — AI System Life Cycle
    "A.6.2.2": (
        "Design and Development",
        "Establish processes for AI system design and development that address "
        "requirements, specifications and responsible AI objectives.",
    ),
    "A.6.2.3": (
        "Training and Testing Data",
        "Ensure appropriate selection, preparation and management of data "
        "used for training, validation and testing of AI systems.",
    ),
    "A.6.2.4": (
        "Verification and Validation",
        "Verify and validate AI systems to ensure they meet requirements "
        "and perform as intended within acceptable boundaries.",
    ),
    "A.6.2.6": (
        "Operation and Monitoring",
        "Establish processes for operating AI systems including monitoring "
        "performance, drift detection and incident response.",
    ),
    "A.6.2.9": (
        "AI System Documentation",
        "Maintain comprehensive documentation of AI systems including purpose, "
        "capabilities, limitations and design decisions.",
    ),
    "A.6.2.10": (
        "Defined Use and Misuse",
        "Define and document intended use, foreseeable misuse and conditions "
        "under which the AI system should not be used.",
    ),
    "A.6.2.11": (
        "Third-Party AI Components",
        "Manage risks associated with third-party AI components, models, "
        "data and services used in AI systems.",
    ),
    # A.7 — Data for AI Systems
    "A.7.2": (
        "Data for Development and Enhancement",
        "Manage data used for developing and enhancing AI systems with "
        "appropriate governance, quality and provenance controls.",
    ),
    "A.7.3": (
        "Data Quality",
        "Establish and maintain data quality measures for data used in "
        "AI systems, including accuracy, completeness and timeliness.",
    ),
    "A.7.5": (
        "Data Acquisition",
        "Ensure data acquisition for AI systems is conducted ethically, "
        "lawfully and with appropriate consent or legal basis.",
    ),
    # A.8 — Information for Interested Parties
    "A.8.2": (
        "Informing About AI Interaction",
        "Inform natural persons when they are interacting with an AI system "
        "or when AI-generated content is presented to them.",
    ),
    "A.8.3": (
        "Informing About AI Outcomes",
        "Inform relevant parties about AI system outcomes, including the "
        "basis for decisions and available recourse mechanisms.",
    ),
    "A.8.5": (
        "Enabling Human Actions",
        "Enable natural persons to take appropriate actions in response to "
        "AI system outputs, including options to contest or override decisions.",
    ),
    # A.9 — Use of AI Systems
    "A.9.2": (
        "Responsible Use Objectives",
        "Establish objectives for the responsible use of AI systems aligned "
        "with organizational values and societal expectations.",
    ),
    "A.9.3": (
        "Intended Use Documentation",
        "Document and communicate the intended use of AI systems to all "
        "relevant stakeholders.",
    ),
    "A.9.5": (
        "Human Oversight of AI Systems",
        "Implement human oversight measures proportionate to the AI system's "
        "risk level, ensuring meaningful human control.",
    ),
    # A.10 — Third-Party Relationships
    "A.10.2": (
        "Supplier AI Component Management",
        "Manage risks from supplier-provided AI components through appropriate "
        "due diligence, contractual requirements and monitoring.",
    ),
    "A.10.3": (
        "Shared ML Models",
        "Manage risks from shared or pre-trained ML models, including "
        "provenance verification and fitness-for-purpose assessment.",
    ),
    "A.10.4": (
        "Provision of AI to Third Parties",
        "Manage risks when providing AI systems or components to third parties, "
        "including documentation and support obligations.",
    ),
}


def _make_control(control_id: str, applicability: str) -> ISOControl:
    """Create an ISOControl from the static definitions."""
    title, description = _CONTROLS[control_id]
    return ISOControl(
        control_id=control_id,
        title=title,
        description=description,
        applicability=applicability,
    )


# ---------------------------------------------------------------------------
# Tier-based control mapping
# ---------------------------------------------------------------------------

# Controls applicable to ALL tiers.
_UNIVERSAL_CONTROLS: list[tuple[str, str]] = [
    ("A.2.2", "AI policy required for all AI systems regardless of risk tier"),
    ("A.2.3", "Responsible AI topics must be addressed for all AI systems"),
    ("A.9.3", "Intended use must be documented for all AI systems"),
]

# Additional controls for HIGH-risk and above.
_HIGH_RISK_CONTROLS: list[tuple[str, str]] = [
    ("A.5.2", "High-risk system requires documented risk assessment per AI Act Art. 9"),
    ("A.5.3", "High-risk system requires impact assessment before deployment"),
    ("A.5.4", "Risk and impact assessment documentation required for audit trail"),
    ("A.6.2.2", "Design and development process required per AI Act Art. 9-15"),
    ("A.6.2.3", "Training and testing data governance required per AI Act Art. 10"),
    ("A.6.2.4", "Verification and validation required per AI Act Art. 9"),
    ("A.6.2.6", "Operational monitoring required per AI Act Art. 9, 72"),
    ("A.6.2.9", "Technical documentation required per AI Act Art. 11"),
    ("A.6.2.10", "Defined use and misuse documentation required per AI Act Art. 13"),
    ("A.7.3", "Data quality governance required per AI Act Art. 10"),
    ("A.8.2", "Users must be informed about AI interaction per AI Act Art. 13"),
    ("A.8.3", "Deployers must be informed about AI outcomes per AI Act Art. 13"),
    ("A.8.5", "Human override capability required per AI Act Art. 14"),
    ("A.9.5", "Human oversight measures required per AI Act Art. 14"),
    ("A.10.2", "Supplier component due diligence required per AI Act Art. 25"),
]

# Additional controls for LIMITED-risk (transparency).
_LIMITED_CONTROLS: list[tuple[str, str]] = [
    ("A.8.2", "Transparency obligation: inform users about AI interaction per Art. 50"),
    ("A.8.3", "Transparency obligation: inform about AI-generated outcomes per Art. 50"),
    ("A.4.4", "Staff awareness of transparency obligations under Art. 50"),
]

# Additional controls for PROHIBITED systems (documentation of the prohibition).
_PROHIBITED_CONTROLS: list[tuple[str, str]] = [
    ("A.5.2", "Risk assessment must document why the system is prohibited under Art. 5"),
    ("A.5.3", "Impact assessment required to document prohibition decision"),
    ("A.3.3", "Reporting mechanism needed to flag prohibited AI practices"),
]


# ---------------------------------------------------------------------------
# Domain-specific control mapping (based on Annex III signals)
# ---------------------------------------------------------------------------

_DOMAIN_CONTROLS: dict[str, list[tuple[str, str]]] = {
    # Biometric systems need data acquisition and quality controls
    "remote_biometric_identification": [
        ("A.7.2", "Biometric data development controls required"),
        ("A.7.5", "Biometric data acquisition must meet ethical/legal standards"),
    ],
    "biometric_categorization": [
        ("A.7.2", "Biometric categorization data governance required"),
        ("A.7.5", "Biometric data acquisition ethical controls required"),
    ],
    "emotion_recognition": [
        ("A.7.2", "Emotion recognition data development controls required"),
        ("A.7.5", "Emotion data acquisition must meet consent requirements"),
    ],
    # Employment systems need reporting and responsible use
    "employment_recruitment": [
        ("A.9.2", "Responsible use objectives for employment AI required"),
        ("A.3.3", "Reporting mechanisms for recruitment AI concerns required"),
        ("A.3.2", "Clear roles for HR AI system oversight required"),
    ],
    "employment_decisions": [
        ("A.9.2", "Responsible use objectives for employment decision AI required"),
        ("A.3.3", "Reporting mechanisms for employment decision concerns required"),
        ("A.3.2", "Clear roles for employment AI oversight required"),
    ],
    # Credit/insurance need data quality
    "creditworthiness_assessment": [
        ("A.7.3", "Credit scoring data quality requirements"),
        ("A.7.2", "Credit model training data governance required"),
    ],
    "insurance_risk_assessment": [
        ("A.7.3", "Insurance risk model data quality requirements"),
        ("A.7.2", "Insurance model training data governance required"),
    ],
    # Third-party components
    "critical_infrastructure_safety": [
        ("A.6.2.11", "Critical infrastructure AI third-party component management"),
        ("A.10.3", "Shared ML model provenance verification for critical systems"),
        ("A.4.5", "Personnel competence requirements for critical infrastructure AI"),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_controls(
    profile: FeatureProfile,
    classification: Classification | None = None,
    tier: RiskTier | None = None,
) -> list[ISOControl]:
    """Map ISO 42001 controls to a feature based on its risk tier and signals.

    Args:
        profile: The extracted feature profile.
        classification: If available, the classification result (used for tier).
        tier: Override tier directly. If neither classification nor tier is
            given, defaults to MINIMAL.

    Returns:
        Deduplicated list of applicable ISOControl objects.
    """
    effective_tier = tier or (classification.final_tier if classification else RiskTier.MINIMAL)

    # Track (control_id -> ISOControl) for deduplication.
    controls: dict[str, ISOControl] = {}

    def _add(control_id: str, applicability: str) -> None:
        # First mapping wins — keeps the most specific applicability reason.
        if control_id not in controls:
            controls[control_id] = _make_control(control_id, applicability)

    # Universal controls
    for cid, reason in _UNIVERSAL_CONTROLS:
        _add(cid, reason)

    # Tier-specific controls
    if effective_tier == RiskTier.PROHIBITED:
        for cid, reason in _PROHIBITED_CONTROLS:
            _add(cid, reason)
        # Prohibited systems also get HIGH controls for documentation purposes
        for cid, reason in _HIGH_RISK_CONTROLS:
            _add(cid, reason)
    elif effective_tier == RiskTier.HIGH:
        for cid, reason in _HIGH_RISK_CONTROLS:
            _add(cid, reason)
    elif effective_tier == RiskTier.LIMITED:
        for cid, reason in _LIMITED_CONTROLS:
            _add(cid, reason)
    # MINIMAL gets only universal controls.

    # Domain-specific controls based on Annex III signals
    for signal in profile.high_risk_signals.value:
        if signal in _DOMAIN_CONTROLS:
            for cid, reason in _DOMAIN_CONTROLS[signal]:
                _add(cid, reason)

    # Sort by control_id for deterministic ordering
    return sorted(controls.values(), key=lambda c: c.control_id)
