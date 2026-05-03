"""Shared test fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from assessor.schema import (
    AutonomyLevel,
    ConfidenceLevel,
    DataSensitivity,
    DeploymentScope,
    ExtractedField,
    FeatureProfile,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def expected_classifications() -> dict[str, dict[str, object]]:
    yaml_path = FIXTURES_DIR / "expected_classifications.yaml"
    return dict(yaml.safe_load(yaml_path.read_text(encoding="utf-8")))


def make_profile(
    *,
    feature_name: str = "Test Feature",
    description: str = "A test AI feature.",
    domain: str = "testing",
    prohibited_signals: list[str] | None = None,
    high_risk_signals: list[str] | None = None,
    generates_content: bool = False,
    interacts_with_humans: bool = False,
    generates_deepfakes: bool = False,
    autonomy_level: AutonomyLevel = AutonomyLevel.ADVISORY_ONLY,
    decision_impact: str = "informational recommendation only",
    input_text: str = "Test Feature: A test AI feature for testing purposes.",
) -> FeatureProfile:
    """Create a FeatureProfile with sensible defaults for testing.

    All source_spans reference the input_text so span verification passes.
    """
    return FeatureProfile(
        feature_name=ExtractedField[str](
            value=feature_name,
            source_span=feature_name if feature_name in input_text else "",
            confidence=ConfidenceLevel.EXPLICIT if feature_name in input_text else ConfidenceLevel.INFERRED,
        ),
        description=ExtractedField[str](
            value=description,
            source_span=description if description in input_text else "",
            confidence=ConfidenceLevel.EXPLICIT if description in input_text else ConfidenceLevel.INFERRED,
        ),
        domain=ExtractedField[str](
            value=domain,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        affected_subjects=ExtractedField[list[str]](
            value=["test_subjects"],
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        operators=ExtractedField[list[str]](
            value=["test_operator"],
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        autonomy_level=ExtractedField[AutonomyLevel](
            value=autonomy_level,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        decision_impact=ExtractedField[str](
            value=decision_impact,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        data_types=ExtractedField[list[DataSensitivity]](
            value=[DataSensitivity.PERSONAL],
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        uses_biometric_data=ExtractedField[bool](
            value=False,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        uses_personal_data=ExtractedField[bool](
            value=True,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        deployment_scope=ExtractedField[DeploymentScope](
            value=DeploymentScope.PRIVATE_SPACE,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        sector=ExtractedField[str](
            value="private_sector",
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        prohibited_signals=ExtractedField[list[str]](
            value=prohibited_signals or [],
            source_span="",
            confidence=ConfidenceLevel.EXPLICIT if prohibited_signals else ConfidenceLevel.INFERRED,
        ),
        high_risk_signals=ExtractedField[list[str]](
            value=high_risk_signals or [],
            source_span="",
            confidence=ConfidenceLevel.EXPLICIT if high_risk_signals else ConfidenceLevel.INFERRED,
        ),
        generates_content=ExtractedField[bool](
            value=generates_content,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        interacts_with_humans=ExtractedField[bool](
            value=interacts_with_humans,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
        generates_deepfakes=ExtractedField[bool](
            value=generates_deepfakes,
            source_span="",
            confidence=ConfidenceLevel.INFERRED,
        ),
    )
