"""Streamlit UI for the AI Act Risk Assessor.

A thin presentation layer over the exact same pipeline the CLI uses. It runs
no logic of its own — every classification still flows through the
deterministic rule engine and verifiers. The UI's job is to make the
neuro-symbolic decision path *visible*: the full rule cascade (including
non-matches), extracted spans with confidence, ISO 42001 controls, and the
hash-chained audit record.

Run with:

    uv pip install -e ".[ui]"
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

from assessor.ai_act import classify
from assessor.audit import assemble_record, list_records, write_record
from assessor.drafter import DEFAULT_DRAFTER_MODEL, draft
from assessor.extractor import DEFAULT_EXTRACTOR_MODEL, extract
from assessor.iso_42001 import map_controls
from assessor.normalizer import hash_text, read_form
from assessor.replay import replay as run_replay
from assessor.schema import (
    AutonomyLevel,
    ConfidenceLevel,
    DataSensitivity,
    DeploymentScope,
    ExtractedField,
    FeatureProfile,
    RiskTier,
)
from assessor.verifier import verify_extraction, verify_memo

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

TIER_COLOR = {
    RiskTier.PROHIBITED: "#dc2626",
    RiskTier.HIGH: "#d97706",
    RiskTier.LIMITED: "#2563eb",
    RiskTier.MINIMAL: "#16a34a",
}


def tier_badge(tier: RiskTier) -> None:
    """Render a large colored tier badge."""
    color = TIER_COLOR.get(tier, "#6b7280")
    st.markdown(
        f"""<div style="background:{color};color:white;padding:0.75rem 1.25rem;
        border-radius:0.5rem;font-size:1.5rem;font-weight:700;text-align:center;">
        {tier.value.upper()}</div>""",
        unsafe_allow_html=True,
    )


def _mock_profile(input_text: str) -> FeatureProfile:
    """Build the same unclassified mock profile the CLI uses for --skip-llm."""
    unclear = ConfidenceLevel.UNCLEAR
    return FeatureProfile(
        feature_name=ExtractedField[str](value="Unknown (skip-llm)", source_span="", confidence=unclear),
        description=ExtractedField[str](
            value=input_text[:200],
            source_span=input_text[:100] if len(input_text) >= 100 else input_text,
            confidence=ConfidenceLevel.INFERRED,
        ),
        domain=ExtractedField[str](value="unknown", source_span="", confidence=unclear),
        affected_subjects=ExtractedField[list[str]](value=[], source_span="", confidence=unclear),
        operators=ExtractedField[list[str]](value=[], source_span="", confidence=unclear),
        autonomy_level=ExtractedField[AutonomyLevel](value=AutonomyLevel.ADVISORY_ONLY, source_span="", confidence=unclear),
        decision_impact=ExtractedField[str](value="unknown", source_span="", confidence=unclear),
        data_types=ExtractedField[list[DataSensitivity]](value=[], source_span="", confidence=unclear),
        uses_biometric_data=ExtractedField[bool](value=False, source_span="", confidence=unclear),
        uses_personal_data=ExtractedField[bool](value=False, source_span="", confidence=unclear),
        deployment_scope=ExtractedField[DeploymentScope](value=DeploymentScope.PRIVATE_SPACE, source_span="", confidence=unclear),
        sector=ExtractedField[str](value="unknown", source_span="", confidence=unclear),
        prohibited_signals=ExtractedField[list[str]](value=[], source_span="", confidence=unclear),
        high_risk_signals=ExtractedField[list[str]](value=[], source_span="", confidence=unclear),
        generates_content=ExtractedField[bool](value=False, source_span="", confidence=unclear),
        interacts_with_humans=ExtractedField[bool](value=False, source_span="", confidence=unclear),
        generates_deepfakes=ExtractedField[bool](value=False, source_span="", confidence=unclear),
    )


def render_profile(profile: FeatureProfile) -> None:
    """Render extracted fields as a table of value / span / confidence."""
    rows = []
    for name, field in profile:  # FeatureProfile iterates (name, ExtractedField)
        if not isinstance(field, ExtractedField):
            continue
        value = field.value
        value_str = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        conf = field.confidence.value if hasattr(field.confidence, "value") else str(field.confidence)
        rows.append({
            "Field": name,
            "Value": value_str or "—",
            "Source span": field.source_span or "—",
            "Confidence": conf,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_cascade(classification) -> None:
    """Render the full rule cascade — matched and non-matched alike."""
    rows = []
    for rule in classification.all_rules_evaluated:
        rows.append({
            "": "✅" if rule.matched else "▫️",
            "Rule": rule.rule_id,
            "Article": rule.article_ref,
            "Asserts": rule.tier.value.upper(),
            "Signals": ", ".join(rule.triggered_signals) or "—",
            "Rationale": rule.rationale or "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Assess tab
# ---------------------------------------------------------------------------


def run_assessment(
    input_text: str,
    form_data: dict | None,
    extractor_model: str,
    drafter_model: str,
    audit_dir: Path,
    skip_llm: bool,
) -> None:
    """Run the full pipeline and render every stage. Mirrors cli.assess."""
    input_sha = hash_text(input_text)
    form_summary = str(form_data) if form_data else None

    status = st.status("Running assessment pipeline…", expanded=True)

    # --- Extraction ---
    if skip_llm:
        status.write("⚠️ --skip-llm: using mock extraction (deterministic path only)")
        profile = _mock_profile(input_text)
    else:
        t0 = time.time()
        try:
            profile = extract(input_text, form_summary=form_summary, model=extractor_model)
        except Exception as e:  # noqa: BLE001 — surface any model/connection error to the user
            status.update(label="Extraction failed", state="error")
            st.error(f"Extraction failed: {e}")
            return
        status.write(f"✅ Extraction — {time.time() - t0:.1f}s ({extractor_model})")

    # --- Verification ---
    verification = verify_extraction(profile, input_text, form_data)
    span_checks = [c for c in verification.checks if c.check_id.startswith("SPAN_")]
    span_ok = sum(1 for c in span_checks if c.passed)
    status.write(
        f"{'✅' if verification.passed else '❌'} Verification — "
        f"{span_ok}/{len(span_checks)} spans found"
        if verification.passed
        else f"❌ Verification — {verification.error_count} errors, {verification.warning_count} warnings"
    )

    # --- Classification (two-pass: preliminary, then with known tier) ---
    iso_ctrls = map_controls(profile, tier=None)
    classification = classify(profile, iso_controls=iso_ctrls)
    iso_ctrls = map_controls(profile, tier=classification.final_tier)
    classification = classify(profile, iso_controls=iso_ctrls)
    refs = ", ".join(r.article_ref for r in classification.triggered_rules[:3])
    status.write(f"✅ Classification — {classification.final_tier.value.upper()} ({refs or 'default'})")

    # --- Drafting ---
    if skip_llm:
        memo = f"# Assessment Memo (mock)\n\nClassification: {classification.final_tier.value.upper()}\n"
        status.write("⚠️ Drafting — mock memo (--skip-llm)")
    else:
        t0 = time.time()
        try:
            memo = draft(classification, model=drafter_model)
            status.write(f"✅ Drafting — {time.time() - t0:.1f}s ({drafter_model})")
        except Exception as e:  # noqa: BLE001
            memo = f"# Drafting Failed\n\nError: {e}\n"
            status.write(f"❌ Drafting failed: {e}")

    # --- Citation grounding ---
    memo_check = verify_memo(memo, classification)
    cite_checks = [c for c in memo_check.checks if c.check_id.startswith("CITATION_")]
    cite_ok = sum(1 for c in cite_checks if c.passed)
    status.write(
        f"{'✅' if memo_check.passed else '❌'} Citation grounding — "
        f"{cite_ok}/{len(cite_checks)} resolved" if cite_checks else "✅ Citation grounding — none to verify"
    )

    # --- Audit record ---
    record = assemble_record(
        input_text=input_text,
        input_text_sha256=input_sha,
        feature_profile=profile,
        extraction_verification=verification,
        classification=classification,
        memo=memo,
        memo_verification=memo_check,
        extractor_model_id=extractor_model,
        drafter_model_id=drafter_model,
        audit_dir=audit_dir,
    )
    try:
        record_file = write_record(record, input_text, audit_dir=audit_dir, skip_schema_validation=skip_llm)
        status.write(f"✅ Audit record written — {record_file}")
    except Exception as e:  # noqa: BLE001
        record_file = write_record(record, input_text, audit_dir=audit_dir, skip_schema_validation=True)
        status.write(f"⚠️ Audit record written without schema validation — {record_file}")

    status.update(label="Assessment complete", state="complete")

    # ----- Results -----
    col1, col2 = st.columns([1, 2])
    with col1:
        tier_badge(classification.final_tier)
        if classification.requires_human_review:
            st.warning("⚖️ Requires human review")
    with col2:
        if classification.obligations:
            st.markdown("**Obligations**")
            for ob in classification.obligations:
                st.markdown(f"- {ob}")
        if classification.downgrade:
            st.info(f"Art. 6(3) downgrade applied: {classification.downgrade.rationale}")

    if classification.human_review_reasons:
        with st.expander("Human review reasons"):
            for r in classification.human_review_reasons:
                st.markdown(f"- {r}")

    st.subheader("Rule cascade")
    st.caption("Every rule is evaluated and recorded — matched (✅) and non-matched (▫️) alike.")
    render_cascade(classification)

    st.subheader("Extracted feature profile")
    st.caption("Populated by the LLM, audited by the deterministic verifier.")
    render_profile(profile)

    with st.expander("Verification checks (extraction)"):
        st.dataframe(
            [{"Check": c.check_id, "Passed": c.passed, "Severity": c.severity, "Detail": c.detail} for c in verification.checks],
            use_container_width=True, hide_index=True,
        )

    if classification.iso_controls:
        st.subheader("ISO 42001 Annex A controls")
        st.dataframe(
            [{"Control": c.control_id, "Title": c.title, "Applicability": c.applicability} for c in classification.iso_controls],
            use_container_width=True, hide_index=True,
        )

    st.subheader("Assessment memo")
    st.markdown(memo)

    with st.expander("Audit record (provenance + hash chain)"):
        st.json({
            "assessment_id": record.assessment_id,
            "timestamp_utc": record.timestamp_utc.isoformat(),
            "final_tier": classification.final_tier.value,
            "provenance": record.provenance.model_dump(mode="json"),
            "previous_record_sha256": record.previous_record_sha256,
            "record_file": str(record_file),
        })


def assess_tab() -> None:
    st.header("Assess a feature")
    st.caption("Paste a feature description, or load one of the bundled fixtures.")

    fixtures_dir = Path("tests/fixtures")
    fixtures = sorted(fixtures_dir.glob("feature_*.txt")) if fixtures_dir.exists() else []
    fixture_names = ["(none)"] + [f.name for f in fixtures]
    chosen = st.selectbox("Load fixture", fixture_names)
    default_text = ""
    if chosen != "(none)":
        default_text = (fixtures_dir / chosen).read_text(encoding="utf-8")

    input_text = st.text_area("Feature description", value=default_text, height=260)
    form_file = st.file_uploader("Optional structured form (JSON)", type=["json"])
    form_data = json.load(form_file) if form_file else None

    if st.button("Run assessment", type="primary", disabled=not input_text.strip()):
        run_assessment(
            input_text=input_text,
            form_data=form_data,
            extractor_model=st.session_state["extractor_model"],
            drafter_model=st.session_state["drafter_model"],
            audit_dir=Path(st.session_state["audit_dir"]),
            skip_llm=st.session_state["skip_llm"],
        )


# ---------------------------------------------------------------------------
# Audit Trail tab
# ---------------------------------------------------------------------------


def audit_tab() -> None:
    st.header("Audit trail")
    audit_dir = Path(st.session_state["audit_dir"])
    records = list_records(audit_dir)

    if not records:
        st.info(f"No audit records found in `{audit_dir}/`. Run an assessment first.")
        return

    st.dataframe(
        [{
            "ID": r["assessment_id"][:12],
            "Timestamp": r["timestamp"][:19],
            "Tier": r["tier"].upper(),
            "Requires review": r["requires_review"],
            "Path": r["path"],
        } for r in records],
        use_container_width=True, hide_index=True,
    )

    st.subheader("Replay verification")
    st.caption("Re-runs all deterministic checks against a stored record — without re-running any LLM — and detects tampering.")
    paths = [r["path"] for r in records]
    chosen = st.selectbox("Record to replay", paths)
    if st.button("Replay", type="primary"):
        result = run_replay(Path(chosen), audit_dir=audit_dir)
        for check in result.checks:
            label = check.step.replace("_", " ").title()
            (st.success if check.passed else st.error)(f"{'✅' if check.passed else '❌'} {label} — {check.detail}")
        if result.overall_passed:
            st.success("**PASS** — record is intact and the classification reproduces exactly.")
        else:
            st.error("**FAIL** — record failed deterministic replay.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="AI Act Risk Assessor", page_icon="⚖️", layout="wide")
    st.title("⚖️ AI Act Risk Assessor")
    st.caption("Neuro-symbolic classification — LLMs extract and draft, but the risk tier is 100% deterministic.")

    with st.sidebar:
        st.header("Settings")
        st.session_state["skip_llm"] = st.toggle(
            "Skip LLM (deterministic path only)", value=False,
            help="Runs the rule engine, verifiers, and audit trail without calling Ollama. Useful when no model is available.",
        )
        st.session_state["extractor_model"] = st.text_input("Extractor model", value=DEFAULT_EXTRACTOR_MODEL)
        st.session_state["drafter_model"] = st.text_input("Drafter model", value=DEFAULT_DRAFTER_MODEL)
        st.session_state["audit_dir"] = st.text_input("Audit directory", value="audit")
        st.divider()
        st.caption("Requires a running Ollama instance unless 'Skip LLM' is on.")

    assess, audit = st.tabs(["Assess", "Audit trail"])
    with assess:
        assess_tab()
    with audit:
        audit_tab()


if __name__ == "__main__":
    main()
