"""CLI entry point for the AI Act Risk Assessor.

Commands:
    assessor assess  --input <file> [--form <file>]   Run full pipeline.
    assessor replay  <record.json>                     Replay deterministic checks.
    assessor list                                      List all audit records.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from assessor.ai_act import classify
from assessor.audit import assemble_record, list_records, write_record
from assessor.drafter import DEFAULT_DRAFTER_MODEL, draft
from assessor.extractor import DEFAULT_EXTRACTOR_MODEL, extract
from assessor.iso_42001 import map_controls
from assessor.normalizer import hash_text, normalize_text, read_form, read_input
from assessor.replay import replay as run_replay
from assessor.verifier import verify_extraction, verify_memo

console = Console()


def _check(label: str, ok: bool, detail: str = "") -> None:
    """Print a status line."""
    mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
    msg = f"{mark} {label}"
    if detail:
        msg += f": {detail}"
    console.print(msg)


# ---------------------------------------------------------------------------
# assess command
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """AI Act Risk Assessor — neuro-symbolic classification with audit trail."""


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input text file.")
@click.option("--form", "form_path", type=click.Path(exists=True), default=None, help="Optional structured form JSON.")
@click.option("--extractor-model", default=DEFAULT_EXTRACTOR_MODEL, help="Ollama model for extraction.")
@click.option("--drafter-model", default=DEFAULT_DRAFTER_MODEL, help="Ollama model for drafting.")
@click.option("--audit-dir", default="audit", type=click.Path(), help="Audit record directory.")
@click.option("--skip-llm", is_flag=True, default=False, help="Skip LLM calls (for testing deterministic pipeline).")
def assess(
    input_path: str,
    form_path: str | None,
    extractor_model: str,
    drafter_model: str,
    audit_dir: str,
    skip_llm: bool,
) -> None:
    """Run the full assessment pipeline on an input file."""
    audit_path = Path(audit_dir)

    # --- Read and normalize input ---
    input_text = read_input(Path(input_path))
    input_sha = hash_text(input_text)
    form_data: dict[str, object] | None = None
    form_summary: str | None = None
    if form_path:
        form_data = read_form(Path(form_path))
        form_summary = str(form_data)

    # --- Extraction ---
    if skip_llm:
        console.print("[yellow]⚠ --skip-llm: using mock extraction[/yellow]")
        from assessor.schema import (
            AutonomyLevel,
            ConfidenceLevel,
            DataSensitivity,
            DeploymentScope,
            ExtractedField,
            FeatureProfile,
        )
        profile = FeatureProfile(
            feature_name=ExtractedField[str](value="Unknown (skip-llm)", source_span="", confidence=ConfidenceLevel.UNCLEAR),
            description=ExtractedField[str](value=input_text[:200], source_span=input_text[:100] if len(input_text) >= 100 else input_text, confidence=ConfidenceLevel.INFERRED),
            domain=ExtractedField[str](value="unknown", source_span="", confidence=ConfidenceLevel.UNCLEAR),
            affected_subjects=ExtractedField[list[str]](value=[], source_span="", confidence=ConfidenceLevel.UNCLEAR),
            operators=ExtractedField[list[str]](value=[], source_span="", confidence=ConfidenceLevel.UNCLEAR),
            autonomy_level=ExtractedField[AutonomyLevel](value=AutonomyLevel.ADVISORY_ONLY, source_span="", confidence=ConfidenceLevel.UNCLEAR),
            decision_impact=ExtractedField[str](value="unknown", source_span="", confidence=ConfidenceLevel.UNCLEAR),
            data_types=ExtractedField[list[DataSensitivity]](value=[], source_span="", confidence=ConfidenceLevel.UNCLEAR),
            uses_biometric_data=ExtractedField[bool](value=False, source_span="", confidence=ConfidenceLevel.UNCLEAR),
            uses_personal_data=ExtractedField[bool](value=False, source_span="", confidence=ConfidenceLevel.UNCLEAR),
            deployment_scope=ExtractedField[DeploymentScope](value=DeploymentScope.PRIVATE_SPACE, source_span="", confidence=ConfidenceLevel.UNCLEAR),
            sector=ExtractedField[str](value="unknown", source_span="", confidence=ConfidenceLevel.UNCLEAR),
            prohibited_signals=ExtractedField[list[str]](value=[], source_span="", confidence=ConfidenceLevel.UNCLEAR),
            high_risk_signals=ExtractedField[list[str]](value=[], source_span="", confidence=ConfidenceLevel.UNCLEAR),
            generates_content=ExtractedField[bool](value=False, source_span="", confidence=ConfidenceLevel.UNCLEAR),
            interacts_with_humans=ExtractedField[bool](value=False, source_span="", confidence=ConfidenceLevel.UNCLEAR),
            generates_deepfakes=ExtractedField[bool](value=False, source_span="", confidence=ConfidenceLevel.UNCLEAR),
        )
        _check("Extraction", True, "mock profile (--skip-llm)")
    else:
        t0 = time.time()
        try:
            profile = extract(
                input_text,
                form_summary=form_summary,
                model=extractor_model,
            )
            elapsed = time.time() - t0
            _check("Extraction", True, f"{elapsed:.1f}s")
        except Exception as e:
            _check("Extraction", False, str(e))
            sys.exit(1)

    # --- Verification ---
    verification = verify_extraction(profile, input_text, form_data)
    span_checks = [c for c in verification.checks if c.check_id.startswith("SPAN_")]
    span_ok = sum(1 for c in span_checks if c.passed)
    _check(
        "Verification",
        verification.passed,
        f"{span_ok}/{len(span_checks)} spans found, coverage ok"
        if verification.passed
        else f"{verification.error_count} errors, {verification.warning_count} warnings",
    )

    # --- Classification ---
    iso_ctrls = map_controls(profile, tier=None)  # Preliminary pass without tier.
    classification = classify(profile, iso_controls=iso_ctrls)
    # Re-map ISO controls with the actual tier now known.
    iso_ctrls = map_controls(profile, tier=classification.final_tier)
    classification = classify(profile, iso_controls=iso_ctrls)

    _check(
        "Classification",
        True,
        f"{classification.final_tier.value.upper()} "
        f"({', '.join(r.article_ref for r in classification.triggered_rules[:3])})"
        if classification.triggered_rules
        else f"{classification.final_tier.value.upper()}",
    )

    # --- Drafting ---
    if skip_llm:
        memo = f"# Assessment Memo (mock)\n\nClassification: {classification.final_tier.value.upper()}\n"
        _check("Drafting", True, "mock memo (--skip-llm)")
    else:
        t0 = time.time()
        try:
            memo = draft(classification, model=drafter_model)
            elapsed = time.time() - t0
            _check("Drafting", True, f"{elapsed:.1f}s")
        except Exception as e:
            _check("Drafting", False, str(e))
            memo = f"# Drafting Failed\n\nError: {e}\n"

    # --- Citation grounding ---
    memo_check = verify_memo(memo, classification)
    cite_checks = [c for c in memo_check.checks if c.check_id.startswith("CITATION_")]
    cite_ok = sum(1 for c in cite_checks if c.passed)
    _check(
        "Citation grounding",
        memo_check.passed,
        f"{cite_ok}/{len(cite_checks)} resolved"
        if cite_checks
        else "no citations to verify",
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
        audit_dir=audit_path,
    )

    try:
        record_file = write_record(
            record, input_text, audit_dir=audit_path, skip_schema_validation=skip_llm,
        )
        _check("Audit record written", True, str(record_file))
    except Exception as e:
        _check("Audit record written", False, str(e))
        # Still try writing without schema validation.
        try:
            record_file = write_record(
                record, input_text, audit_dir=audit_path, skip_schema_validation=True,
            )
            _check("Audit record written (no schema)", True, str(record_file))
        except Exception as e2:
            _check("Audit record written (fallback)", False, str(e2))
            sys.exit(1)

    _check("Schema valid", True)

    console.print()
    memo_path = record_file.parent / f"{record.assessment_id}.memo.md"
    console.print(f"Memo: {memo_path}")


# ---------------------------------------------------------------------------
# replay command
# ---------------------------------------------------------------------------


@main.command()
@click.argument("record_path", type=click.Path(exists=True))
@click.option("--audit-dir", default="audit", type=click.Path(), help="Audit record directory.")
def replay(record_path: str, audit_dir: str) -> None:
    """Replay deterministic checks against a stored audit record."""
    result = run_replay(Path(record_path), audit_dir=Path(audit_dir))

    for check in result.checks:
        _check(check.step.replace("_", " ").title(), check.passed, check.detail)

    console.print()
    if result.overall_passed:
        console.print("[bold green]PASS[/bold green]")
    else:
        console.print("[bold red]FAIL[/bold red]")
        sys.exit(1)


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


@main.command(name="list")
@click.option("--audit-dir", default="audit", type=click.Path(), help="Audit record directory.")
def list_cmd(audit_dir: str) -> None:
    """List all audit records with timestamps, classifications, review status."""
    records = list_records(Path(audit_dir))

    if not records:
        console.print("No audit records found.")
        return

    table = Table(title="Audit Records")
    table.add_column("ID", style="cyan", max_width=12)
    table.add_column("Timestamp", style="green")
    table.add_column("Tier", style="bold")
    table.add_column("Review", style="yellow")
    table.add_column("Path", style="dim")

    for rec in records:
        tier = rec["tier"].upper()
        tier_style = {
            "PROHIBITED": "[bold red]PROHIBITED[/bold red]",
            "HIGH": "[bold yellow]HIGH[/bold yellow]",
            "LIMITED": "[blue]LIMITED[/blue]",
            "MINIMAL": "[green]MINIMAL[/green]",
        }.get(tier, tier)

        table.add_row(
            rec["assessment_id"][:12],
            rec["timestamp"][:19],
            tier_style,
            rec["requires_review"],
            rec["path"],
        )

    console.print(table)


if __name__ == "__main__":
    main()
