"""Deterministic validation checks — SPEC §6. This is the confidence gate, and it
is deliberately NOT the model grading its own certainty: every failure is a
specific, auditable, human-readable string a buyer can point to ("the move-in
date didn't parse" beats "the model felt unsure").
"""

from __future__ import annotations

from datetime import date

import phonenumbers
from email_validator import EmailNotValidError, validate_email
from rapidfuzz import fuzz

import config
from schema import ApplicationFields, Extraction, Intent, LeadFields, ValidationResult


def _check_email(email: str | None) -> str | None:
    if email is None:
        return None
    try:
        validate_email(email, check_deliverability=False)
    except EmailNotValidError:
        return "invalid email format"
    return None


def _check_phone(phone: str | None) -> str | None:
    if phone is None:
        return None
    try:
        parsed = phonenumbers.parse(phone, "US")
    except phonenumbers.NumberParseException:
        return "invalid phone number"
    if not phonenumbers.is_valid_number(parsed):
        return "invalid phone number"
    return None


def _check_move_in_date(move_in: date | None) -> str | None:
    if move_in is None:
        return None
    if move_in < date.today():
        return "move-in date invalid or in the past"
    return None


def _check_property_ref(property_ref: str | None, raw_text: str) -> str | None:
    """The hallucination guard — the most important check in the system."""
    if property_ref is None:
        return None
    score = fuzz.partial_ratio(property_ref.lower(), raw_text.lower())
    if score < config.PROPERTY_MATCH_THRESHOLD:
        return "property reference not found in source — possible hallucination"
    return None


def missing_required_fields(extraction: Extraction) -> list[str]:
    """SPEC §5. Computed in Python, never by the LLM."""
    required = config.REQUIRED_FIELDS.get(extraction.intent)
    if not required:
        return []

    fields: LeadFields | ApplicationFields | None
    if extraction.intent == Intent.new_lead:
        fields = extraction.lead
    elif extraction.intent == Intent.application:
        fields = extraction.application
    else:
        fields = None

    if fields is None:
        return [entry if isinstance(entry, str) else " or ".join(entry) for entry in required]

    missing = []
    for entry in required:
        if isinstance(entry, tuple):
            if not any(getattr(fields, alt, None) for alt in entry):
                missing.append(" or ".join(entry))
        elif not getattr(fields, entry, None):
            missing.append(entry)
    return missing


def validate(extraction: Extraction, raw_text: str) -> ValidationResult:
    failed: list[str] = []

    if extraction.intent == Intent.new_lead and extraction.lead is not None:
        lead = extraction.lead
        checks = (
            _check_email(lead.contact_email),
            _check_phone(lead.contact_phone),
            _check_move_in_date(lead.desired_move_in),
            _check_property_ref(lead.property_ref, raw_text),
        )
        failed.extend(c for c in checks if c)

    elif extraction.intent == Intent.application and extraction.application is not None:
        app = extraction.application
        checks = (
            _check_move_in_date(app.desired_move_in),
            _check_property_ref(app.desired_unit, raw_text),
        )
        failed.extend(c for c in checks if c)

    missing = missing_required_fields(extraction)
    if extraction.intent == Intent.application and extraction.application is not None:
        # The only schema field that persists this (SPEC §4) — leads have no
        # equivalent slot, so their missing-field list is routing-only (below).
        extraction.application.missing_required_fields = missing
    if missing:
        failed.append(f"missing required fields: {', '.join(missing)}")

    return ValidationResult(passed=len(failed) == 0, failed_checks=failed)
