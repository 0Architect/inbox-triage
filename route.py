"""Routing gate — SPEC §6. Plain Python `if` rules, evaluated in order. No model
self-grading; the gate is exactly as auditable as the rules below.
"""

from __future__ import annotations

from schema import Extraction, Intent, Route, TriageResult, ValidationResult


def route(extraction: Extraction, validation: ValidationResult) -> tuple[Route, list[str]]:
    if extraction.intent == Intent.spam:
        return Route.discarded, ["intent classified as spam"]

    reasons: list[str] = list(validation.failed_checks)
    if extraction.urgent_flag:
        reasons.append("flagged urgent")

    if reasons:
        return Route.human_review, reasons

    return Route.auto, []
