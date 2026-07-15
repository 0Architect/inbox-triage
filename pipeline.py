"""process(item) -> TriageResult — the whole pipeline, wired end to end (SPEC §2,
§12). Deterministic orchestration, NOT an agent: the only decisions an LLM makes
are classify/extract/draft; validate and route are plain Python.
"""

from __future__ import annotations

import time

import anthropic

import storage
from classify import classify
from draft import draft
from extract import extract
from ingest import IngestedItem
from route import route
from schema import Extraction, Intent, Route, TriageResult, ValidationResult
from validate import validate


def process(
    item: IngestedItem,
    *,
    client: anthropic.Anthropic | None = None,
    conn=None,
) -> TriageResult:
    start = time.monotonic()
    client = client or anthropic.Anthropic()

    intent = classify(item.raw_text, client=client)

    if intent == Intent.spam:
        result = TriageResult(
            source_id=item.source_id,
            raw_text=item.raw_text,
            extraction=Extraction(intent=Intent.spam),
            validation=ValidationResult(passed=True, failed_checks=[]),
            route=Route.discarded,
            draft_reply=None,
            reasons=["intent classified as spam"],
        )
    else:
        extraction = extract(item.raw_text, intent, client=client)
        validation = validate(extraction, item.raw_text)
        chosen_route, reasons = route(extraction, validation)

        draft_reply = draft(extraction, client=client) if chosen_route == Route.auto else None

        result = TriageResult(
            source_id=item.source_id,
            raw_text=item.raw_text,
            extraction=extraction,
            validation=validation,
            route=chosen_route,
            draft_reply=draft_reply,
            reasons=reasons,
        )

    processing_ms = int((time.monotonic() - start) * 1000)
    if conn is not None:
        storage.save_triage_result(conn, result, item.source_meta, processing_ms)

    return result
