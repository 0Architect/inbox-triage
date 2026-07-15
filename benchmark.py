"""Benchmark harness — SPEC §9. Runs process() over the gold set, reports
field-level accuracy (overall + per-field), routing accuracy, average
processing time, and the derived time-saved / $-saved headline number.

Usage:
    python benchmark.py [--data-dir data] [--db data/inbox_triage.db]

Requires ANTHROPIC_API_KEY — this exercises the live classify/extract calls.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import anthropic
import phonenumbers
from rapidfuzz import fuzz

import config
import pipeline
import storage
from ingest import IngestedItem
from schema import Intent, Route

FUZZY_FIELDS = {"property_ref", "desired_unit"}
PHONE_FIELDS = {"contact_phone"}
LIST_FIELDS = {"co_applicants"}
DATE_FIELDS = {"desired_move_in"}


def _norm_str(v: str) -> str:
    return v.strip().lower()


def _norm_phone(v: str) -> str:
    try:
        parsed = phonenumbers.parse(v, "US")
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    except phonenumbers.NumberParseException:
        return _norm_str(v)


def _coerce_gold_field(field_name: str, value):
    if value is not None and field_name in DATE_FIELDS and isinstance(value, str):
        return date.fromisoformat(value)
    return value


def fields_match(field_name: str, predicted, expected) -> bool:
    if expected is None and predicted is None:
        return True
    if expected is None or predicted is None:
        return False
    if field_name in FUZZY_FIELDS:
        return fuzz.ratio(str(predicted).lower(), str(expected).lower()) >= config.PROPERTY_MATCH_THRESHOLD
    if field_name in PHONE_FIELDS:
        return _norm_phone(str(predicted)) == _norm_phone(str(expected))
    if field_name in LIST_FIELDS:
        return {_norm_str(x) for x in predicted} == {_norm_str(x) for x in expected}
    if isinstance(expected, str):
        return _norm_str(str(predicted)) == _norm_str(expected)
    if isinstance(expected, float):
        try:
            return abs(float(predicted) - expected) < 0.01
        except (TypeError, ValueError):
            return False
    if isinstance(expected, date):
        return predicted == expected
    return predicted == expected


def _gold_section(ground_truth: dict) -> dict | None:
    intent = ground_truth["intent"]
    if intent == Intent.new_lead.value:
        return ground_truth.get("lead")
    if intent == Intent.application.value:
        return ground_truth.get("application")
    return None


def _predicted_section(extraction, intent: str) -> dict:
    if intent == Intent.new_lead.value and extraction.lead is not None:
        return extraction.lead.model_dump()
    if intent == Intent.application.value and extraction.application is not None:
        return extraction.application.model_dump()
    return {}


def load_dataset(data_dir: Path) -> tuple[dict[str, dict], list[dict]]:
    items = {json.loads(l)["source_id"]: json.loads(l) for l in (data_dir / "items.jsonl").open()}
    gold = [json.loads(l) for l in (data_dir / "gold.jsonl").open()]
    return items, gold


def run_benchmark(data_dir: Path, db_path: Path) -> dict:
    items, gold = load_dataset(data_dir)
    conn = storage.get_connection(db_path)
    client = anthropic.Anthropic()

    field_correct: dict[str, int] = defaultdict(int)
    field_total: dict[str, int] = defaultdict(int)
    intent_correct = 0
    adversarial_total = 0
    adversarial_correct = 0
    route_counts: dict[str, int] = defaultdict(int)
    processing_times: list[int] = []

    for i, record in enumerate(gold, 1):
        source_id = record["source_id"]
        item_row = items[source_id]
        ground_truth = record["ground_truth"]
        is_adversarial = bool(record.get("adversarial"))

        ingested = IngestedItem(source_id, item_row["raw_text"], item_row["source_meta"])
        result = pipeline.process(ingested, client=client, conn=conn)
        print(
            f"[{i}/{len(gold)}] {source_id} intent={result.extraction.intent.value} "
            f"route={result.route.value}",
            flush=True,
        )

        processing_row = conn.execute(
            "SELECT processing_ms FROM items WHERE id = ?", (source_id,)
        ).fetchone()
        processing_times.append(processing_row["processing_ms"])
        route_counts[result.route.value] += 1

        expected_intent = ground_truth["intent"]
        predicted_intent = result.extraction.intent.value
        if predicted_intent == expected_intent:
            intent_correct += 1

        if is_adversarial:
            adversarial_total += 1
            if result.route == Route.human_review:
                adversarial_correct += 1

        gold_fields = _gold_section(ground_truth)
        if gold_fields is not None:
            predicted_fields = _predicted_section(result.extraction, expected_intent)
            for field_name, expected_value in gold_fields.items():
                if field_name == "missing_required_fields":
                    continue
                expected_value = _coerce_gold_field(field_name, expected_value)
                predicted_value = predicted_fields.get(field_name)
                field_total[field_name] += 1
                if fields_match(field_name, predicted_value, expected_value):
                    field_correct[field_name] += 1

    avg_processing_ms = sum(processing_times) / len(processing_times) if processing_times else 0.0

    per_field_accuracy = {
        name: field_correct[name] / field_total[name] for name in field_total if field_total[name] > 0
    }
    overall_field_accuracy = (
        sum(field_correct.values()) / sum(field_total.values()) if field_total else 0.0
    )

    n_items = len(gold)
    pct_auto = route_counts.get(Route.auto.value, 0) / n_items if n_items else 0.0
    pct_human_review = route_counts.get(Route.human_review.value, 0) / n_items if n_items else 0.0
    pct_discarded = route_counts.get(Route.discarded.value, 0) / n_items if n_items else 0.0

    manual_minutes_total = n_items * config.MANUAL_BASELINE_MINUTES
    automated_minutes_total = sum(processing_times) / 1000 / 60 if processing_times else 0.0
    hours_saved = (manual_minutes_total - automated_minutes_total) / 60
    dollars_saved = hours_saved * config.STAFF_HOURLY_COST_USD

    metrics = {
        "n_items": n_items,
        "field_accuracy": overall_field_accuracy,
        "per_field_accuracy": per_field_accuracy,
        "intent_accuracy": intent_correct / n_items if n_items else 0.0,
        "adversarial_route_accuracy": (
            adversarial_correct / adversarial_total if adversarial_total else 0.0
        ),
        "pct_auto": pct_auto,
        "pct_human_review": pct_human_review,
        "pct_discarded": pct_discarded,
        "avg_processing_ms": avg_processing_ms,
        "manual_baseline_minutes": config.MANUAL_BASELINE_MINUTES,
        "hours_saved": hours_saved,
        "dollars_saved": dollars_saved,
    }
    storage.save_metrics(conn, metrics)
    return metrics


def print_summary(metrics: dict) -> None:
    print(f"\n=== Benchmark summary ({metrics['n_items']} gold items) ===")
    print(f"Overall field accuracy:      {metrics['field_accuracy']:.1%}")
    for name, acc in sorted(metrics["per_field_accuracy"].items()):
        print(f"  - {name:<24} {acc:.1%}")
    print(f"Intent (classify) accuracy:  {metrics['intent_accuracy']:.1%}")
    print(f"Adversarial -> human_review: {metrics['adversarial_route_accuracy']:.1%}")
    print(
        f"Route mix: auto={metrics['pct_auto']:.1%} "
        f"human_review={metrics['pct_human_review']:.1%} "
        f"discarded={metrics['pct_discarded']:.1%}"
    )
    print(f"Avg processing time:         {metrics['avg_processing_ms']:.0f} ms/item")
    print(
        f"Manual baseline:              {metrics['manual_baseline_minutes']:.1f} min/item -> "
        f"{metrics['hours_saved']:.2f} hours saved, ${metrics['dollars_saved']:.2f} saved "
        f"over {metrics['n_items']} items"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--db", default=config.DB_PATH)
    args = parser.parse_args()

    result_metrics = run_benchmark(Path(args.data_dir), Path(args.db))
    print_summary(result_metrics)
