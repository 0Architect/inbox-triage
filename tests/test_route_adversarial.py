"""SPEC §11 step 5: every adversarial/ item must route to human_review.

Runs validate() + route() against each adversarial item's ground-truth
extraction (a stand-in for a correct LLM extraction, since this offline test
suite has no API key) to confirm the *routing rules themselves* are strict
enough — a real extraction pass will be re-verified once live calls exist.
"""

import json
from pathlib import Path

from schema import ApplicationFields, Extraction, LeadFields, Route
from route import route
from validate import validate

DATA_DIR = Path(__file__).parent.parent / "data"


def _load_adversarial():
    items = {json.loads(l)["source_id"]: json.loads(l) for l in open(DATA_DIR / "items.jsonl")}
    gold = [json.loads(l) for l in open(DATA_DIR / "gold.jsonl") if json.loads(l).get("adversarial")]
    return items, gold


def _extraction_from_ground_truth(gt: dict) -> Extraction:
    data = dict(gt)
    if data.get("lead") is not None:
        data["lead"] = LeadFields(**data["lead"])
    if data.get("application") is not None:
        data["application"] = ApplicationFields(**data["application"])
    return Extraction(**data)


def test_all_adversarial_items_route_to_human_review():
    items, gold = _load_adversarial()
    assert len(gold) == 10

    failures = []
    for record in gold:
        source_id = record["source_id"]
        raw_text = items[source_id]["raw_text"]
        extraction = _extraction_from_ground_truth(record["ground_truth"])
        validation = validate(extraction, raw_text)
        chosen_route, reasons = route(extraction, validation)
        if chosen_route != Route.human_review:
            failures.append((source_id, chosen_route, reasons))

    assert not failures, f"adversarial items that did NOT route to human_review: {failures}"
