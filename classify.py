"""Classify step — one cheap/fast LLM call, intent only (SPEC §2 step 2).

Deliberately narrow: this call answers exactly one question (which intent
bucket) so a cheap model (Haiku) is sufficient. Field extraction is a
separate, stronger-model call in extract.py.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel

import config
from schema import Intent

CLASSIFY_SYSTEM_PROMPT = """You triage inbound messages for a residential property \
management company. Classify the message into exactly one intent:

- new_lead: a prospective tenant inquiring about a property/listing.
- application: someone submitting or referencing a rental application.
- existing_tenant: a current tenant with a maintenance, lease, or payment issue.
- spam: unsolicited marketing, scams, or "we buy houses" style solicitations \
  (including ones disguised as a lead).
- other: anything that doesn't fit the above (vendor emails, misc questions).

If the message mixes a lead inquiry with something else (e.g. a maintenance \
complaint), classify by the primary/first request. Base the classification \
only on the message text — do not guess at intent the text doesn't support."""


class _ClassifyResult(BaseModel):
    intent: Intent


def classify(raw_text: str, *, client: anthropic.Anthropic | None = None) -> Intent:
    client = client or anthropic.Anthropic()
    response = client.messages.parse(
        model=config.MODEL_CLASSIFY,
        max_tokens=256,
        system=CLASSIFY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": raw_text}],
        output_format=_ClassifyResult,
    )
    return response.parsed_output.intent
