"""Draft step — templated reply for AUTO-routed items (SPEC §2 step 6). Low-stakes
and templated, so the cheap/fast model is enough.
"""

from __future__ import annotations

import anthropic

import config
from schema import Extraction, Intent

DRAFT_SYSTEM_PROMPT = """You draft a short reply on behalf of a residential property \
management company, responding to an inbound message that has already been \
validated and is complete enough to act on automatically.

Rules:
- Reference only the fields given below — do not invent details not provided.
- Keep it brief (3-5 sentences), professional, and warm.
- For a lead: confirm interest, reference the property, and propose next steps \
  (e.g. scheduling a viewing).
- For an application: acknowledge receipt and state what happens next \
  (screening/background check review).
- Never promise approval, a specific price, or availability you don't have \
  confirmed information about.
- Output only the reply body — no subject line, no signature block."""


def _fields_summary(extraction: Extraction) -> str:
    if extraction.intent == Intent.new_lead and extraction.lead is not None:
        return extraction.lead.model_dump_json(exclude_none=True)
    if extraction.intent == Intent.application and extraction.application is not None:
        return extraction.application.model_dump_json(exclude_none=True)
    return "{}"


def draft(extraction: Extraction, *, client: anthropic.Anthropic | None = None) -> str:
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=config.MODEL_DRAFT,
        max_tokens=512,
        system=DRAFT_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Intent: {extraction.intent.value}\n"
                    f"Validated fields: {_fields_summary(extraction)}"
                ),
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")
