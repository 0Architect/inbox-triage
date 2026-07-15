"""Extract step — structured extraction against the Extraction schema (SPEC §2 step 3,
§4). This is the hard step; accuracy matters most here, hence the stronger model.

Anti-hallucination discipline (SPEC §4): the model must return None for any field
not clearly present in the source text, and must copy property_ref verbatim
rather than infer it. A None is checkable by validate.py; a confident guess is
what destroys trust in the system.

Implementation note: classify() has already resolved the intent before extract()
runs, so the LLM never needs to choose between the lead/application branches of
the Extraction wrapper. Empirically, the Anthropic structured-outputs endpoint
also rejects (or hangs compiling) a single flat Pydantic schema once it crosses
roughly ten fields — LeadFields (13 fields) and ApplicationFields (11 fields)
both exceed that ceiling on their own. So each intent's fields are split across
two calls of ~6-7 fields, well under the observed limit, and merged in Python
into the canonical LeadFields/ApplicationFields objects.
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

import config
from schema import ApplicationFields, Extraction, Intent, LeadFields, Urgency
from datetime import date

EXTRACT_SYSTEM_PROMPT = """You extract structured fields from a single inbound message \
for a residential property management company. The message has already been \
classified with intent = {intent}.

Rules — follow these exactly, they are what makes this system trustworthy:
- For any field not CLEARLY stated in the message, return null. Do not guess, \
  infer, or fill in a plausible-sounding value. A null is fine and expected — \
  most messages are incomplete.
- Any address or listing reference field MUST be copied verbatim from the source \
  text (same characters, including any typos). Never infer or construct one the \
  text doesn't literally contain.
- A "raw budget"-style field must be the verbatim budget phrase as written \
  (e.g. "$1500/mo"); a numeric budget field is your best-effort parse of that phrase.
- Set `urgent_flag` true only for genuine emergencies, legal matters, or explicit \
  time-critical language — not for ordinary requests phrased politely.
- Dates must be real calendar dates. If a date is ambiguous or not a real date, \
  leave the field null rather than guessing."""


class _LeadGroupA(BaseModel):
    urgent_flag: bool = False
    contact_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    property_ref: str | None = Field(
        None, description="Address or listing ID. MUST be copied verbatim from the source text."
    )
    desired_move_in: date | None = None
    budget_max: float | None = None


class _LeadGroupB(BaseModel):
    budget_raw: str | None = Field(None, description="Verbatim budget phrase, e.g. '$1500/mo'")
    bedrooms: int | None = None
    pets: bool | None = None
    pets_detail: str | None = None
    stated_income: str | None = None
    lead_source: str | None = Field(None, description="e.g. Zillow, Apartments.com, direct, referral")
    urgency: Urgency | None = None


class _ApplicationGroupA(BaseModel):
    urgent_flag: bool = False
    applicant_name: str | None = None
    co_applicants: list[str] = Field(default_factory=list)
    current_address: str | None = None
    employer: str | None = None
    monthly_income: float | None = None


class _ApplicationGroupB(BaseModel):
    desired_unit: str | None = Field(
        None, description="Address or listing ID. MUST be copied verbatim from the source text."
    )
    desired_move_in: date | None = None
    occupants: int | None = None
    pets: bool | None = None
    pets_detail: str | None = None
    screening_consent: bool | None = None


class _UrgencyOnly(BaseModel):
    urgent_flag: bool = False


def _parse(client: anthropic.Anthropic, raw_text: str, intent: Intent, output_format):
    response = client.messages.parse(
        model=config.MODEL_EXTRACT,
        max_tokens=1024,
        system=EXTRACT_SYSTEM_PROMPT.format(intent=intent.value),
        messages=[{"role": "user", "content": raw_text}],
        output_format=output_format,
    )
    return response.parsed_output


def extract(raw_text: str, intent: Intent, *, client: anthropic.Anthropic | None = None) -> Extraction:
    client = client or anthropic.Anthropic()

    if intent == Intent.new_lead:
        group_a = _parse(client, raw_text, intent, _LeadGroupA)
        group_b = _parse(client, raw_text, intent, _LeadGroupB)
        lead = LeadFields(**group_a.model_dump(exclude={"urgent_flag"}), **group_b.model_dump())
        return Extraction(intent=intent, urgent_flag=group_a.urgent_flag, lead=lead)

    if intent == Intent.application:
        group_a = _parse(client, raw_text, intent, _ApplicationGroupA)
        group_b = _parse(client, raw_text, intent, _ApplicationGroupB)
        application = ApplicationFields(
            **group_a.model_dump(exclude={"urgent_flag"}), **group_b.model_dump()
        )
        return Extraction(intent=intent, urgent_flag=group_a.urgent_flag, application=application)

    urgency_result = _parse(client, raw_text, intent, _UrgencyOnly)
    return Extraction(intent=intent, urgent_flag=urgency_result.urgent_flag)
