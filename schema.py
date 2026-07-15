"""The contract. Every other module (extract, validate, route, draft, benchmark) builds
against these Pydantic models. Get this right before writing anything else — see SPEC §4.
"""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    new_lead = "new_lead"
    application = "application"
    existing_tenant = "existing_tenant"
    spam = "spam"
    other = "other"


class Urgency(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"


class LeadFields(BaseModel):
    contact_name: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    property_ref: str | None = Field(
        None,
        description=(
            "Address or listing ID the inquiry is about. "
            "MUST be copied from the source text — do not infer."
        ),
    )
    desired_move_in: date | None = None
    budget_max: float | None = None
    budget_raw: str | None = Field(None, description="Verbatim budget phrase, e.g. '$1500/mo'")
    bedrooms: int | None = None
    pets: bool | None = None
    pets_detail: str | None = None
    stated_income: str | None = None
    lead_source: str | None = Field(
        None, description="e.g. Zillow, Apartments.com, direct, referral"
    )
    urgency: Urgency | None = None


class ApplicationFields(BaseModel):
    applicant_name: str | None = None
    co_applicants: list[str] = Field(default_factory=list)
    current_address: str | None = None
    employer: str | None = None
    monthly_income: float | None = None
    desired_unit: str | None = None
    desired_move_in: date | None = None
    occupants: int | None = None
    pets: bool | None = None
    pets_detail: str | None = None
    screening_consent: bool | None = None
    # Computed by a validator, NOT by the LLM — see SPEC §5.
    missing_required_fields: list[str] = Field(default_factory=list)


class Extraction(BaseModel):
    intent: Intent
    urgent_flag: bool = Field(False, description="Cross-cutting: emergency/legal/time-critical")
    lead: LeadFields | None = None  # populated iff intent == new_lead
    application: ApplicationFields | None = None  # populated iff intent == application


class ValidationResult(BaseModel):
    passed: bool
    failed_checks: list[str] = Field(default_factory=list)  # human-readable reasons


class Route(str, Enum):
    auto = "auto"
    human_review = "human_review"
    discarded = "discarded"  # spam


class TriageResult(BaseModel):
    source_id: str
    raw_text: str
    extraction: Extraction
    validation: ValidationResult
    route: Route
    draft_reply: str | None = None
    reasons: list[str] = Field(default_factory=list)  # why this route was chosen
