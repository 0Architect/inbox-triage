"""All model IDs, thresholds, and required-field sets live here — nothing hard-coded
inline in the pipeline modules. See SPEC §13.
"""

import os

from dotenv import load_dotenv

from schema import Intent

load_dotenv()

# --- LLM provider -----------------------------------------------------------
# Never hard-code the key; read it from the environment (.env file or shell export).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# --- Model assignments (SPEC §3) --------------------------------------------
# Classify: fast/cheap, classification is easy.
MODEL_CLASSIFY = "claude-haiku-4-5-20251001"
# Extract: the hard step; accuracy matters most here. Start with Sonnet; only
# escalate to Opus if gold-set accuracy is unacceptable (SPEC §3).
MODEL_EXTRACT = "claude-sonnet-5"
# Draft: templated, low-stakes.
MODEL_DRAFT = "claude-haiku-4-5-20251001"

# --- Required fields per intent (SPEC §5) -----------------------------------
# A lead is "complete" if it has a contact method (email OR phone) + a property
# reference. Encoded as a list of alternatives: each entry is either a single
# field name (all must be present) or a tuple of field names (at least one of
# the tuple must be present).
REQUIRED_FIELDS = {
    Intent.new_lead: [
        ("contact_email", "contact_phone"),
        "property_ref",
    ],
    Intent.application: [
        "applicant_name",
        "desired_unit",
        "monthly_income",
        "screening_consent",
    ],
}

# --- Validation thresholds (SPEC §6) ----------------------------------------
# Fuzzy-match ratio (0-100, via rapidfuzz/thefuzz-style scoring) required for
# property_ref to be considered "found" in raw_text — the hallucination guard.
PROPERTY_MATCH_THRESHOLD = 85

# --- Benchmark / ROI (SPEC §9) ----------------------------------------------
# Assumed minutes a human takes to manually triage one inbound item; used to
# derive the time-saved / $-saved headline metric.
MANUAL_BASELINE_MINUTES = 3
# Fully-loaded hourly cost of the staff member doing manual triage, for the
# $-saved figure.
STAFF_HOURLY_COST_USD = 35.0

# --- Storage ------------------------------------------------------------
DB_PATH = os.environ.get("INBOX_TRIAGE_DB", "data/inbox_triage.db")

# --- Demo abuse guard --------------------------------------------------
# Minimum seconds between "Run pipeline" clicks in the Streamlit live-demo
# page, per browser session. Caps API-cost abuse from someone spamming the
# button without affecting a genuine prospect trying a couple of examples.
DEMO_COOLDOWN_SECONDS = 20
