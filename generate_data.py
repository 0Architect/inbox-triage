"""Synthetic messy-inbound dataset generator — SPEC §8.

Produces three artifacts under data/:
  - items.jsonl        every generated item: source_id, raw_text, source_meta, true_intent
  - gold.jsonl         ground-truth field values for a hand-verifiable subset (the benchmark denominator)
  - adversarial/*.txt   deliberately ambiguous items that MUST route to human_review

The mess is the point. Do not sand the noise down to make numbers look better later.

Usage:
    python generate_data.py --n 50 --gold 20 --adversarial 10 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ADVERSARIAL_DIR = DATA_DIR / "adversarial"

# ---------------------------------------------------------------------------
# Synthetic personas / properties (fake data only — no real PII, SPEC §8)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Maria", "James", "Aisha", "Wei", "Carlos", "Priya", "Liam", "Fatima",
    "Noah", "Sofia", "Ethan", "Yuki", "Omar", "Grace", "Diego", "Chloe",
    "Kwame", "Elena", "Ravi", "Hannah",
]
LAST_NAMES = [
    "Nguyen", "Garcia", "Smith", "Patel", "Johnson", "Kim", "Rossi",
    "Okafor", "Silva", "Muller", "Chen", "Ivanov", "Novak", "Haddad",
    "Larsen", "Costa", "Abara", "Fischer", "Doyle", "Sato",
]
STREETS = [
    "Maple Ave", "5th St", "Sunset Blvd", "Oakwood Dr", "Birchwood Ln",
    "Riverside Dr", "Cedar Ct", "Highland Ave", "Elm St", "Franklin Pkwy",
    "Willow Way", "Crescent Rd", "Magnolia St", "Harbor View Dr", "Pine St",
]
CITIES = [
    ("Springfield", "IL"), ("Riverton", "NJ"), ("Fairview", "TX"),
    ("Georgetown", "CO"), ("Clayton", "OH"), ("Bristol", "CT"),
    ("Ashland", "OR"), ("Salem", "MA"), ("Manchester", "NH"), ("Dover", "DE"),
]
UNIT_LABELS = ["Apt 1", "Apt 2B", "Unit 4B", "#3", "Suite 100", "Unit 12", None]
LEAD_SOURCES = ["Zillow", "Apartments.com", "direct", "referral", "Craigslist", "Facebook Marketplace"]
EMPLOYERS = [
    "Meridian Logistics", "Brightline Health", "Cascade Software",
    "Union Freight Co.", "Oakridge Consulting", "Delta Retail Group",
    "Northstar Manufacturing", "Ivy Lane Design", "Quantum Analytics",
]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "proton.me", "icloud.com"]
SIGNATURES = [
    "\n\nSent from my iPhone",
    "\n\nBest,\n{name}",
    "\n\nThanks,\n{name}\n{phone}",
    "\n\n--\n{name}\nSent via mobile",
    "",
]
LEGAL_DISCLAIMER = (
    "\n\nCONFIDENTIALITY NOTICE: This email and any attachments are intended solely "
    "for the addressee(s) and may contain confidential information. If you received "
    "this in error, please notify the sender and delete this message."
)


def _rand_phone(rng: random.Random) -> str:
    return f"({rng.randint(200, 989)}) {rng.randint(200, 989)}-{rng.randint(1000, 9999)}"


def _rand_email(rng: random.Random, name: str) -> str:
    first, last = name.lower().split(" ", 1)
    last = last.replace(" ", "")
    sep = rng.choice([".", "_", ""])
    num = rng.choice(["", str(rng.randint(1, 99))])
    return f"{first}{sep}{last}{num}@{rng.choice(EMAIL_DOMAINS)}"


def _rand_address(rng: random.Random) -> str:
    unit = rng.choice(UNIT_LABELS)
    city, state = rng.choice(CITIES)
    base = f"{rng.randint(100, 9999)} {rng.choice(STREETS)}"
    if unit:
        base += f", {unit}"
    return f"{base}, {city}, {state}"


def _rand_date(rng: random.Random) -> str:
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    year = 2026 if month >= 7 else 2027
    return f"{year}-{month:02d}-{day:02d}"


def _persona(rng: random.Random) -> dict:
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    return {
        "name": name,
        "email": _rand_email(rng, name),
        "phone": _rand_phone(rng),
    }


# ---------------------------------------------------------------------------
# Noise injectors
# ---------------------------------------------------------------------------

_TYPO_SUBS = {"the": "teh", "you": "u", "your": "ur", "please": "pls",
              "thanks": "thx", "with": "w/", "before": "b4", "for": "4",
              "are": "r", "to": "2"}


def add_txtspeak(text: str, rng: random.Random, rate: float = 0.25) -> str:
    words = text.split(" ")
    out = []
    for w in words:
        stripped = w.strip(".,!?").lower()
        if stripped in _TYPO_SUBS and rng.random() < rate:
            repl = _TYPO_SUBS[stripped]
            trail = w[len(stripped):] if w.lower().endswith(stripped) else ""
            out.append(repl + trail)
        else:
            out.append(w)
    return " ".join(out)


def add_char_typos(text: str, rng: random.Random, rate: float = 0.04) -> str:
    chars = list(text)
    i = 0
    out = []
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate and i + 1 < len(chars) and chars[i + 1].isalpha():
            out.append(chars[i + 1])
            out.append(c)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def wrap_forwarded_chain(text: str, rng: random.Random, persona: dict) -> str:
    depth = rng.randint(1, 3)
    body = text
    for _ in range(depth):
        quoted = "\n".join(f"> {line}" for line in body.split("\n"))
        header = (
            f"---------- Forwarded message ---------\n"
            f"From: {persona['name']} <{persona['email']}>\n"
            f"Date: Mon, {rng.randint(1,28)} Jun 2026\n"
            f"Subject: Fwd: inquiry\n\n"
        )
        body = header + quoted
    return body


def add_signature(text: str, rng: random.Random, persona: dict) -> str:
    sig = rng.choice(SIGNATURES).format(name=persona["name"], phone=persona["phone"])
    return text + sig


def maybe_add_disclaimer(text: str, rng: random.Random, p: float = 0.3) -> str:
    return text + LEGAL_DISCLAIMER if rng.random() < p else text


# ---------------------------------------------------------------------------
# Templates — each returns (raw_text, ground_truth_extraction_dict)
# ---------------------------------------------------------------------------

def gen_lead(rng: random.Random, adversarial: bool = False) -> tuple[str, dict]:
    p = _persona(rng)
    address = _rand_address(rng)
    bedrooms = rng.choice([1, 1, 2, 2, 3, None])
    budget = rng.choice([1200, 1450, 1500, 1800, 2100, 2400, None])
    move_in = _rand_date(rng) if rng.random() < 0.7 else None
    pets = rng.choice([True, False, None])
    source = rng.choice(LEAD_SOURCES)

    include_email = rng.random() < 0.8
    include_phone = rng.random() < 0.5 or not include_email  # ensure at least one contact usually
    include_property = rng.random() < 0.9 if not adversarial else rng.random() < 0.3

    # Each entry: (line_text, field_names it establishes). Kept in lockstep with
    # ground_truth so a "partial info" cut of the message also nulls the right fields
    # — the gold set must never claim a field is present when the text doesn't have it.
    segments: list[tuple[str, list[str]]] = [(f"Hi, my name is {p['name']}.", ["contact_name"])]
    if include_property:
        segments.append((f"I'm interested in the listing at {address}.", ["property_ref"]))
    else:
        segments.append(("I saw your listing online and wanted to ask about availability.", []))
    if bedrooms:
        segments.append((f"Looking for something with {bedrooms} bedroom{'s' if bedrooms != 1 else ''}.", ["bedrooms"]))
    if budget:
        segments.append((f"My budget is around ${budget}/mo.", ["budget_max", "budget_raw"]))
    if move_in:
        segments.append((f"I'd like to move in by {move_in}.", ["desired_move_in"]))
    if pets is True:
        segments.append(("I have a small dog, is that ok?", ["pets", "pets_detail"]))
    elif pets is False:
        segments.append(("No pets.", ["pets"]))
    segments.append((f"Found you on {source}.", ["lead_source"]))
    contact_line = []
    contact_fields = []
    if include_email:
        contact_line.append(f"Email: {p['email']}")
        contact_fields.append("contact_email")
    if include_phone:
        contact_line.append(f"Phone: {p['phone']}")
        contact_fields.append("contact_phone")
    if contact_line:
        segments.append((" ".join(contact_line), contact_fields))

    # "Partial info" noise: keep only a random prefix of segments, so the fields it
    # drops are nulled in ground_truth rather than post-hoc stripped from the text.
    if not adversarial and rng.random() < 0.15:
        keep = max(1, rng.randint(1, len(segments)))
        segments = segments[:keep]

    present_fields = {f for _, fields in segments for f in fields}
    lines = [s for s, _ in segments]
    text = " ".join(lines)
    text = add_signature(text, rng, p)

    def keep(field: str, value):
        return value if field in present_fields else None

    ground_truth = {
        "intent": "new_lead",
        "urgent_flag": False,
        "lead": {
            "contact_name": keep("contact_name", p["name"]),
            "contact_email": keep("contact_email", p["email"]) if include_email else None,
            "contact_phone": keep("contact_phone", p["phone"]) if include_phone else None,
            "property_ref": keep("property_ref", address) if include_property else None,
            "desired_move_in": keep("desired_move_in", move_in),
            "budget_max": keep("budget_max", float(budget)) if budget else None,
            "budget_raw": keep("budget_raw", f"${budget}/mo") if budget else None,
            "bedrooms": keep("bedrooms", bedrooms),
            "pets": keep("pets", pets),
            "pets_detail": keep("pets_detail", "small dog") if pets is True else None,
            "stated_income": None,
            "lead_source": keep("lead_source", source),
            # None, not a hardcoded "normal" — the generated text never states an
            # urgency level, so claiming one in ground truth would itself violate
            # the "don't expect the extractor to invent what isn't there" rule.
            "urgency": None,
        },
        "application": None,
    }
    return text, ground_truth


def gen_application(rng: random.Random) -> tuple[str, dict]:
    p = _persona(rng)
    co_applicant = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}" if rng.random() < 0.3 else None
    unit = _rand_address(rng)
    income = rng.choice([3200, 4100, 4800, 5200, 6000, 7200])
    employer = rng.choice(EMPLOYERS)
    move_in = _rand_date(rng)
    occupants = rng.randint(1, 4)
    pets = rng.choice([True, False, None])
    consent = rng.random() < 0.85

    segments: list[tuple[str, list[str]]] = [
        (f"Hello, I'd like to submit a rental application for {unit}.", ["desired_unit"]),
        (f"Full name: {p['name']}.", ["applicant_name"]),
    ]
    if co_applicant:
        segments.append((f"My co-applicant is {co_applicant}.", ["co_applicants"]))
    segments.append((f"I currently work at {employer}, monthly income ${income}.", ["employer", "monthly_income"]))
    segments.append((f"Planning to move in {move_in}, {occupants} occupant{'s' if occupants != 1 else ''} total.",
                      ["desired_move_in", "occupants"]))
    if pets is not None:
        segments.append(("We have a cat." if pets else "No pets.", ["pets", "pets_detail"]))
    if consent:
        segments.append(("I consent to a background and credit check.", ["screening_consent"]))

    if rng.random() < 0.15:
        keep = max(1, rng.randint(1, len(segments)))
        segments = segments[:keep]

    present_fields = {f for _, fields in segments for f in fields}
    text = " ".join(s for s, _ in segments)
    text = add_signature(text, rng, p)

    def keep(field: str, value):
        return value if field in present_fields else None

    ground_truth = {
        "intent": "application",
        "urgent_flag": False,
        "lead": None,
        "application": {
            "applicant_name": keep("applicant_name", p["name"]),
            "co_applicants": [co_applicant] if co_applicant and "co_applicants" in present_fields else [],
            "current_address": None,
            "employer": keep("employer", employer),
            "monthly_income": keep("monthly_income", float(income)),
            "desired_unit": keep("desired_unit", unit),
            "desired_move_in": keep("desired_move_in", move_in),
            "occupants": keep("occupants", occupants),
            "pets": keep("pets", pets),
            "pets_detail": keep("pets_detail", "cat") if pets is True else None,
            "screening_consent": keep("screening_consent", consent) if consent else None,
        },
    }
    return text, ground_truth


def gen_existing_tenant(rng: random.Random) -> tuple[str, dict]:
    p = _persona(rng)
    unit = _rand_address(rng)
    issues = [
        "the sink in the kitchen is leaking",
        "the heater stopped working",
        "there's a strange noise coming from unit upstairs",
        "my rent payment didn't go through this month",
        "the lock on the front door is broken",
        "I need to renew my lease",
    ]
    urgent = rng.random() < 0.3
    issue = rng.choice(issues)
    text = f"Hi, this is {p['name']} from {unit}. {issue.capitalize()}. Please let me know next steps."
    if urgent:
        text += " This is urgent, please respond ASAP."
    text = add_signature(text, rng, p)
    ground_truth = {"intent": "existing_tenant", "urgent_flag": urgent, "lead": None, "application": None}
    return text, ground_truth


def gen_spam(rng: random.Random, lead_shaped: bool = False) -> tuple[str, dict]:
    if lead_shaped:
        text = rng.choice([
            "We buy houses fast for CASH! No repairs needed, close in 7 days. Call now!!!",
            "Investor looking to purchase properties in your area - any condition, cash offer within 24hrs.",
            "Get top dollar for your rental property - free no-obligation quote today!",
        ])
    else:
        text = rng.choice([
            "CONGRATULATIONS!! You've been selected for a free cruise! Click here to claim.",
            "Increase your credit score instantly - act now, limited spots!",
            "Cheap meds online no prescription needed, huge discounts this week only.",
        ])
    ground_truth = {"intent": "spam", "urgent_flag": False, "lead": None, "application": None}
    return text, ground_truth


def gen_mixed_intent(rng: random.Random) -> tuple[str, dict]:
    """Lead inquiry + unrelated maintenance complaint in one message — adversarial-flavored.

    SPEC §6's routing gate has no dedicated "mixed intent" rule — only validation
    failures, missing required fields, and urgent_flag trigger human_review. A
    genuine emergency bundled into the message is the correct (not contrived) way
    for this to land in the review queue, per SPEC §8's requirement that every
    adversarial item route to human_review.

    The secondary complaint must be UNAMBIGUOUSLY an emergency, not just phrased
    as a maintenance request — a live benchmark run showed a mildly-worded "my
    sink is leaking, can someone fix that?" gets extracted as urgent_flag=False
    (correctly, per extract.py's own "genuine emergencies... explicit
    time-critical language" bar), which silently lets the mixed intent slip past
    the gate as a clean AUTO lead. Explicit urgency language closes that gap.
    """
    lead_text, lead_truth = gen_lead(rng)
    unit = _rand_address(rng)
    emergency = rng.choice([
        f"URGENT — separately, there's a burst pipe flooding my unit at {unit} right now, I need someone here immediately.",
        f"Also, this is urgent: I smell gas at {unit} and need someone out here right away.",
        f"One more thing, this can't wait: there's a small fire risk from an outlet at {unit}, please send someone ASAP.",
    ])
    text = f"{lead_text} {emergency}"
    lead_truth = {**lead_truth, "urgent_flag": True}
    return text, lead_truth


def gen_hallucination_bait(rng: random.Random) -> tuple[str, dict]:
    """Mentions a property vaguely without ever stating it clearly — model must NOT invent one."""
    p = _persona(rng)
    city, state = rng.choice(CITIES)
    text = (
        f"Hi, my name is {p['name']}, I saw one of your listings somewhere in {city} "
        f"and wanted to ask if it's still available. Not sure of the exact address, "
        f"someone sent me a screenshot. Email: {p['email']}"
    )
    ground_truth = {
        "intent": "new_lead",
        "urgent_flag": False,
        "lead": {
            "contact_name": p["name"], "contact_email": p["email"], "contact_phone": None,
            "property_ref": None,  # deliberately absent — hallucination-bait
            "desired_move_in": None, "budget_max": None, "budget_raw": None,
            "bedrooms": None, "pets": None, "pets_detail": None, "stated_income": None,
            "lead_source": None, "urgency": None,
        },
        "application": None,
    }
    return text, ground_truth


def gen_missing_critical(rng: random.Random) -> tuple[str, dict]:
    """Application missing required fields (no income, no consent) — must flag incomplete."""
    p = _persona(rng)
    unit = _rand_address(rng)
    text = f"Hi, I want to apply for {unit}. My name is {p['name']}. Let me know what's next."
    ground_truth = {
        "intent": "application",
        "urgent_flag": False,
        "lead": None,
        "application": {
            "applicant_name": p["name"], "co_applicants": [], "current_address": None,
            "employer": None, "monthly_income": None, "desired_unit": unit,
            "desired_move_in": None, "occupants": None, "pets": None, "pets_detail": None,
            "screening_consent": None,
        },
    }
    return text, ground_truth


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

GENERATORS = [
    ("new_lead", gen_lead, 0.40),
    ("application", gen_application, 0.25),
    ("existing_tenant", gen_existing_tenant, 0.20),
    ("spam", lambda rng: gen_spam(rng, lead_shaped=rng.random() < 0.4), 0.15),
]

ADVERSARIAL_GENERATORS = [gen_mixed_intent, gen_hallucination_bait, gen_missing_critical]


def apply_random_noise(text: str, rng: random.Random, persona_for_forward: dict) -> str:
    if rng.random() < 0.35:
        text = wrap_forwarded_chain(text, rng, persona_for_forward)
    if rng.random() < 0.5:
        text = add_txtspeak(text, rng)
    if rng.random() < 0.3:
        text = add_char_typos(text, rng)
    text = maybe_add_disclaimer(text, rng)
    return text


def weighted_choice(rng: random.Random, generators: list[tuple[str, callable, float]]):
    weights = [w for _, _, w in generators]
    return rng.choices(generators, weights=weights, k=1)[0]


def generate_items(n: int, rng: random.Random) -> list[dict]:
    items = []
    for idx in range(n):
        intent_name, fn, _ = weighted_choice(rng, GENERATORS)
        text, truth = fn(rng)
        persona = _persona(rng)
        text = apply_random_noise(text, rng, persona)
        source_id = f"item-{idx:04d}"
        items.append({
            "source_id": source_id,
            "raw_text": text,
            "source_meta": {"channel": "email", "filename": None},
            "true_intent": intent_name,
            "_ground_truth": truth,
        })
    return items


def generate_adversarial(n: int, rng: random.Random) -> list[dict]:
    items = []
    for idx in range(n):
        fn = ADVERSARIAL_GENERATORS[idx % len(ADVERSARIAL_GENERATORS)]
        text, truth = fn(rng)
        persona = _persona(rng)
        if rng.random() < 0.5:
            text = apply_random_noise(text, rng, persona)
        source_id = f"adv-{idx:03d}"
        items.append({
            "source_id": source_id,
            "raw_text": text,
            "source_meta": {"channel": "email", "filename": None},
            "true_intent": truth["intent"],
            "_ground_truth": truth,
            "adversarial": True,
        })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="total non-adversarial items")
    ap.add_argument("--gold", type=int, default=20, help="how many items also get gold labels")
    ap.add_argument("--adversarial", type=int, default=10, help="adversarial items")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    DATA_DIR.mkdir(exist_ok=True)
    ADVERSARIAL_DIR.mkdir(exist_ok=True)

    items = generate_items(args.n, rng)
    adversarial_items = generate_adversarial(args.adversarial, rng)

    gold_n = min(args.gold, len(items))
    gold_ids = set(x["source_id"] for x in rng.sample(items, gold_n))

    items_path = DATA_DIR / "items.jsonl"
    gold_path = DATA_DIR / "gold.jsonl"

    with items_path.open("w") as f:
        for it in items + adversarial_items:
            record = {k: v for k, v in it.items() if k != "_ground_truth"}
            f.write(json.dumps(record) + "\n")

    with gold_path.open("w") as f:
        for it in items:
            if it["source_id"] in gold_ids:
                f.write(json.dumps({
                    "source_id": it["source_id"],
                    "ground_truth": it["_ground_truth"],
                }) + "\n")
        # every adversarial item is gold too — they anchor the routing benchmark
        for it in adversarial_items:
            f.write(json.dumps({
                "source_id": it["source_id"],
                "ground_truth": it["_ground_truth"],
                "adversarial": True,
            }) + "\n")

    for it in adversarial_items:
        (ADVERSARIAL_DIR / f"{it['source_id']}.txt").write_text(it["raw_text"])

    print(f"Wrote {len(items)} items -> {items_path}")
    print(f"Wrote {gold_n + len(adversarial_items)} gold records -> {gold_path}")
    print(f"Wrote {len(adversarial_items)} adversarial items -> {ADVERSARIAL_DIR}/")

    intent_counts = {}
    for it in items:
        intent_counts[it["true_intent"]] = intent_counts.get(it["true_intent"], 0) + 1
    print("Intent mix:", intent_counts)


if __name__ == "__main__":
    main()
