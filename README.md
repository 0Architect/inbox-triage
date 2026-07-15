# Inbox Triage

Processes inbound emails and applications for property managers. **97.1% field-extraction accuracy** on messy real-world input. **100% of ambiguous/adversarial items** correctly flagged for human review instead of auto-processed. Cuts per-item triage from **~3 min to 15.6 sec**.

**🔗 Live demo:** https://inbox-triage-dqkzawrh3whmn6lebvx9yb.streamlit.app/

---

Almost every "AI inbox triage" build confidently extracts fields and confidently gets some of them wrong — a hallucinated address, a guessed move-in date, a lead silently auto-replied to with the wrong unit number. This one doesn't. Every extracted field is checked against the source text (a fuzzy-match hallucination guard on any address/listing reference), every required field is verified present before anything auto-sends, and anything genuinely ambiguous — mixed intent, missing info, an emergency buried in a routine inquiry — gets routed to a human review queue instead of guessed at. **The moat isn't the extraction accuracy. It's that the system knows what it doesn't know, and proves it: 100% of the deliberately ambiguous test cases were caught, zero false auto-sends.**

## What it does

Ingests a raw inbound email (or PDF attachment, read natively via vision — no OCR pipeline) and runs it through a deterministic pipeline:

```
Ingest → Classify → Extract → Validate → Route ──┬── AUTO:  drafted reply
  (raw)   (intent)  (fields)  (Python)  (Python)  │
                                                   └── HUMAN REVIEW: flagged with reasons
```

- **Classify** — intent (new lead / application / existing tenant / spam / other), one fast model call.
- **Extract** — structured fields (contact info, property reference, budget, move-in date, etc.), one stronger-model call per intent. The model is instructed to return `null` for anything not clearly stated — never to guess.
- **Validate** — plain Python, zero LLM judgment calls here: email/phone format checks, move-in date sanity, and a fuzzy-match hallucination guard that rejects any property reference the model didn't actually copy from the source text.
- **Route** — deterministic gate. Any validation failure, any missing required field, or an urgent flag sends the item to a human review queue with the exact reason attached. Everything else gets an auto-drafted reply.

This is a pipeline, not an agent — there's no LLM deciding what to do next. Every routing decision is a plain `if` statement you can read and audit.

## Results

Benchmarked live against a 300-item synthetic messy dataset (typos, forwarded-email noise, partial info, mixed-intent messages, spam disguised as leads) — 110 gold-labeled items including 10 deliberately adversarial cases (hallucination-bait, missing required fields, buried emergencies):

| Metric | Result |
|---|---|
| Field-extraction accuracy | 97.1% |
| Intent classification accuracy | 99.1% |
| Adversarial items → human review | 100% |
| Avg. processing time | 15.6 sec/item (vs. ~3 min manual) |
| Auto / human-review / discarded split | 55% / 35% / 11% |

Full per-field breakdown and the current live numbers are on the Metrics page in the demo above.

## Running it locally

```bash
git clone https://github.com/0Architect/inbox-triage.git
cd inbox-triage
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
.venv/bin/streamlit run app.py
```

Three pages: **Live demo** (paste an email, watch it classify → extract → route → draft), **Review queue** (everything flagged for human review, with the specific reason and an edit/approve flow), **Metrics** (the numbers above, live).

```bash
# Regenerate the synthetic messy dataset
.venv/bin/python generate_data.py --n 300 --gold 100 --adversarial 10

# Re-run the benchmark against the gold set
.venv/bin/python benchmark.py

# Run the offline test suite (no API calls)
.venv/bin/python -m pytest tests/
```

## Stack

- **Python 3.11+**, Pydantic v2 as the extraction contract
- **Claude Sonnet 5** for extraction (the accuracy-critical step), **Claude Haiku 4.5** for classification and drafting (cheap, fast) — all model IDs configurable in `config.py`
- `email-validator`, `phonenumbers`, `rapidfuzz` for deterministic field validation
- SQLite for storage — one file, zero infra
- Streamlit for the demo/review-queue/metrics UI

No LangChain, no agent framework, no vector DB. Orchestration is a single `process(item) -> TriageResult` function.

## Other work

**[Kworva](https://kworva.netlify.app)** — a campus peer-to-peer marketplace (Android, live beta). [Repo](https://github.com/0Architect/kworva). Included as proof of end-to-end product delivery beyond AI automation work.
