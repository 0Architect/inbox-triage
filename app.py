"""Streamlit multipage app — SPEC §10.

Page 1 (Live demo) is the 60-second close: paste an email or upload a PDF,
run the pipeline live, and show classify -> extract (per-field checks) ->
route (with reasons) -> draft. Page 2 (Review queue) is the human-in-the-loop
feature made visible — a first-class product feature, not an error path.
Page 3 (Metrics) shows the headline accuracy / $-saved number.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anthropic
import streamlit as st

import config
import storage
from ingest import ingest_email_with_attachment
from pipeline import process
from schema import Route
from validate import missing_required_fields as compute_missing_required_fields

st.set_page_config(page_title="Inbox Triage", layout="wide")


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@st.cache_resource
def get_conn():
    return storage.get_connection(config.DB_PATH)


# --- shared helpers ----------------------------------------------------------

FIELD_CHECK_KEYWORDS = {
    "contact_email": "invalid email format",
    "contact_phone": "invalid phone number",
    "desired_move_in": "move-in date invalid or in the past",
    "property_ref": "hallucination",
    "desired_unit": "hallucination",
}


def field_status(field_name: str, value, failed_checks: list[str], missing: list[str]) -> str:
    """Returns 'pass', 'fail', or 'empty' for a single extracted field."""
    keyword = FIELD_CHECK_KEYWORDS.get(field_name)
    if keyword and any(keyword in c for c in failed_checks):
        return "fail"
    if field_name in missing:
        return "fail"
    if value in (None, [], ""):
        return "empty"
    return "pass"


STATUS_ICON = {"pass": "✅", "fail": "❌", "empty": "➖"}


def render_fields_table(fields_dict: dict, failed_checks: list[str], missing: list[str]) -> None:
    for name, value in fields_dict.items():
        if name == "missing_required_fields":
            continue
        status = field_status(name, value, failed_checks, missing)
        st.markdown(f"{STATUS_ICON[status]} **{name}**: {value if value not in (None, '') else '_(not provided)_'}")


def run_pipeline_on_text(raw_text: str, source_id: str, pdf_path: str | None = None):
    client = get_client()
    conn = get_conn()
    item = ingest_email_with_attachment(source_id, raw_text, pdf_path, client=client)
    return process(item, client=client, conn=conn)


# --- pages ---------------------------------------------------------------


def page_live_demo() -> None:
    st.title("Live demo")
    st.caption("Paste an inbound email (and optionally a PDF attachment) and run the pipeline live.")

    email_body = st.text_area("Email body", height=200, placeholder="Paste the raw email text here...")
    uploaded_pdf = st.file_uploader("Optional PDF attachment", type=["pdf"])
    source_id = st.text_input("Source ID (for the demo record)", value="demo-item")

    if st.button("Run pipeline", type="primary", disabled=not email_body.strip()):
        pdf_path = None
        if uploaded_pdf is not None:
            tmp_dir = Path(tempfile.mkdtemp())
            pdf_path = tmp_dir / uploaded_pdf.name
            pdf_path.write_bytes(uploaded_pdf.getvalue())

        with st.spinner("Classifying, extracting, validating, routing..."):
            result = run_pipeline_on_text(email_body, source_id, str(pdf_path) if pdf_path else None)

        st.subheader("1. Intent")
        st.write(f"**{result.extraction.intent.value}**" + (" — urgent" if result.extraction.urgent_flag else ""))

        st.subheader("2. Extracted fields")
        fields = None
        if result.extraction.lead is not None:
            fields = result.extraction.lead.model_dump(mode="json")
            missing = compute_missing_required_fields(result.extraction)
        elif result.extraction.application is not None:
            fields = result.extraction.application.model_dump(mode="json")
            missing = fields.get("missing_required_fields", [])
        else:
            missing = []

        if fields:
            render_fields_table(fields, result.validation.failed_checks, missing)
        else:
            st.write("_No structured fields for this intent._")

        st.subheader("3. Route decision")
        route_color = {
            Route.auto: "green",
            Route.human_review: "orange",
            Route.discarded: "gray",
        }[result.route]
        st.markdown(f":{route_color}[**{result.route.value.upper()}**]")
        if result.reasons:
            st.write("Reasons:")
            for reason in result.reasons:
                st.write(f"- {reason}")

        st.subheader("4. Drafted reply")
        if result.draft_reply:
            st.text_area("Suggested reply", result.draft_reply, height=150)
        else:
            st.write("_No draft — item was not auto-routed._")


def page_review_queue() -> None:
    st.title("Review queue")
    st.caption("Every item routed to human_review, with the specific failed checks that put it here.")

    conn = get_conn()
    rows = storage.list_items(conn, route=Route.human_review)

    if not rows:
        st.info("Nothing in the review queue right now.")
        return

    for row in rows:
        result = storage.row_to_triage_result(row)
        with st.expander(f"{row['id']} — {result.extraction.intent.value}  (status: {row['review_status']})"):
            st.write("**Raw text**")
            st.text(result.raw_text[:1000])

            st.write("**Reasons flagged for review**")
            for reason in result.reasons:
                st.write(f"- {reason}")

            fields = None
            if result.extraction.lead is not None:
                fields = result.extraction.lead.model_dump(mode="json")
            elif result.extraction.application is not None:
                fields = result.extraction.application.model_dump(mode="json")

            corrected = {}
            if fields:
                st.write("**Fields (editable)**")
                for name, value in fields.items():
                    if name == "missing_required_fields":
                        continue
                    corrected[name] = st.text_input(
                        f"{row['id']}::{name}", value="" if value is None else str(value)
                    )

            col1, col2 = st.columns(2)
            if col1.button("Approve", key=f"approve-{row['id']}"):
                storage.update_review(conn, row["id"], "approved")
                st.rerun()
            if col2.button("Save correction", key=f"correct-{row['id']}"):
                storage.update_review(conn, row["id"], "corrected", corrected_fields=corrected)
                st.rerun()


def page_metrics() -> None:
    st.title("Metrics")
    conn = get_conn()
    metrics = storage.get_metrics(conn)

    if metrics is None:
        st.info("No benchmark run yet. Run `python benchmark.py` to populate this page.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Field extraction accuracy", f"{metrics['field_accuracy']:.1%}")
    col2.metric("Intent (classify) accuracy", f"{metrics['intent_accuracy']:.1%}")
    col3.metric("Auto vs. human review", f"{metrics['pct_auto']:.0%} / {metrics['pct_human_review']:.0%}")
    col4.metric("Avg time / item", f"{metrics['avg_processing_ms']:.0f} ms")

    st.divider()
    col5, col6 = st.columns(2)
    col5.metric("Hours saved", f"{metrics['hours_saved']:.2f}")
    col6.metric("$ saved", f"${metrics['dollars_saved']:.2f}")

    st.divider()
    st.subheader("Per-field accuracy")
    st.table(
        {"field": list(metrics["per_field_accuracy"].keys()),
         "accuracy": [f"{v:.1%}" for v in metrics["per_field_accuracy"].values()]}
    )

    st.subheader("Recent items")
    rows = conn.execute(
        "SELECT id, intent, route, review_status, processing_ms FROM items ORDER BY received_at DESC LIMIT 25"
    ).fetchall()
    st.table([dict(r) for r in rows])


PAGES = [
    st.Page(page_live_demo, title="Live demo", icon="\U0001F4E5"),
    st.Page(page_review_queue, title="Review queue", icon="\U0001F4CB"),
    st.Page(page_metrics, title="Metrics", icon="\U0001F4CA"),
]

st.navigation(PAGES).run()
