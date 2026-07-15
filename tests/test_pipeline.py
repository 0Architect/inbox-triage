from datetime import date, timedelta

import pipeline
import storage
from ingest import IngestedItem
from schema import Extraction, Intent, LeadFields, Route


def test_spam_short_circuits(monkeypatch):
    monkeypatch.setattr(pipeline, "classify", lambda raw_text, **kw: Intent.spam)
    item = IngestedItem("item-1", "buy your house fast for cash", {"channel": "email"})
    result = pipeline.process(item)
    assert result.route == Route.discarded
    assert result.draft_reply is None
    assert result.extraction.intent == Intent.spam


def test_auto_route_gets_a_draft(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "classify", lambda raw_text, **kw: Intent.new_lead)
    ext = Extraction(
        intent=Intent.new_lead,
        lead=LeadFields(
            contact_name="Jane",
            contact_email="jane@example.com",
            property_ref="123 Main St",
            desired_move_in=date.today() + timedelta(days=10),
        ),
    )
    monkeypatch.setattr(pipeline, "extract", lambda raw_text, intent, **kw: ext)
    monkeypatch.setattr(pipeline, "draft", lambda extraction, **kw: "Thanks for reaching out!")

    item = IngestedItem(
        "item-2",
        "Hi, I'm interested in 123 Main St. Email: jane@example.com",
        {"channel": "email"},
    )
    conn = storage.get_connection(tmp_path / "test.db")
    result = pipeline.process(item, conn=conn)

    assert result.route == Route.auto
    assert result.draft_reply == "Thanks for reaching out!"

    fetched = storage.get_item(conn, "item-2")
    assert fetched.route == Route.auto
    assert fetched.draft_reply == "Thanks for reaching out!"
    assert fetched.extraction.lead.contact_email == "jane@example.com"


def test_human_review_route_skips_draft(monkeypatch):
    monkeypatch.setattr(pipeline, "classify", lambda raw_text, **kw: Intent.new_lead)
    ext = Extraction(intent=Intent.new_lead, lead=LeadFields(contact_name="Jane"))
    monkeypatch.setattr(pipeline, "extract", lambda raw_text, intent, **kw: ext)

    def fail_if_called(*a, **kw):
        raise AssertionError("draft() should not be called for human_review items")

    monkeypatch.setattr(pipeline, "draft", fail_if_called)

    item = IngestedItem("item-3", "Hi, my name is Jane.", {"channel": "email"})
    result = pipeline.process(item)

    assert result.route == Route.human_review
    assert result.draft_reply is None
