"""Ingest raw inbound (email text and/or PDF attachment) into raw_text + source_meta.

PDFs are read natively by passing them straight to a vision-capable model — no
dedicated OCR subsystem unless native reading measurably underperforms on bad
scans (SPEC §1, §2 step 1).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import anthropic

import config


@dataclass
class IngestedItem:
    source_id: str
    raw_text: str
    source_meta: dict


def ingest_email(source_id: str, email_body: str, *, channel: str = "email") -> IngestedItem:
    return IngestedItem(
        source_id=source_id,
        raw_text=email_body,
        source_meta={"channel": channel, "filename": None},
    )


def _transcribe_pdf(pdf_path: str | Path, client: anthropic.Anthropic) -> str:
    pdf_bytes = Path(pdf_path).read_bytes()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    response = client.messages.create(
        model=config.MODEL_EXTRACT,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transcribe all text content from this document verbatim, "
                            "including form field labels and their filled-in values. "
                            "Preserve layout cues (e.g. which value belongs to which "
                            "label) as plain text. Do not summarize or omit anything."
                        ),
                    },
                ],
            }
        ],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def ingest_pdf(
    source_id: str,
    pdf_path: str | Path,
    *,
    channel: str = "email_attachment",
    client: anthropic.Anthropic | None = None,
) -> IngestedItem:
    client = client or anthropic.Anthropic()
    raw_text = _transcribe_pdf(pdf_path, client)
    return IngestedItem(
        source_id=source_id,
        raw_text=raw_text,
        source_meta={"channel": channel, "filename": Path(pdf_path).name},
    )


def ingest_email_with_attachment(
    source_id: str,
    email_body: str,
    pdf_path: str | Path | None = None,
    *,
    channel: str = "email",
    client: anthropic.Anthropic | None = None,
) -> IngestedItem:
    """Combine an email body with an optional PDF attachment's transcribed text."""
    if pdf_path is None:
        return ingest_email(source_id, email_body, channel=channel)

    client = client or anthropic.Anthropic()
    filename = Path(pdf_path).name
    pdf_text = _transcribe_pdf(pdf_path, client)
    raw_text = f"{email_body}\n\n--- Attachment: {filename} ---\n{pdf_text}"
    return IngestedItem(
        source_id=source_id,
        raw_text=raw_text,
        source_meta={"channel": channel, "filename": filename},
    )
