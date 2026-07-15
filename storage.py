"""SQLite persistence — SPEC §7. One file, zero infra. `items` holds every
processed item (the review queue is just `route = 'human_review'` rows);
`gold` holds ground-truth labels for the benchmark.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config
from schema import Extraction, Route, TriageResult, ValidationResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    received_at TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    source_meta TEXT NOT NULL,
    intent TEXT NOT NULL,
    extraction TEXT NOT NULL,
    validation TEXT NOT NULL,
    route TEXT NOT NULL,
    reasons TEXT NOT NULL,
    draft_reply TEXT,
    review_status TEXT NOT NULL DEFAULT 'pending',
    corrected_fields TEXT,
    processing_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS gold (
    source_id TEXT PRIMARY KEY,
    ground_truth TEXT NOT NULL,
    adversarial INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    computed_at TEXT NOT NULL,
    n_items INTEGER NOT NULL,
    field_accuracy REAL NOT NULL,
    per_field_accuracy TEXT NOT NULL,
    intent_accuracy REAL NOT NULL,
    adversarial_route_accuracy REAL NOT NULL,
    pct_auto REAL NOT NULL,
    pct_human_review REAL NOT NULL,
    pct_discarded REAL NOT NULL,
    avg_processing_ms REAL NOT NULL,
    manual_baseline_minutes REAL NOT NULL,
    hours_saved REAL NOT NULL,
    dollars_saved REAL NOT NULL
);
"""


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def save_triage_result(
    conn: sqlite3.Connection,
    result: TriageResult,
    source_meta: dict,
    processing_ms: int,
) -> None:
    conn.execute(
        """
        INSERT INTO items (
            id, received_at, raw_text, source_meta, intent, extraction,
            validation, route, reasons, draft_reply, review_status,
            corrected_fields, processing_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?)
        ON CONFLICT(id) DO UPDATE SET
            received_at=excluded.received_at, raw_text=excluded.raw_text,
            source_meta=excluded.source_meta, intent=excluded.intent,
            extraction=excluded.extraction, validation=excluded.validation,
            route=excluded.route, reasons=excluded.reasons,
            draft_reply=excluded.draft_reply, processing_ms=excluded.processing_ms
        """,
        (
            result.source_id,
            datetime.now(timezone.utc).isoformat(),
            result.raw_text,
            json.dumps(source_meta),
            result.extraction.intent.value,
            result.extraction.model_dump_json(),
            result.validation.model_dump_json(),
            result.route.value,
            json.dumps(result.reasons),
            result.draft_reply,
            processing_ms,
        ),
    )
    conn.commit()


def row_to_triage_result(row: sqlite3.Row) -> TriageResult:
    return TriageResult(
        source_id=row["id"],
        raw_text=row["raw_text"],
        extraction=Extraction.model_validate_json(row["extraction"]),
        validation=ValidationResult.model_validate_json(row["validation"]),
        route=Route(row["route"]),
        draft_reply=row["draft_reply"],
        reasons=json.loads(row["reasons"]),
    )


def get_item(conn: sqlite3.Connection, source_id: str) -> TriageResult | None:
    row = conn.execute("SELECT * FROM items WHERE id = ?", (source_id,)).fetchone()
    return row_to_triage_result(row) if row else None


def list_items(
    conn: sqlite3.Connection,
    *,
    route: Route | None = None,
    review_status: str | None = None,
) -> list[sqlite3.Row]:
    query = "SELECT * FROM items WHERE 1=1"
    params: list[str] = []
    if route is not None:
        query += " AND route = ?"
        params.append(route.value)
    if review_status is not None:
        query += " AND review_status = ?"
        params.append(review_status)
    query += " ORDER BY received_at DESC"
    return conn.execute(query, params).fetchall()


def update_review(
    conn: sqlite3.Connection,
    source_id: str,
    review_status: str,
    corrected_fields: dict | None = None,
) -> None:
    conn.execute(
        "UPDATE items SET review_status = ?, corrected_fields = ? WHERE id = ?",
        (review_status, json.dumps(corrected_fields) if corrected_fields else None, source_id),
    )
    conn.commit()


def save_gold(
    conn: sqlite3.Connection,
    source_id: str,
    ground_truth: dict,
    *,
    adversarial: bool = False,
) -> None:
    conn.execute(
        """
        INSERT INTO gold (source_id, ground_truth, adversarial) VALUES (?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            ground_truth=excluded.ground_truth, adversarial=excluded.adversarial
        """,
        (source_id, json.dumps(ground_truth), int(adversarial)),
    )
    conn.commit()


def list_gold(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM gold").fetchall()


def save_metrics(conn: sqlite3.Connection, metrics: dict) -> None:
    """Single-row table (id=1) — always the most recent benchmark run."""
    conn.execute(
        """
        INSERT INTO metrics (
            id, computed_at, n_items, field_accuracy, per_field_accuracy,
            intent_accuracy, adversarial_route_accuracy, pct_auto,
            pct_human_review, pct_discarded, avg_processing_ms,
            manual_baseline_minutes, hours_saved, dollars_saved
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            computed_at=excluded.computed_at, n_items=excluded.n_items,
            field_accuracy=excluded.field_accuracy,
            per_field_accuracy=excluded.per_field_accuracy,
            intent_accuracy=excluded.intent_accuracy,
            adversarial_route_accuracy=excluded.adversarial_route_accuracy,
            pct_auto=excluded.pct_auto, pct_human_review=excluded.pct_human_review,
            pct_discarded=excluded.pct_discarded,
            avg_processing_ms=excluded.avg_processing_ms,
            manual_baseline_minutes=excluded.manual_baseline_minutes,
            hours_saved=excluded.hours_saved, dollars_saved=excluded.dollars_saved
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            metrics["n_items"],
            metrics["field_accuracy"],
            json.dumps(metrics["per_field_accuracy"]),
            metrics["intent_accuracy"],
            metrics["adversarial_route_accuracy"],
            metrics["pct_auto"],
            metrics["pct_human_review"],
            metrics["pct_discarded"],
            metrics["avg_processing_ms"],
            metrics["manual_baseline_minutes"],
            metrics["hours_saved"],
            metrics["dollars_saved"],
        ),
    )
    conn.commit()


def get_metrics(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute("SELECT * FROM metrics WHERE id = 1").fetchone()
    if row is None:
        return None
    result = dict(row)
    result["per_field_accuracy"] = json.loads(result["per_field_accuracy"])
    return result
