"""
database.py — Persistence layer (SQLite by default, PostgreSQL via
DATABASE_URL) for verification logs and calibration history.

Uses SQLAlchemy Core (sync engine) wrapped with asyncio.to_thread so the
rest of the codebase can stay async without pulling in an async DB driver.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
)

from config import DATABASE_URL

metadata = MetaData()

# One row per (claim, model, proposition) verification event used for
# calibration: what confidence the model stated, and whether it was
# eventually judged correct against evidence.
calibration_log = Table(
    "calibration_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime, nullable=False),
    Column("model_name", String, nullable=False, index=True),
    Column("claim_id", String, nullable=False),
    Column("stated_confidence", Float, nullable=False),  # 0-100
    Column("correct", Integer, nullable=True),  # 1, 0, or NULL if unresolved
)

# Full verification result, stored for audit / debugging / re-scoring.
verification_log = Table(
    "verification_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("claim_id", String, nullable=False, unique=True),
    Column("timestamp", DateTime, nullable=False),
    Column("claim_text", Text, nullable=False),
    Column("result_json", Text, nullable=False),
)

_engine = create_engine(DATABASE_URL, future=True)
metadata.create_all(_engine)


def _sync_log_calibration(model_name: str, claim_id: str, stated_confidence: float, correct: int | None) -> None:
    with _engine.begin() as conn:
        conn.execute(
            insert(calibration_log).values(
                timestamp=datetime.now(timezone.utc),
                model_name=model_name,
                claim_id=claim_id,
                stated_confidence=stated_confidence,
                correct=correct,
            )
        )


def _sync_get_calibration_history(model_name: str, window: int) -> list[tuple[float, int]]:
    with _engine.begin() as conn:
        rows = conn.execute(
            select(calibration_log.c.stated_confidence, calibration_log.c.correct)
            .where(calibration_log.c.model_name == model_name)
            .where(calibration_log.c.correct.is_not(None))
            .order_by(calibration_log.c.id.desc())
            .limit(window)
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _sync_log_verification(claim_id: str, claim_text: str, result: dict) -> None:
    with _engine.begin() as conn:
        conn.execute(
            insert(verification_log).values(
                claim_id=claim_id,
                timestamp=datetime.now(timezone.utc),
                claim_text=claim_text,
                result_json=json.dumps(result, default=str),
            )
        )


async def log_calibration_sample(model_name: str, claim_id: str, stated_confidence: float, correct: int | None) -> None:
    await asyncio.to_thread(_sync_log_calibration, model_name, claim_id, stated_confidence, correct)


async def get_calibration_history(model_name: str, window: int) -> list[tuple[float, int]]:
    return await asyncio.to_thread(_sync_get_calibration_history, model_name, window)


async def log_verification_result(claim_id: str, claim_text: str, result: dict) -> None:
    await asyncio.to_thread(_sync_log_verification, claim_id, claim_text, result)
