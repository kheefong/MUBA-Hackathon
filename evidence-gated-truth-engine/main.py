"""
main.py — FastAPI app exposing the Evidence-Gated Truth Engine.

Run with:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline import verify_claim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("truth_engine.main")

app = FastAPI(
    title="Evidence-Gated Truth Engine",
    description="Multi-model fact verification with evidence-gated Truth Scoring.",
    version="1.0.0",
)


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="The claim text to verify.")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/verify")
async def verify(request: VerifyRequest) -> dict:
    try:
        return await verify_claim(request.claim)
    except Exception as e:  # noqa: BLE001
        logger.exception("Verification failed for claim: %s", request.claim)
        raise HTTPException(status_code=500, detail=f"Verification failed: {e}") from e
