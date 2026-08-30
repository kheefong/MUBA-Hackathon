"""
conftest.py — Shared pytest fixtures.

The full pipeline calls external LLM and search APIs. To keep the test
suite runnable offline / without real API keys, we monkeypatch
`llm_client.call_model_json` and `evidence_retriever._search` with
deterministic canned responses keyed by claim scenario.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure dummy API keys exist so MissingAPIKeyError isn't raised before our
# monkeypatched call_model_json ever runs.
os.environ.setdefault("DEEPSEEK_API_KEY", "test-key")
os.environ.setdefault("KIMI_API_KEY", "test-key")
os.environ.setdefault("MINIMAX_API_KEY", "test-key")
import tempfile

_tmp_db_path = os.path.join(tempfile.gettempdir(), "truth_engine_test.db")
if os.path.exists(_tmp_db_path):
    os.remove(_tmp_db_path)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_db_path}")

import pytest
