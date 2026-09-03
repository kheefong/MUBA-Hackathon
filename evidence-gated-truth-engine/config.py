"""
config.py — Central configuration for the Evidence-Gated Truth Engine.

All model identifiers, API endpoints, and environment variable names live
here so the rest of the codebase never hardcodes them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    name: str
    env_key: str
    base_url: str
    model_id: str
    # Some providers use OpenAI-compatible chat completion endpoints; others
    # need bespoke request shapes. This flag lets model_interrogator.py pick
    # the right adapter.
    api_style: str = "openai_compatible"


# ---------------------------------------------------------------------------
# The three models this engine interrogates for every claim.
# Base URLs are the documented OpenAI-compatible endpoints for each
# provider at the time of writing. If a provider changes their endpoint,
# update it here — nothing else in the codebase needs to change.
# ---------------------------------------------------------------------------
DEEPSEEK = ModelConfig(
    name="DeepSeek-V4-Flash",
    env_key="DEEPSEEK_API_KEY",
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.gonkarouter.io/v1"),
    model_id=os.getenv("DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-V4-Flash-0731"),
)

KIMI = ModelConfig(
    name="Kimi-K2.6",
    env_key="KIMI_API_KEY",
    base_url=os.getenv("KIMI_BASE_URL", "https://api.gonkarouter.io/v1"),
    model_id=os.getenv("KIMI_MODEL_ID", "moonshotai/Kimi-K2.6"),
)

MINIMAX = ModelConfig(
    name="MiniMax-M2.7",
    env_key="MINIMAX_API_KEY",
    base_url=os.getenv("MINIMAX_BASE_URL", "https://api.gonkarouter.io/v1"),
    model_id=os.getenv("MINIMAX_MODEL_ID", "MiniMaxAI/MiniMax-M2.7"),
)

ALL_MODELS: list[ModelConfig] = [DEEPSEEK, KIMI, MINIMAX]

# Model used for claim classification, proposition extraction, and (if no
# dedicated NLI model is configured) entailment checks. Defaults to
# DeepSeek per the spec, but can be overridden.
UTILITY_MODEL: ModelConfig = next(
    (m for m in ALL_MODELS if m.name == os.getenv("UTILITY_MODEL_NAME", DEEPSEEK.name)),
    DEEPSEEK,
)

# ---------------------------------------------------------------------------
# Search / evidence retrieval
# ---------------------------------------------------------------------------
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily")  # tavily | serper | perplexity
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "10"))
QUERIES_PER_PROPOSITION = int(os.getenv("QUERIES_PER_PROPOSITION", "3"))

# ---------------------------------------------------------------------------
# NLI engine: "llm" (prompt-based, default, zero extra dependencies) or
# "hf" (HuggingFace DeBERTa-v3-large-mnli, requires `transformers`+`torch`)
# ---------------------------------------------------------------------------
NLI_BACKEND = os.getenv("NLI_BACKEND", "llm")
HF_NLI_MODEL = os.getenv("HF_NLI_MODEL", "microsoft/deberta-v3-large-mnli")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./truth_engine.db")

# ---------------------------------------------------------------------------
# Source credibility table (spec §5)
# ---------------------------------------------------------------------------
SOURCE_CREDIBILITY = {
    "official_government": 0.95,
    "peer_reviewed_or_official_stats": 0.90,
    "established_national_news": 0.75,
    "regional_news_or_think_tank": 0.60,
    "blog_opinion_social_media": 0.30,
    "unsourced_unknown": 0.10,
}

# Rolling-accuracy window for calibration (spec §6)
CALIBRATION_WINDOW = int(os.getenv("CALIBRATION_WINDOW", "200"))
CALIBRATION_MIN_SAMPLES_FOR_ISOTONIC = int(os.getenv("CALIBRATION_MIN_SAMPLES", "20"))

# LLM call tuning
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
