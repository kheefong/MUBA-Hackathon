# Evidence-Gated Truth Engine

A multi-model fact-verification service. Three LLMs (**DeepSeek-V4-Flash**,
**Kimi-K2.6**, **MiniMax-M2.7**) independently verdict a claim; their
reasoning is decomposed into atomic propositions, cross-checked against
each other (NLI) and against retrieved web evidence (source-credibility
weighted), and combined into a **Truth Score** that is *gated* by evidence
alignment — models agreeing with each other is not enough to produce a
high Truth Score if the evidence contradicts them.

## How scoring works

| Score | Meaning |
|---|---|
| **Weighted Verdict Agreement** | Chance-corrected, trust-weighted kappa across the three models' TRUE/FALSE/NOT_ENOUGH_INFO verdicts. |
| **Reasoning Consistency** | Trust-weighted mean NLI score across propositions shared by different models (entailment=1.0, neutral=0.5, contradiction=0.0). |
| **Evidence Coverage** | Fraction of extracted propositions with at least one non-trivial evidence snippet. If < 50%, the claim is marked `UNVERIFIABLE` and gets no Truth Score. |
| **Evidence Alignment** | Mean support over *covered* propositions — but any proposition with a credible (credibility > 0.2) contradicting source is forced to 0, regardless of how much support it also has. |
| **Truth Score** | `Evidence Alignment × (0.4 + 0.3×Reasoning Consistency + 0.3×Weighted Verdict Agreement)`. Zero evidence alignment ⇒ zero Truth Score, no matter how much the models agree. |
| **Consensus Score** | `0.5×Weighted Verdict Agreement + 0.5×Reasoning Consistency` — purely about model agreement, independent of evidence. |
| **Confidence Score** | Trust-weighted sum of each model's *calibrated* confidence (calibrated against that model's real track record, not its self-reported number). `null` when coverage < 50%. |

Trust weights are each model's rolling accuracy (over its last 200
evidence-verified propositions), normalized to sum to 1. They start equal
(1/3 each) and adapt as the calibration log fills in.

## Pipeline

```
claim
  → claim_classifier.py      (factual/opinion/prediction/mixed/ambiguous)
  → model_interrogator.py    (parallel calls to the 3 models)
  → proposition_extractor.py (atomic, canonicalized propositions)
  → nli_engine.py             (model-vs-model contradiction matrix)
  → evidence_retriever.py    (web search + credibility + entailment)
  → calibration.py           (isotonic-regression calibrated confidence, trust weights)
  → scoring.py                (Truth / Consensus / Confidence scores)
  → pipeline.py               (assembles the final JSON, logs to database.py)
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then fill in DEEPSEEK_API_KEY, KIMI_API_KEY, MINIMAX_API_KEY,
# and optionally SEARCH_API_KEY (Tavily/Serper/Perplexity) for evidence retrieval.
```

If you don't set `SEARCH_API_KEY`, the engine still runs, but evidence
coverage will be 0 for every claim, so every claim resolves to
`UNVERIFIABLE` — the gate is intentionally strict. Evidence retrieval is
what makes this an *evidence-gated* engine rather than a pure
model-consensus poll.

### Optional: local NLI model instead of LLM-prompted NLI

By default, contradiction/entailment checks are done via a prompt to the
utility model (`NLI_BACKEND=llm`, zero extra dependencies). To use a
dedicated NLI model instead:

```bash
pip install transformers torch
```

```env
NLI_BACKEND=hf
HF_NLI_MODEL=microsoft/deberta-v3-large-mnli
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"claim": "Malaysia raised its minimum wage to RM1700 in 2026."}'
```

## Tests

The test suite monkeypatches all LLM and search calls with deterministic
fixtures, so it runs fully offline — no API keys required:

```bash
pytest tests/ -v
```

It covers the 5 required scenarios (models agree + evidence supports,
models agree + evidence contradicts, opinion, prediction, mixed claim)
plus unit tests for the core scoring formulas (kappa agreement,
evidence-gated alignment, coverage gating).

## Project layout

```
main.py                  FastAPI app, /verify endpoint
config.py                Model names, endpoints, env vars, credibility table
llm_client.py             Shared async HTTP client + JSON extraction/retries
claim_classifier.py       Step 1: is this claim checkable?
model_interrogator.py     Step 2: parallel multi-model verdicts
proposition_extractor.py  Step 3: atomic, canonicalized propositions
nli_engine.py              Step 4: model-vs-model contradiction matrix
evidence_retriever.py     Step 5: search, source credibility, entailment
calibration.py            Step 6: isotonic calibration + dynamic trust weights
scoring.py                 Step 7: Truth / Consensus / Confidence formulas
pipeline.py                Orchestrates all steps into the final JSON
database.py                SQLite/Postgres persistence for logs + calibration
tests/                     pytest suite (offline, mocked externals)
```

## Notes on the configured models

`DeepSeek-V4-Flash`, `Kimi-K2.6`, and `MiniMax-M2.7` are treated purely as
configuration — model name strings and OpenAI-compatible endpoints in
`config.py`. If a provider's actual API base URL, model ID, or request
shape differs from what's hardcoded here, update `config.py` and (if the
provider isn't OpenAI-compatible) add a branch in `llm_client.py`; nothing
else in the codebase needs to change.
