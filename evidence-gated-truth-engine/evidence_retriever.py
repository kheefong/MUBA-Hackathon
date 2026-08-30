"""
evidence_retriever.py — Step 5: search for evidence per proposition, score
source credibility, and run entailment between evidence and proposition.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from config import (
    QUERIES_PER_PROPOSITION,
    SEARCH_API_KEY,
    SEARCH_PROVIDER,
    SEARCH_RESULTS_PER_QUERY,
    SOURCE_CREDIBILITY,
    UTILITY_MODEL,
)
from llm_client import LLMCallError, call_model_json
from proposition_extractor import Proposition

logger = logging.getLogger("truth_engine.evidence_retriever")

# Lightweight, extensible domain -> credibility-tier heuristics. Real
# deployments should replace/extend this with a maintained source registry.
GOV_TLDS = (".gov", ".gov.my", ".gov.uk", ".europa.eu")
OFFICIAL_STATS_HINTS = ("who.int", "worldbank.org", "imf.org", "un.org", "oecd.org")
NATIONAL_NEWS_HINTS = (
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "wsj.com",
    "bloomberg.com", "channelnewsasia.com", "thestar.com.my", "nst.com.my",
)
THINKTANK_HINTS = ("brookings.edu", "rand.org", "chathamhouse.org")
BLOG_HINTS = ("medium.com", "blogspot.com", "substack.com", "reddit.com", "twitter.com", "x.com")


def classify_source_credibility(url: str) -> tuple[str, float]:
    domain = urlparse(url).netloc.lower()
    if any(domain.endswith(tld) for tld in GOV_TLDS):
        return "official_government", SOURCE_CREDIBILITY["official_government"]
    if any(hint in domain for hint in OFFICIAL_STATS_HINTS):
        return "peer_reviewed_or_official_stats", SOURCE_CREDIBILITY["peer_reviewed_or_official_stats"]
    if any(hint in domain for hint in NATIONAL_NEWS_HINTS):
        return "established_national_news", SOURCE_CREDIBILITY["established_national_news"]
    if any(hint in domain for hint in THINKTANK_HINTS):
        return "regional_news_or_think_tank", SOURCE_CREDIBILITY["regional_news_or_think_tank"]
    if any(hint in domain for hint in BLOG_HINTS):
        return "blog_opinion_social_media", SOURCE_CREDIBILITY["blog_opinion_social_media"]
    return "unsourced_unknown", SOURCE_CREDIBILITY["unsourced_unknown"]


@dataclass
class EvidenceSnippet:
    title: str
    url: str
    text: str
    credibility_tier: str
    credibility: float
    relation: str = "NOT_COVERED"  # SUPPORT | CONTRADICT | NOT_COVERED
    explanation: str = ""


@dataclass
class PropositionEvidence:
    proposition: Proposition
    snippets: list[EvidenceSnippet] = field(default_factory=list)

    @property
    def support_score(self) -> float:
        supports = [s.credibility for s in self.snippets if s.relation == "SUPPORT"]
        return max(supports) if supports else 0.0

    @property
    def contradiction_score(self) -> float:
        contradicts = [s.credibility for s in self.snippets if s.relation == "CONTRADICT"]
        return max(contradicts) if contradicts else 0.0

    @property
    def coverage(self) -> float:
        covered = any(s.relation != "NOT_COVERED" and s.credibility > 0.2 for s in self.snippets)
        return 1.0 if covered else 0.0


# ---------------------------------------------------------------------------
# Search backends
# ---------------------------------------------------------------------------
async def _search_tavily(client: httpx.AsyncClient, query: str) -> list[dict]:
    resp = await client.post(
        "https://api.tavily.com/search",
        json={"api_key": SEARCH_API_KEY, "query": query, "max_results": SEARCH_RESULTS_PER_QUERY},
    )
    resp.raise_for_status()
    data = resp.json()
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "text": r.get("content", "")} for r in data.get("results", [])]


async def _search_serper(client: httpx.AsyncClient, query: str) -> list[dict]:
    resp = await client.post(
        "https://google.serper.dev/search",
        json={"q": query, "num": SEARCH_RESULTS_PER_QUERY},
        headers={"X-API-KEY": SEARCH_API_KEY, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {"title": r.get("title", ""), "url": r.get("link", ""), "text": r.get("snippet", "")}
        for r in data.get("organic", [])
    ]


async def _search_perplexity(client: httpx.AsyncClient, query: str) -> list[dict]:
    resp = await client.post(
        "https://api.perplexity.ai/chat/completions",
        json={
            "model": "sonar",
            "messages": [{"role": "user", "content": query}],
        },
        headers={"Authorization": f"Bearer {SEARCH_API_KEY}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    citations = data.get("citations", [])
    content = data["choices"][0]["message"]["content"] if data.get("choices") else ""
    return [{"title": url, "url": url, "text": content} for url in citations[:SEARCH_RESULTS_PER_QUERY]]


SEARCH_BACKENDS = {"tavily": _search_tavily, "serper": _search_serper, "perplexity": _search_perplexity}


async def _search(client: httpx.AsyncClient, query: str) -> list[dict]:
    if not SEARCH_API_KEY:
        return []
    backend = SEARCH_BACKENDS.get(SEARCH_PROVIDER)
    if backend is None:
        logger.error("Unknown SEARCH_PROVIDER=%s", SEARCH_PROVIDER)
        return []
    try:
        return await backend(client, query)
    except httpx.HTTPError as e:
        logger.warning("Search query %r failed: %s", query, e)
        return []


def _generate_queries(prop: Proposition) -> list[str]:
    base = prop.proposition
    queries = [base]
    if prop.subject and prop.object:
        queries.append(f"{prop.subject} {prop.object}")
    if prop.subject and prop.time:
        queries.append(f"{prop.subject} {prop.time}")
    return queries[:QUERIES_PER_PROPOSITION] or [base]


ENTAILMENT_SYSTEM_PROMPT = "You judge whether evidence supports, contradicts, or does not cover a proposition. Return JSON only."
ENTAILMENT_USER_PROMPT = """\
Given the evidence and the proposition, does the evidence SUPPORT, CONTRADICT, or NOT_COVER the proposition?

Proposition: {prop}
Evidence: {evidence_text}

Return JSON:
{{
  "relation": "SUPPORT" | "CONTRADICT" | "NOT_COVERED",
  "explanation": "..."
}}
"""


async def _judge_relation(prop_text: str, evidence_text: str) -> tuple[str, str]:
    try:
        raw = await call_model_json(
            UTILITY_MODEL,
            ENTAILMENT_SYSTEM_PROMPT,
            ENTAILMENT_USER_PROMPT.format(prop=prop_text, evidence_text=evidence_text[:2000]),
            max_tokens=200,
        )
    except LLMCallError as e:
        logger.warning("Evidence entailment call failed: %s", e)
        return "NOT_COVERED", str(e)
    if not isinstance(raw, dict):
        return "NOT_COVERED", "Malformed output."
    relation = str(raw.get("relation", "NOT_COVERED")).upper()
    if relation not in ("SUPPORT", "CONTRADICT", "NOT_COVERED"):
        relation = "NOT_COVERED"
    return relation, str(raw.get("explanation", ""))


async def retrieve_evidence_for_proposition(
    client: httpx.AsyncClient, prop: Proposition
) -> PropositionEvidence:
    queries = _generate_queries(prop)
    result_lists = await asyncio.gather(*(_search(client, q) for q in queries))
    raw_results = [r for group in result_lists for r in group]

    # De-duplicate by URL.
    seen: set[str] = set()
    deduped = []
    for r in raw_results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            deduped.append(r)

    snippets: list[EvidenceSnippet] = []
    for r in deduped:
        tier, credibility = classify_source_credibility(r["url"])
        snippets.append(
            EvidenceSnippet(
                title=r.get("title", ""),
                url=r["url"],
                text=r.get("text", ""),
                credibility_tier=tier,
                credibility=credibility,
            )
        )

    async def _judge(snippet: EvidenceSnippet) -> None:
        relation, explanation = await _judge_relation(prop.proposition, snippet.text)
        snippet.relation = relation
        snippet.explanation = explanation

    await asyncio.gather(*(_judge(s) for s in snippets))
    return PropositionEvidence(proposition=prop, snippets=snippets)


async def retrieve_all_evidence(propositions: list[Proposition]) -> list[PropositionEvidence]:
    if not propositions:
        return []
    async with httpx.AsyncClient(timeout=30) as client:
        results = await asyncio.gather(
            *(retrieve_evidence_for_proposition(client, p) for p in propositions)
        )
    return list(results)
