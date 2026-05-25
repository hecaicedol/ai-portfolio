"""Tavily web-search tool wrapper.

HTTP client injected so tests never hit the network. Production wires
an `httpx.AsyncClient`; tests pass a fake whose `.post()` returns a
canned payload.
"""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


TAVILY_URL = "https://api.tavily.com/search"


class WebSearchResult(BaseModel):
    url: str
    title: str
    content: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class _HttpClient(Protocol):
    async def post(self, url: str, *, json: dict) -> Any: ...


class WebSearchTool:
    def __init__(
        self,
        api_key: str,
        *,
        http: _HttpClient | None = None,
    ) -> None:
        self.api_key = api_key
        self._http = http

    def _client(self) -> _HttpClient:
        if self._http is None:
            import httpx  # lazy
            self._http = httpx.AsyncClient(timeout=15.0)
        return self._http

    async def search(self, query: str, *, k: int = 5) -> list[WebSearchResult]:
        body = {
            "api_key": self.api_key,
            "query": query,
            "max_results": k,
            "search_depth": "advanced",
        }
        response = await self._client().post(TAVILY_URL, json=body)
        data = response.json() if callable(getattr(response, "json", None)) else response
        results = data.get("results", []) if isinstance(data, dict) else []
        out: list[WebSearchResult] = []
        for r in results[:k]:
            score = r.get("score")
            if score is None:
                score = r.get("relevance_score", 0.5)
            out.append(WebSearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                content=r.get("content", ""),
                relevance_score=float(max(0.0, min(1.0, score))),
            ))
        return out
