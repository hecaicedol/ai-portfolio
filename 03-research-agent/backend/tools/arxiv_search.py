"""arXiv search tool. Queries the arXiv ATOM API directly via httpx so
the suite doesn't need the `arxiv` PyPI package. The XML parser is the
stdlib `xml.etree`. Production code can swap in the arxiv lib for
richer results if needed."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol
from xml.etree import ElementTree as ET

from pydantic import BaseModel


ARXIV_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivPaper(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    published: datetime
    arxiv_id: str


class _HttpClient(Protocol):
    async def get(self, url: str, *, params: dict | None = None) -> Any: ...


class ArxivSearchTool:
    def __init__(self, *, http: _HttpClient | None = None) -> None:
        self._http = http

    def _client(self) -> _HttpClient:
        if self._http is None:
            import httpx  # lazy
            self._http = httpx.AsyncClient(timeout=20.0)
        return self._http

    async def search(self, query: str, *, max_results: int = 10) -> list[ArxivPaper]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        response = await self._client().get(ARXIV_URL, params=params)
        xml = response.text if hasattr(response, "text") else str(response)
        return _parse_atom(xml)


def _parse_atom(xml_text: str) -> list[ArxivPaper]:
    root = ET.fromstring(xml_text)
    papers: list[ArxivPaper] = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title_el = entry.find(f"{ATOM_NS}title")
        summary_el = entry.find(f"{ATOM_NS}summary")
        id_el = entry.find(f"{ATOM_NS}id")
        published_el = entry.find(f"{ATOM_NS}published")
        if any(el is None for el in (title_el, summary_el, id_el, published_el)):
            continue
        title = (title_el.text or "").strip()
        abstract = (summary_el.text or "").strip()
        arxiv_id = (id_el.text or "").rsplit("/", 1)[-1]
        published_raw = (published_el.text or "").rstrip("Z")
        try:
            published = datetime.fromisoformat(published_raw)
        except ValueError:
            continue

        authors: list[str] = []
        for author_el in entry.findall(f"{ATOM_NS}author"):
            name_el = author_el.find(f"{ATOM_NS}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append(ArxivPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            pdf_url=pdf_url,
            published=published,
            arxiv_id=arxiv_id,
        ))
    return papers
