"""Tests for tool wrappers: web (Tavily), arXiv, ReportGenerator."""
from __future__ import annotations

import pytest
from types import SimpleNamespace

from tools.arxiv_search import ArxivSearchTool, _parse_atom
from tools.report_generator import ReportGenerator, ReportInput, ReportSection, ReportSource
from tools.web_search import WebSearchTool, WebSearchResult, TAVILY_URL


# ── Web search (Tavily) ────────────────────────────────────────────────

class FakeHttpPost:
    def __init__(self, payload):
        self.payload = payload
        self.last_url = None
        self.last_body = None

    async def post(self, url, *, json):
        self.last_url = url
        self.last_body = json
        return SimpleNamespace(json=lambda: self.payload)


@pytest.mark.asyncio
async def test_web_search_posts_to_tavily_with_api_key():
    http = FakeHttpPost({
        "results": [
            {"url": "https://example.com/a", "title": "A", "content": "about a", "score": 0.9},
            {"url": "https://example.com/b", "title": "B", "content": "about b", "score": 0.6},
        ]
    })
    tool = WebSearchTool(api_key="SECRET", http=http)
    out = await tool.search("what is RRF?", k=2)
    assert len(out) == 2
    assert out[0].title == "A"
    assert out[0].relevance_score == pytest.approx(0.9)
    assert http.last_url == TAVILY_URL
    assert http.last_body["api_key"] == "SECRET"
    assert http.last_body["query"] == "what is RRF?"
    assert http.last_body["max_results"] == 2


@pytest.mark.asyncio
async def test_web_search_clamps_relevance_score_into_unit_interval():
    http = FakeHttpPost({"results": [
        {"url": "x", "title": "x", "content": "", "score": 1.7},   # over 1
        {"url": "y", "title": "y", "content": "", "score": -0.4},  # under 0
    ]})
    tool = WebSearchTool(api_key="k", http=http)
    out = await tool.search("q", k=2)
    assert out[0].relevance_score == 1.0
    assert out[1].relevance_score == 0.0


@pytest.mark.asyncio
async def test_web_search_empty_results_returns_empty():
    http = FakeHttpPost({"results": []})
    tool = WebSearchTool(api_key="k", http=http)
    out = await tool.search("q")
    assert out == []


# ── arXiv search ───────────────────────────────────────────────────────

ATOM_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2004.04906v1</id>
    <title>Dense Passage Retrieval for Open-Domain QA</title>
    <summary>We study dense passage retrieval for open-domain QA.</summary>
    <author><name>Vladimir Karpukhin</name></author>
    <author><name>Barlas Oguz</name></author>
    <link href="http://arxiv.org/pdf/2004.04906v1.pdf" type="application/pdf"/>
    <published>2020-04-10T18:00:00</published>
  </entry>
</feed>"""


def test_parse_atom_extracts_paper_fields():
    papers = _parse_atom(ATOM_FIXTURE)
    assert len(papers) == 1
    p = papers[0]
    assert p.title.startswith("Dense Passage Retrieval")
    assert "Vladimir Karpukhin" in p.authors
    assert "Barlas Oguz" in p.authors
    assert p.arxiv_id == "2004.04906v1"
    assert p.pdf_url.endswith(".pdf")
    assert p.published.year == 2020


class FakeHttpGet:
    def __init__(self, text):
        self._text = text
        self.last_params = None

    async def get(self, url, *, params=None):
        self.last_params = params
        return SimpleNamespace(text=self._text)


@pytest.mark.asyncio
async def test_arxiv_search_calls_api_and_parses_response():
    http = FakeHttpGet(ATOM_FIXTURE)
    tool = ArxivSearchTool(http=http)
    out = await tool.search("dense retrieval", max_results=5)
    assert len(out) == 1
    assert http.last_params["search_query"] == "all:dense retrieval"
    assert http.last_params["max_results"] == 5


# ── Report generator ───────────────────────────────────────────────────

def test_report_generator_creates_output_directory(tmp_path):
    gen = ReportGenerator(output_dir=tmp_path / "reports")
    assert (tmp_path / "reports").exists()


def test_report_generator_writes_file_with_session_id_in_name(tmp_path):
    gen = ReportGenerator(output_dir=tmp_path / "reports")
    payload = ReportInput(
        title="Test report",
        executive_summary="Summary",
        methodology="3 steps",
        sections=[ReportSection(heading="Findings", body="content")],
        sources=[ReportSource(title="src", url="http://x", credibility=0.8)],
        confidence_assessment="High",
    )
    path = gen.generate(payload, session_id="sess-001")
    assert path.exists()
    assert "sess-001" in path.name


def test_report_generator_sanitizes_unsafe_session_id_characters(tmp_path):
    gen = ReportGenerator(output_dir=tmp_path / "reports")
    payload = ReportInput(
        title="t", executive_summary="s", methodology="m",
    )
    path = gen.generate(payload, session_id="../malicious/id\\with/slashes")
    # Slashes/dots should be replaced with underscores
    assert "/" not in path.name
    assert "\\" not in path.name
    assert path.parent == (tmp_path / "reports")
