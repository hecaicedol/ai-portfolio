from pydantic import BaseModel, Field


class WebSearchResult(BaseModel):
    url: str
    title: str
    content: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class WebSearchTool:
    """Thin wrapper over Tavily Search API."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(self, query: str, *, k: int = 5) -> list[WebSearchResult]:
        raise NotImplementedError("Call tavily-python AsyncTavilyClient.search()")
