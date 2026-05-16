from datetime import datetime
from pydantic import BaseModel


class ArxivPaper(BaseModel):
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    published: datetime
    arxiv_id: str


class ArxivSearchTool:
    """Wrapper over the `arxiv` PyPI package, returning typed Pydantic objects."""

    async def search(self, query: str, *, max_results: int = 10) -> list[ArxivPaper]:
        raise NotImplementedError
