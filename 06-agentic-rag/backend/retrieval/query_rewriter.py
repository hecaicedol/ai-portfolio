class QueryRewriter:
    """
    Claude generates `n` alternative phrasings of the query.
    Goal: surface chunks that match the question's intent even when the
    user's wording differs from the document's vocabulary.

    Prompt enforces JSON-array output.
    """

    SYSTEM_PROMPT = """You are a query-rewriting agent. Given a user question,
return exactly N alternative phrasings as a JSON array of strings.
Make the rewrites lexically diverse: vary domain vocabulary, level of formality,
and granularity. Do NOT answer the question — only rephrase it.

Output: ["...", "...", "..."]"""

    def __init__(self, *, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    async def rewrite(self, query: str, n: int = 3) -> list[str]:
        raise NotImplementedError
