from typing import Any


class MCPClient:
    """
    Thin async client that talks to a local MCP server (stdio transport).

    Methods:
      - list_tools()  -> list[ToolSpec]   # for the planner's tool catalogue
      - call(tool_name, params) -> result
      - close()

    Each enterprise integration (github_server, jira_server, slack_server, gdrive_server)
    is launched as a subprocess MCP server and the executor holds one MCPClient per tool.
    """

    def __init__(self, *, server_command: list[str], env: dict[str, str] | None = None) -> None:
        self.server_command = server_command
        self.env = env or {}
        self._session = None

    async def connect(self) -> None:
        raise NotImplementedError("Use `mcp.ClientSession` over stdio with self.server_command")

    async def list_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def call(self, tool_name: str, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        raise NotImplementedError
