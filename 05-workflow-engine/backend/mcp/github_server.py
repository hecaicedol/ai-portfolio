"""
MCP server exposing GitHub actions to the workflow engine.

Run as a subprocess; the executor connects to it via stdio.

Exposed tools:
  - get_issue(repo, number)
  - create_issue(repo, title, body, labels)
  - create_pr(repo, title, body, base, head)
  - get_repo_summary(repo)  → recent_commits, open_issues_count

Implementation outline:
  1. mcp.server.Server('github')
  2. @server.list_tools — return JSON-schema for each action above
  3. @server.call_tool — dispatch by name, using PyGithub with GITHUB_TOKEN
  4. asyncio.run(server.run(stdio_transport))
"""
