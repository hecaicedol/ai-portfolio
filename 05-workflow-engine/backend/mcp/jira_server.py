"""
MCP server exposing Jira actions.

Exposed tools:
  - create_ticket(project, summary, description, priority)
  - update_ticket_status(ticket_id, status)
  - get_sprint_status(project)

Uses atlassian-python-api with JIRA_HOST / JIRA_USER / JIRA_TOKEN from env.
"""
