"""
MCP server exposing Google Drive actions.

Exposed tools:
  - search_files(query, mime_type=None)
  - read_doc(file_id)
  - create_doc(title, content, folder_id=None)
  - share_file(file_id, email, role)

Uses google-api-python-client with a service-account credentials JSON
(GDRIVE_CREDENTIALS_JSON env var, file path or inline JSON).
"""
