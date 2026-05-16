"""
MCP server exposing Slack actions.

Exposed tools:
  - send_message(channel, message)
  - get_channel_history(channel, limit=10)
  - create_thread_reply(channel, thread_ts, message)

Uses slack-sdk WebClient with SLACK_BOT_TOKEN from env.
"""
