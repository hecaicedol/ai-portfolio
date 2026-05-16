# Frontend — Research Agent

Next.js 14 + Tailwind + shadcn/ui.

## Three views

1. **Research** (`app/page.tsx`) — Submit a question, watch the LangGraph plan and execution stream in real time (SSE). Plan steps update live with status.
2. **Memory viewer** (`app/memory/page.tsx`) — Three panels: current working memory snapshot, episodic timeline of past sessions, searchable semantic knowledge base.
3. **Reports** (`app/reports/page.tsx`) — Library of generated PDF reports per session, downloadable.

## Setup

```bash
npx create-next-app@14.2.15 . --typescript --tailwind --app --import-alias="@/*"
npx shadcn@latest init && npx shadcn@latest add card tabs progress badge button input
```

`.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
