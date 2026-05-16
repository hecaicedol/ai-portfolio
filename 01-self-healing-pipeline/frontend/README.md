# Frontend — Self-Healing Pipeline

Next.js 14 (App Router) + Tailwind CSS + shadcn/ui.

## What lives here

- `app/page.tsx` — Upload a document, watch the agents run in real time (SSE), see the final extracted data with critic scores.
- `app/memory/page.tsx` — Episodic memory viewer: most recent errors, search by similarity.
- `components/PipelineLane.tsx` — Visualization of the LangGraph state machine while the run is in progress.
- `components/PrincipleMeter.tsx` — Critic principle scores as 4 small gauges (completeness / accuracy / consistency / format).

## Initial setup (when you scaffold the app)

```bash
npx create-next-app@14.2.15 . --typescript --tailwind --app --eslint --src-dir=false --import-alias="@/*"
npx shadcn@latest init
npx shadcn@latest add button card input progress badge tabs
```

Add to `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running

```bash
npm install
npm run dev   # → http://localhost:3000
```

When the full stack is up via `docker compose up`, this frontend talks to the FastAPI backend at `http://localhost:8000` and consumes `/api/process/stream` via Server-Sent Events.
