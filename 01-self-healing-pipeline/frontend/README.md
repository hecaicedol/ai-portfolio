# Self-Healing Pipeline — frontend

Next.js 14 (App Router) + Tailwind + TypeScript SPA that talks to the FastAPI
backend over SSE.

## What it shows

- **Input panel** — choose a document type, pick a built-in sample or paste
  your own text.
- **Pipeline panel** — live stream of every node the LangGraph executes
  (`extract → validate → critique → reflect → synthesize`), with iteration
  number when reflection kicks in.
- **Critic report** — per-principle progress bars (completeness, accuracy,
  consistency, format) with the per-principle feedback the critic produced.
  Chips at the bottom show which past similar errors the critic consulted
  before scoring.
- **Extracted data** — final JSON output plus a `self-healed` badge when
  the system needed reflection to reach a passing score.
- **Episodic memory** — recent errors stored in pgvector. Refreshes
  automatically after each run so you can watch the memory grow.

## Quick start

```bash
# from this directory
npm install
cp .env.example .env.local        # default points to http://localhost:8000
npm run dev                       # http://localhost:3000
```

The backend must be running locally (`docker compose up` from the project
root, or run uvicorn manually).

## Project layout

```
src/
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx               # orchestrates the SSE stream → component state
├── components/
│   ├── Header.tsx             # title + live API health badge
│   ├── DocumentInput.tsx      # textarea + sample picker + run button
│   ├── PipelineEvents.tsx     # step-by-step graph execution
│   ├── CriticReportPanel.tsx  # per-principle scores
│   ├── ExtractedDataPanel.tsx # final JSON + self-healed badge
│   └── MemoryPanel.tsx        # recent errors from /api/memory/errors
├── data/
│   └── samples.ts             # 5 built-in sample documents
└── lib/
    ├── api.ts                 # health, listRecentErrors, streamPipeline (SSE)
    ├── types.ts               # TS mirrors of the backend's Pydantic schemas
    └── cn.ts                  # tailwind class merger
```

## Tech notes

- **SSE over POST.** The backend uses `sse-starlette` and accepts the
  document body as POST JSON, so the browser's built-in `EventSource`
  (GET-only) is not usable. `lib/api.ts:streamPipeline` reads the
  `ReadableStream` returned by `fetch` and parses the `event:`/`data:`
  frames manually.
- **No state library.** Three `useState` hooks in `app/page.tsx` are
  enough for a single-page demo.
- **No images / no fonts to load.** The UI is text-first by design so the
  page is instantly interactive.
