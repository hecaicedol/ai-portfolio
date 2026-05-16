# Frontend — Agentic RAG Benchmark Dashboard

Next.js 14 + Tailwind + shadcn/ui + `recharts`.

## Views

1. **`app/page.tsx`** — Document upload (drag-and-drop). Shows per-store ingestion progress and contextual-enrichment status.
2. **`app/query/page.tsx`** — Question input. Side-by-side cards: results from pgvector | Qdrant | Pinecone (answer, source chunks, latency, Ragas scores).
3. **`app/benchmark/page.tsx`** — Live dashboard (auto-refresh 5s):
   - Latency time-series (recharts `LineChart`): p50/p95 per store.
   - Quality time-series: Ragas scores per store.
   - Cost tracker: $ per 1000 queries per store.
   - Winner indicator: which store wins on each metric *right now*.
4. **`app/optimizer/page.tsx`** — Timeline of automatic optimizer interventions, with before/after metrics deltas.

## Setup

```bash
npx create-next-app@14.2.15 . --typescript --tailwind --app --import-alias="@/*"
npx shadcn@latest init && npx shadcn@latest add card tabs badge button input
npm install recharts lucide-react
```

`.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
