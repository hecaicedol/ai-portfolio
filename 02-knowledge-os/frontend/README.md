# Frontend — Knowledge OS

Next.js 14 + Tailwind + `react-force-graph-2d`.

## What lives here

- `app/page.tsx` — Query panel: ask a question, get a GraphRAG answer with the reasoning path highlighted in the graph viz.
- `app/graph/page.tsx` — Full force-graph view of the knowledge graph (filterable by entity type).
- `app/staleness/page.tsx` — Latest staleness report from the background agent.
- `components/ForceGraph.tsx` — `react-force-graph-2d` with type-based coloring and click-to-expand.

## Setup

```bash
npx create-next-app@14.2.15 . --typescript --tailwind --app --import-alias="@/*"
npm install react-force-graph-2d d3
```

`.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Run

```bash
npm install && npm run dev   # → http://localhost:3000
```
