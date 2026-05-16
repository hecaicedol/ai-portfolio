# Frontend — Workflow Engine

Next.js 14 + Tailwind + `react-flow` + shadcn/ui.

## Views

- **`app/page.tsx`** — Goal input form → planner returns DAG → review screen with the DAG rendered in `react-flow` (read-only). User clicks "Approve & Run" to start execution.
- **`app/run/[id]/page.tsx`** — Live execution: nodes change color as states transition (pending/running/awaiting_approval/complete/failed). SSE stream from `/api/workflows/{id}/stream`.
- **`app/history/page.tsx`** — Past workflows from `workflow_memory`. Click → re-run with same or edited parameters.
- **`components/ApprovalModal.tsx`** — Pops up when a node hits `awaiting_approval`: shows action, params (editable), impact assessment. Approve / Reject.

## Setup

```bash
npx create-next-app@14.2.15 . --typescript --tailwind --app --import-alias="@/*"
npm install reactflow lucide-react
npx shadcn@latest init && npx shadcn@latest add dialog button card input badge
```

`.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
