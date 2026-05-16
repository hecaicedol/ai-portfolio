# Frontend — Debate System

Next.js 14 + Tailwind + framer-motion for the animated debate room.

## Layout

- 5 agent cards arranged in a circle (icon + role + live stance badge).
- Center: round indicator (1/2/3 + Final) and the consensus gauge.
- Statements feed below — each new statement animates *from* the agent's card to the feed.
- After Round 3: the executive memo card slides in with a "Download PDF" button.

## Setup

```bash
npx create-next-app@14.2.15 . --typescript --tailwind --app --import-alias="@/*"
npm install framer-motion lucide-react
```

`.env.local`:
```
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Run

```bash
npm install && npm run dev
```
