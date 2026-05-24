# Deploying the P1 frontend to Vercel

Public, free, takes ~5 minutes from cold start. Recruiters can open the URL
and play the scripted self-healing demo without any backend running.

## Pre-flight checks

1. The commit with the frontend code must be pushed to `origin/main`:
   ```bash
   git push origin main
   ```
2. You need a Vercel account. Sign up at https://vercel.com — pick "Continue
   with GitHub" so it can read your repos.

## One-time setup

1. From the Vercel dashboard, click **Add New → Project**.
2. Pick `hecaicedol/ai-portfolio` from your repo list and click **Import**.
3. Configure the project:
   - **Framework Preset:** Vercel will auto-detect Next.js (since this repo
     has a Next config) — but it won't find it in the root. Override:
   - **Root Directory:** click **Edit** next to it and set
     `01-self-healing-pipeline/frontend`.
   - **Build Command:** leave as `next build` (default).
   - **Output Directory:** leave as `.next` (default).
   - **Install Command:** leave as `npm install` (default).
4. **Environment variables** — none are required for the public demo:
   - `NEXT_PUBLIC_API_BASE_URL` defaults to `http://localhost:8000`, which is
     unreachable from a recruiter's browser → the page auto-flips to demo
     mode. That's exactly the behavior we want.
   - If you later deploy the backend (Railway / Fly.io), set this variable
     to the backend's public URL and redeploy. The frontend will auto-detect
     it and switch to live mode.
5. Click **Deploy**. First build takes 1–2 minutes.

## After deploy

- Vercel gives you a URL like `https://ai-portfolio-xxxxx.vercel.app` —
  every push to `main` triggers an auto-redeploy.
- Add a custom domain later if you want (`heims.dev` / `caicedo.ai` /
  whatever) from **Project → Settings → Domains**.
- For previews on PRs: Vercel does that automatically once the repo is
  connected.

## What a visitor sees

1. The page loads, shows the Globex invoice in the input panel.
2. The header badge reads **Demo mode · scripted scenario** (because the
   page tried to reach `http://localhost:8000/health` and got nothing).
3. Clicking **Play scripted demo** plays a ~7-second client-side scenario:
   the extractor first grabs the wrong total, the critic flags it
   (accuracy + consistency below threshold), reflection saves the errors,
   the extractor re-runs with the feedback, the critic passes at 93%,
   `self-healed` badge appears on the final output.
4. The episodic memory panel populates with 5 plausible-looking past
   errors so the panel isn't empty.

All client-side. No LLM calls. $0 cost per visit.

## CLI alternative (skip the web UI)

If you have the Vercel CLI installed:
```bash
cd 01-self-healing-pipeline/frontend
npx vercel              # interactive: first time will prompt for the project
npx vercel --prod       # later, to push a production deploy
```
