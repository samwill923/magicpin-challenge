# Deploying the Vera bot to Render

The repo is committed and `render.yaml` is ready. Render cannot be driven from this
machine (no Render API key, no `gh`/`render` CLI), so the two account steps below are yours.

## 1. Push to a Git remote

```bash
# GitHub: create an empty repo first (github.com/new), then
git remote add origin https://github.com/<you>/magicpin-vera-bot.git
git branch -M main
git push -u origin main
```

`.env` and `.llm_cache.json` are gitignored, so no keys leave your machine.

## 2. Create the service on Render

1. render.com → **New** → **Blueprint** → connect the repo → Render reads `render.yaml`.
2. It will prompt for the two secrets marked `sync: false`:
   - `GROQ_API_KEY` — required
   - `GEMINI_API_KEY` — optional backup provider
3. **Apply**. First build takes ~2-3 minutes (installs FastAPI + uvicorn, expands the dataset).

Then verify:

```bash
curl https://<your-service>.onrender.com/v1/healthz
curl https://<your-service>.onrender.com/v1/metadata
BOT_URL=https://<your-service>.onrender.com python selftest.py   # 174 contract checks
```

Submit `https://<your-service>.onrender.com` as the bot URL.

## Things that will bite you if ignored

- **Free plan spins down after 15 minutes of inactivity.** A cold start takes 30-60s, and the
  judge's `/v1/healthz` budget is 2s with disqualification after 3 consecutive failures. Before
  the scored window either upgrade to Starter, or wake the service and keep it warm (a 1-minute
  cron hitting `/v1/healthz`). This is the single biggest operational risk in the deployment.
- **One worker only.** Context and conversation state live in memory; `--workers 2` would serve
  half the judge's calls from an empty store. Already set in `render.yaml` — do not raise it.
- **Render restarts wipe state.** The testing brief requires context to persist for the whole
  test ("don't restart between calls"), so do not push a commit during the test window —
  `autoDeployTrigger: commit` would redeploy and drop every stored context mid-run.
- **Groq quota is the quality ceiling**, not the server. 8000 tokens/minute and 200k/day per
  model means roughly three fresh compositions a minute; past that the bot serves its
  deterministic template instead of an LLM message. Two mitigations:
  - Run `python prewarm.py` locally and commit the cache so the deployed instance starts warm:
    `git add -f .llm_cache.json && git commit -m "prewarm cache" && git push`
    (the cache holds prompt hashes and model output only — no keys). Note the facts block
    includes the current month, so a cache warmed in one month misses in the next.
  - Keep `JUDGE_MODEL` different from `GROQ_MODEL` when running `run_judge.py`, so scoring and
    composing draw on separate per-model pools.

## Alternative: any other host

Nothing here is Render-specific. Any platform that runs
`uvicorn bot:app --host 0.0.0.0 --port $PORT --workers 1` with `GROQ_API_KEY` set will work
(Fly, Railway, a VM, or `ngrok http 8080` against a local run for a quick test).
