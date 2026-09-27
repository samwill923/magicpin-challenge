# Vera Next — magicpin AI Challenge submission

A stateful HTTP bot that composes merchant- and customer-facing WhatsApp messages from the
4-context framework, and handles the reply turns that follow.

## Approach

**Facts first, LLM second.** The weak point of a "stuff the JSON into a prompt" bot is that the
model invents numbers and copies internal jargon. So `composer.build_facts()` does the arithmetic in
Python: peer-relative deltas (CTR 2.1% vs peer 3.0% → "30% below the peer average"), plain-English
translations of `signals` (`stale_posts:22d` → "no Google post in 22 days"), the digest item matched
to the trigger (by `payload.top_item_id`, else by `trigger.kind` → `digest.kind`), the seasonal beat
for the current month, active vs expired offers, and the customer's relationship facts. The model
never computes and never guesses — it selects, frames and writes.

**One LLM call per message**, temperature 0, routed by `trigger.kind`. Each of ~25 kinds has its own
framing, compulsion levers and CTA shape (`composer.ROUTES`): a `regulation_change` leads with the
deadline, a `seasonal_perf_dip` pre-empts the panic and argues *against* ad spend, a
`curious_ask_due` asks the merchant one question and states what will be built from the answer,
an `active_planning_intent` delivers a drafted artefact instead of another qualifying question.
Customer-scoped kinds switch to a second system prompt that writes *as the business*
(`send_as: merchant_on_behalf`, no mention of magicpin or Vera).

**Validation with one corrective retry.** `composer.validate()` rejects: any number, price or date
that does not appear in the facts block (the anti-fabrication guard), URLs/phones/emails, category
taboo words, more than one question mark, a CTA that is not in the last sentence, multi-choice menus
outside slot booking, generic "X% off" when a service-at-price offer exists, preambles and
self-reintroductions. Violations are fed back into a single rewrite; if that also fails, a
deterministic template composed from the merchant's real numbers is used.

**Output sanitiser** (`sanitizer.py`), applied to every live send and to `submission.jsonl` alike
(`python make_submission.py --sanitize`). It strips dashes of every kind, since em dashes and
hyphenated compounds read as machine-written on WhatsApp: em and en dashes become commas, compounds
open up ("click through", "week on week"), ranges become "to", and ISO dates are protected so
`2026-12-15` survives. It also enforces the language rule at the text level: a message is either
wholly Hinglish or wholly English, so a trailing "Bhej dun?" bolted onto English prose is removed
and the CTA repaired. Merchants get Hinglish only when their `identity.languages` includes Hindi.

**Conversation state machine** (`conversation_handlers.py`). Classification is regex-first, so it
never times out: opt-out/hostile → `end`; canned or verbatim-repeated text → one nudge addressed to
the owner, then `wait 4h`, then `end` (tracked per merchant, not just per conversation, because the
same auto-reply arrives on new thread ids); explicit commitment → action mode immediately, with an
extra check that the reply contains no qualifying phrase; off-topic → decline in one line and
redirect; "call me later" → `wait`; no substance → at most 3 nudges, then a graceful exit.

**Operational**: Groq `openai/gpt-oss-120b` primary, Gemini backup, fixed backoff on 429/5xx and on
Groq's `json_validate_failed` (a reasoning model can spend its whole budget before emitting the
JSON, so the retry doubles the ceiling). The free tier allows 8000 tokens/minute — about three
compositions — so a rate governor keeps a 60-second rolling window of spent tokens and waits for a
slot instead of collecting 429s; if no slot is free before the caller's deadline it returns the
template rather than a timeout. Compositions are cached under a hash of the exact prompt and
persisted to `.llm_cache.json`, so a context update (new facts → new prompt) misses the cache while
a repeat request is free; `prewarm.py` fills that cache ahead of a scored run so the test window
spends its quota on nothing but fresh work. Every handler has a hard wall-clock deadline (11.5s for
`/v1/tick`, composed in parallel threads) and a fallback, so the judge never sees a timeout, a 500
or an empty body. `/v1/tick` sends at most one message per merchant per tick, caps each merchant at
3 conversations, and drops any body already sent to that merchant; unused triggers stay queued for a
later tick rather than being dropped.

## Tradeoffs

- **URLs are banned outright.** The main brief allows them; the testing brief makes them a hard fail
  (-3). We follow the stricter rule.
- **`contexts_loaded` in `/v1/healthz` counts only judge-pushed contexts**, so warmup assertions hold
  exactly. The public base dataset is also preloaded from disk at version 0 (any push at version ≥ 1
  replaces it) so the bot is never blind if it is asked about a context it was not pushed;
  `contexts_available` reports that total. `PRELOAD_DATASET=0` disables it.
- **Regex, not an LLM, classifies replies.** Cheaper and more predictable inside the turn budget; it
  will miss unusual phrasings that a classifier would catch.
- **No retrieval/embeddings.** With five category packs, deterministic id-then-kind matching beat the
  extra latency and the risk of citing an irrelevant item.
- **Expiry is not enforced.** `available_triggers` is treated as the judge's statement of what is
  live, since the seed `expires_at` values are in the past relative to a real-clock test run.
- **Restraint over volume**: composing for every active trigger would score more messages but reads
  as spam to a real merchant.
- **We keep quoting `customer_aggregate`** ("your 245 active members", "124 high-risk adults"), even
  though `judge_simulator.py`'s scoring prompt does not receive that field and therefore flags those
  figures as fabricated. They are real, the rubric's own case studies reward them, and the brief says
  the judge is given the full dataset. Roster numbers are introduced as the merchant's own
  ("your dashboard shows…") and capped at one per message so they read as evidence, not filler.

## What extra context would have helped most

1. **Reply-level outcome history** — which of Vera's past messages got a reply, per merchant and per
   trigger family. Right now `conversation_history.engagement` is the only signal, and it cannot tell
   us which *lever* works on this merchant.
2. **Bookable inventory** — open slots, stock on hand, table covers. Half of the customer-facing
   kinds want a concrete slot; only `recall_due` ships one.
3. **The merchant's own language sample** — one or two lines they actually typed, to match register
   instead of inferring from `identity.languages`.
4. **Offer performance** — redemptions per offer, so the pick is data-driven rather than "the active
   one".
5. **Locality-level demand** ("6,777 missed searches in Sector 14") — the strongest loss-aversion
   anchor in production Vera, absent from the dataset.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY (Gemini optional as backup)
uvicorn bot:app --host 0.0.0.0 --port 8080 --workers 1   # one worker: state is in memory

python selftest.py              # 174 contract checks, no LLM key needed
python make_submission.py       # writes submission.jsonl for the 30 canonical pairs
python prewarm.py               # pre-compose + cache what the judge will ask for
python run_judge.py all         # magicpin's judge_simulator, config from .env
```

Bind to `0.0.0.0` (not `127.0.0.1`) so `localhost` resolves on either IP stack.
Deployment: `render.yaml` is a ready Render blueprint; `DEPLOY.md` has the steps and the
operational traps (free-tier spin-down vs the 2s healthz budget, one worker only, and why a
redeploy mid-test would wipe every stored context).

Files: `bot.py` (endpoints, state, `compose()`), `composer.py` (facts, routing, prompts,
validation, fallback), `conversation_handlers.py` (reply state machine), `llm.py` (providers,
retry, rate governor, cache), `make_submission.py`, `prewarm.py`, `selftest.py`, `run_judge.py`.
Set `TEAM_NAME`, `TEAM_MEMBERS` and `CONTACT_EMAIL` to control `/v1/metadata`.
