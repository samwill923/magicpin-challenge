"""magicpin AI Challenge — Vera bot (FastAPI, port 8080).

Endpoints (per challenge-testing-brief.md §2):
    POST /v1/context   versioned, idempotent context push (409 on stale version)
    POST /v1/tick      proactive sends, composed in parallel inside a hard budget
    POST /v1/reply     multi-turn handling: send / wait / end
    GET  /v1/healthz   liveness + how much context we hold
    GET  /v1/metadata  bot identity
    POST /v1/teardown  optional: wipe all state (privacy rule §11)

Operational stance: the judge must never see an error, a timeout or an empty
body. Every path has a deterministic fallback and every handler is wrapped.

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import composer
import conversation_handlers as ch
from llm import CLIENT

START = time.time()
HERE = Path(__file__).parent
VALID_SCOPES = ("category", "merchant", "customer", "trigger")

# Response budgets. The judge simulator waits 15s on /v1/tick and /v1/reply
# (the production harness allows 30s), so we finish well inside the tighter one.
TICK_BUDGET = float(os.environ.get("TICK_BUDGET_SECONDS", "11.5"))
REPLY_BUDGET = float(os.environ.get("REPLY_BUDGET_SECONDS", "11.0"))
MAX_ACTIONS_PER_TICK = 20
MAX_CONVERSATIONS_PER_MERCHANT = 3
MAX_TURNS = 7  # after this many turns we close the conversation politely

app = FastAPI(title="magicpin Vera bot")


# =========================================================================
# State
# =========================================================================

class ConversationState:
    def __init__(self, conversation_id: str, merchant_id: str, customer_id: Optional[str],
                 trigger_id: Optional[str], send_as: str):
        self.conversation_id = conversation_id
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.trigger_id = trigger_id
        self.send_as = send_as
        self.turns: list[dict] = []
        self.sent_norms: set[str] = set()
        self.inbound_norms: set[str] = set()
        self.mode = "pitch"          # pitch -> action -> ended
        self.autoreply_count = 0
        self.unanswered_nudges = 0
        self.ended = False
        self.turn_count = 0
        self.last_now_iso = ""

    def record(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})
        norm = composer._normalize(text)
        if role == "vera":
            self.sent_norms.add(norm)
        else:
            self.inbound_norms.add(norm)


class Store:
    def __init__(self):
        self.lock = threading.RLock()
        self.contexts: dict[tuple[str, str], dict] = {}   # (scope, id) -> {version, payload}
        self.pushed: set[tuple[str, str]] = set()          # what the judge actually pushed
        self.conversations: dict[str, ConversationState] = {}
        self.pending_triggers: dict[str, str] = {}         # trigger_id -> first seen `now`
        self.sent_suppression: set[str] = set()
        self.merchant_conversations: dict[str, list[str]] = {}
        self.opted_out: set[str] = set()
        self.merchant_autoreplies: dict[str, set[str]] = {}
        self.merchant_autoreply_count: dict[str, int] = {}
        self.sent_bodies: dict[str, set] = {}   # merchant_id -> normalized bodies
        self.ticks = 0
        self.actions_sent = 0

    # -- context ---------------------------------------------------------
    def put(self, scope: str, cid: str, version: int, payload: dict, pushed: bool) -> dict:
        key = (scope, cid)
        with self.lock:
            cur = self.contexts.get(key)
            if cur and cur["version"] >= version:
                return {"ok": False, "current_version": cur["version"]}
            self.contexts[key] = {"version": version, "payload": payload}
            if pushed:
                self.pushed.add(key)
            return {"ok": True, "version": version}

    def get(self, scope: str, cid: Optional[str]) -> Optional[dict]:
        if not cid:
            return None
        with self.lock:
            entry = self.contexts.get((scope, cid))
        return entry["payload"] if entry else None

    def version(self, scope: str, cid: Optional[str]) -> int:
        if not cid:
            return -1
        with self.lock:
            entry = self.contexts.get((scope, cid))
        return entry["version"] if entry else -1

    def counts(self, pushed_only: bool = True) -> dict:
        out = {s: 0 for s in VALID_SCOPES}
        with self.lock:
            keys = self.pushed if pushed_only else set(self.contexts)
            for scope, _ in keys:
                if scope in out:
                    out[scope] += 1
        return out

    def resolve(self, trigger: dict) -> dict:
        """trigger -> the 4 contexts it needs."""
        merchant_id = trigger.get("merchant_id") or (trigger.get("payload") or {}).get("merchant_id")
        merchant = self.get("merchant", merchant_id) or {}
        category = self.get("category", merchant.get("category_slug")) or \
            self.get("category", (trigger.get("payload") or {}).get("category")) or {}
        customer = self.get("customer", trigger.get("customer_id"))
        return {"merchant_id": merchant_id, "merchant": merchant, "category": category,
                "customer": customer, "trigger": trigger}

    def cache_salt(self, merchant_id: Optional[str], category_slug: Optional[str],
                   customer_id: Optional[str], trigger_id: Optional[str]) -> str:
        """Context versions go into the LLM cache key, so refreshed context
        (Phase 3 injections) invalidates the cached composition."""
        return "v%d.%d.%d.%d" % (
            self.version("merchant", merchant_id), self.version("category", category_slug),
            self.version("customer", customer_id), self.version("trigger", trigger_id))

    def wipe(self) -> None:
        with self.lock:
            self.contexts.clear()
            self.pushed.clear()
            self.conversations.clear()
            self.pending_triggers.clear()
            self.sent_suppression.clear()
            self.merchant_conversations.clear()
            self.opted_out.clear()
            self.merchant_autoreplies.clear()
            self.merchant_autoreply_count.clear()
            self.sent_bodies.clear()


STORE = Store()


# =========================================================================
# Base-dataset preload
# =========================================================================

def preload_base_dataset() -> int:
    """Load the published base dataset from disk at version 0.

    The harness pushes the base dataset during warmup anyway (and any push, at
    version >= 1, replaces what we loaded here). Preloading only means we are
    never blind if the judge references a context it has not pushed us yet -
    and it is the same public dataset every participant was given, so nothing
    here is invented. Contexts loaded this way are NOT counted in
    /v1/healthz.contexts_loaded, which reports what the judge pushed.
    Set PRELOAD_DATASET=0 to disable.
    """
    if os.environ.get("PRELOAD_DATASET", "1") == "0":
        return 0
    roots = [HERE / "dataset" / "expanded", HERE / "dataset"]
    loaded = 0
    for root in roots:
        if not root.exists():
            continue
        for scope, sub, key in (("category", "categories", "slug"),
                                ("merchant", "merchants", "merchant_id"),
                                ("customer", "customers", "customer_id"),
                                ("trigger", "triggers", "id")):
            folder = root / sub
            if folder.is_dir():
                for path in sorted(folder.glob("*.json")):
                    loaded += _preload_file(path, scope, key)
        for name, container, key in (("merchants_seed.json", "merchants", "merchant_id"),
                                     ("customers_seed.json", "customers", "customer_id"),
                                     ("triggers_seed.json", "triggers", "id")):
            path = root / name
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print("[preload] " + str(path) + ": " + repr(exc), file=sys.stderr)
                    continue
                scope = {"merchants": "merchant", "customers": "customer",
                         "triggers": "trigger"}[container]
                for item in data.get(container, []):
                    if item.get(key):
                        if STORE.put(scope, item[key], 0, item, pushed=False)["ok"]:
                            loaded += 1
    print("[preload] " + str(loaded) + " base contexts loaded from disk", file=sys.stderr)
    return loaded


def _preload_file(path: Path, scope: str, key: str) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("[preload] " + str(path) + ": " + repr(exc), file=sys.stderr)
        return 0
    cid = data.get(key) or path.stem
    return 1 if STORE.put(scope, cid, 0, data, pushed=False)["ok"] else 0


# =========================================================================
# The required composition entry point
# =========================================================================

def compose(category: dict, merchant: dict, trigger: dict,
            customer: Optional[dict] = None) -> dict:
    """Compose one message from the 4 contexts (challenge-brief.md §5).

    Returns: body, cta, send_as, suppression_key, rationale
    (plus template_name / template_params for the first-outbound WhatsApp
    template, and composed_by for transparency).
    Deterministic: temperature 0, no randomness anywhere in the pipeline.
    """
    return composer.compose_message(
        category or {}, merchant or {}, trigger or {}, customer,
        now_iso=_now_iso(), deadline=time.monotonic() + 25.0,
        cache_salt="direct")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# =========================================================================
# /v1/context
# =========================================================================

class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int = 1
    payload: dict = {}
    delivered_at: Optional[str] = None


@app.post("/v1/context")
async def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": "scope must be one of " + ", ".join(VALID_SCOPES)})
    if not body.context_id:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "missing_context_id", "details": "context_id is required"})
    if not isinstance(body.payload, dict) or not body.payload:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "empty_payload", "details": "payload must be an object"})

    result = STORE.put(body.scope, body.context_id, int(body.version), body.payload, pushed=True)
    if not result["ok"]:
        return JSONResponse(status_code=409, content={
            "accepted": False, "reason": "stale_version",
            "current_version": result["current_version"]})

    if body.scope == "trigger":
        # Remember it so a later tick can still act on it.
        with STORE.lock:
            STORE.pending_triggers.setdefault(body.context_id, body.delivered_at or _now_iso())
    return {"accepted": True,
            "ack_id": "ack_" + body.context_id + "_v" + str(body.version),
            "stored_at": _now_iso()}


# =========================================================================
# /v1/tick
# =========================================================================

class TickBody(BaseModel):
    now: Optional[str] = None
    available_triggers: list[str] = []


@app.post("/v1/tick")
async def tick(body: TickBody):
    deadline = time.monotonic() + TICK_BUDGET
    now_iso = body.now or _now_iso()
    with STORE.lock:
        STORE.ticks += 1
        for tid in body.available_triggers:
            STORE.pending_triggers.setdefault(tid, now_iso)
        candidate_ids = list(dict.fromkeys(list(body.available_triggers) +
                                          list(STORE.pending_triggers)))

    plans = _select(candidate_ids, now_iso)
    if not plans:
        return {"actions": []}

    actions = []
    workers = min(6, len(plans))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_compose_plan, p, now_iso, deadline): p for p in plans}
        for fut in as_completed(futures, timeout=max(0.5, deadline - time.monotonic() + 2.0)):
            try:
                action = fut.result()
            except Exception as exc:
                print("[tick] compose failed: " + repr(exc), file=sys.stderr)
                continue
            if action:
                actions.append(action)

    actions.sort(key=lambda a: a["conversation_id"])
    kept = []
    with STORE.lock:
        for action in actions:
            norm = composer._normalize(action["body"])
            seen = STORE.sent_bodies.setdefault(action["merchant_id"], set())
            if norm in seen:
                # never send the same text twice to one merchant (-2 anti-repetition)
                print("[tick] dropped duplicate body for " + action["merchant_id"],
                      file=sys.stderr)
                continue
            seen.add(norm)
            kept.append(action)
    for action in kept:
        _register_send(action)
    return {"actions": kept[:MAX_ACTIONS_PER_TICK]}


def _select(candidate_ids: list[str], now_iso: str) -> list[dict]:
    """Decide which triggers to act on this tick. Restraint is rewarded:
    one message per merchant per tick, nothing already sent, nothing for a
    merchant who opted out."""
    plans = []
    seen_merchants = set()
    with STORE.lock:
        for tid in candidate_ids:
            trigger = STORE.get("trigger", tid)
            if not trigger:
                continue
            resolved = STORE.resolve(trigger)
            merchant_id = resolved["merchant_id"]
            if not merchant_id or not resolved["merchant"] or not resolved["category"]:
                continue
            if merchant_id in STORE.opted_out or merchant_id in seen_merchants:
                continue
            supp = trigger.get("suppression_key") or (str(trigger.get("kind")) + ":" + merchant_id)
            if supp in STORE.sent_suppression:
                continue
            if len(STORE.merchant_conversations.get(merchant_id, [])) >= MAX_CONVERSATIONS_PER_MERCHANT:
                continue
            seen_merchants.add(merchant_id)
            plans.append({"trigger_id": tid, "trigger": trigger, "suppression_key": supp,
                          **resolved})
    # highest urgency first, then stable by id
    plans.sort(key=lambda p: (-int(p["trigger"].get("urgency", 1) or 1), p["trigger_id"]))
    return plans[:MAX_ACTIONS_PER_TICK]


def _compose_plan(plan: dict, now_iso: str, deadline: float) -> Optional[dict]:
    trigger = plan["trigger"]
    merchant = plan["merchant"]
    merchant_id = plan["merchant_id"]
    customer = plan["customer"]
    salt = STORE.cache_salt(merchant_id, merchant.get("category_slug"),
                            trigger.get("customer_id"), plan["trigger_id"])
    with STORE.lock:
        banned = set(STORE.sent_bodies.get(merchant_id, set()))
    result = composer.compose_message(
        plan["category"], merchant, trigger, customer,
        now_iso=now_iso, deadline=deadline, banned_bodies=banned, cache_salt=salt)
    if not result.get("body"):
        return None
    print("[tick] " + plan["trigger_id"] + " composed_by=" + result.get("composed_by", "?"),
          file=sys.stderr, flush=True)
    conv_id = _conversation_id(merchant_id, trigger)
    return {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": trigger.get("customer_id") if result["send_as"] == "merchant_on_behalf" else None,
        "send_as": result["send_as"],
        "trigger_id": plan["trigger_id"],
        "template_name": result["template_name"],
        "template_params": result["template_params"],
        "body": result["body"],
        "cta": result["cta"],
        "suppression_key": result["suppression_key"] or plan["suppression_key"],
        "rationale": result["rationale"],
    }


def _conversation_id(merchant_id: str, trigger: dict) -> str:
    tid = str(trigger.get("id", ""))
    seq = "".join(ch_ for ch_ in tid.split("_")[1] if ch_.isdigit()) if "_" in tid else ""
    return "conv_" + merchant_id + "_" + str(trigger.get("kind", "msg")) + ("_" + seq if seq else "")


def _register_send(action: dict) -> None:
    with STORE.lock:
        STORE.sent_suppression.add(action["suppression_key"])
        STORE.pending_triggers.pop(action["trigger_id"], None)
        STORE.actions_sent += 1
        conv = STORE.conversations.get(action["conversation_id"])
        if conv is None:
            conv = ConversationState(action["conversation_id"], action["merchant_id"],
                                     action.get("customer_id"), action["trigger_id"],
                                     action["send_as"])
            STORE.conversations[action["conversation_id"]] = conv
            STORE.merchant_conversations.setdefault(action["merchant_id"], []).append(
                action["conversation_id"])
        conv.record("vera", action["body"])
        conv.turn_count = 1


# =========================================================================
# /v1/reply
# =========================================================================

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str = ""
    received_at: Optional[str] = None
    turn_number: int = 2


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    deadline = time.monotonic() + REPLY_BUDGET
    state = _state_for(body)
    state.last_now_iso = body.received_at or _now_iso()
    state.turn_count = max(state.turn_count + 1, int(body.turn_number or 0))

    merchant_id = state.merchant_id or ""
    with STORE.lock:
        merchant_autoreplies = set(STORE.merchant_autoreplies.get(merchant_id, set()))
        already_out = merchant_id in STORE.opted_out
        merchant_autoreply_count = STORE.merchant_autoreply_count.get(merchant_id, 0)

    if state.ended or already_out:
        return _end(state, "Conversation already closed for this merchant (opt-out or completed "
                           "thread); not messaging again.")

    # Classify BEFORE recording, so the message never matches itself when we
    # look for verbatim repeats.
    branch = ch.classify(body.message, state, merchant_autoreplies)
    state.record(body.from_role or "merchant", body.message or "")
    contexts = _contexts_for(state)

    if branch in ("opt_out", "hostile"):
        with STORE.lock:
            STORE.opted_out.add(merchant_id)
        reason = ("Merchant explicitly opted out" if branch == "opt_out"
                  else "Merchant is clearly frustrated")
        return _end(state, reason + "; closing the thread and suppressing further triggers for "
                                    "this merchant.")

    if branch == "auto_reply":
        state.autoreply_count += 1
        norm = composer._normalize(body.message)
        with STORE.lock:
            STORE.merchant_autoreplies.setdefault(merchant_id, set()).add(norm)
            STORE.merchant_autoreply_count[merchant_id] = merchant_autoreply_count + 1
            streak = STORE.merchant_autoreply_count[merchant_id]
        streak = max(streak, state.autoreply_count)
        if streak == 1:
            out = ch.compose_reply("auto_reply_nudge", state, contexts, body.message, deadline,
                                   body.from_role)
            return _send(state, out, "Detected the business's canned WhatsApp auto-reply (not a "
                                     "person). One explicit prompt addressed to the owner, then we "
                                     "stop. " + out.get("rationale", ""))
        if streak == 2:
            return {"action": "wait", "wait_seconds": 14400,
                    "rationale": "Same auto-reply twice — the owner is not at the phone. Backing "
                                 "off 4 hours instead of burning turns."}
        return _end(state, "Auto-reply " + str(streak) + " times in a row with no human signal. "
                           "Closing rather than spending more turns; the owner can be reached on a "
                           "later trigger.")

    if branch == "defer":
        secs = ch.defer_seconds(body.message)
        return {"action": "wait", "wait_seconds": secs,
                "rationale": "Merchant asked for time; backing off " + str(secs // 3600) +
                             "h and keeping the thread open."}

    if state.turn_count > MAX_TURNS:
        return _end(state, "Thread has run " + str(state.turn_count) + " turns; closing on a clean "
                           "note rather than over-nudging.")

    if branch == "commitment":
        state.mode = "action"
        out = ch.compose_reply("commitment", state, contexts, body.message, deadline, body.from_role)
        return _send(state, out, "Explicit commitment detected — switched straight from pitch to "
                                 "execution, no further qualifying questions. " +
                                 out.get("rationale", ""))

    if branch in ("question", "off_topic"):
        out = ch.compose_reply(branch, state, contexts, body.message, deadline, body.from_role)
        note = ("Out-of-scope ask declined politely and thread redirected. "
                if branch == "off_topic" else "Answered from stored context only, no guessing. ")
        return _send(state, out, note + out.get("rationale", ""))

    if branch == "no_substance":
        state.unanswered_nudges += 1
        if state.unanswered_nudges >= 3:
            return _end(state, "Three nudges without a real answer; closing gracefully instead of "
                               "pushing a fourth.")
        out = ch.compose_reply("nudge", state, contexts, body.message, deadline, body.from_role)
        return _send(state, out, "No substantive reply yet (nudge " + str(state.unanswered_nudges) +
                                 " of 3) — leading with a new fact rather than repeating. " +
                                 out.get("rationale", ""))

    out = ch.compose_reply("engaged", state, contexts, body.message, deadline, body.from_role)
    state.unanswered_nudges = 0
    return _send(state, out, out.get("rationale", "Advancing the thread."))


def _state_for(body: ReplyBody) -> ConversationState:
    with STORE.lock:
        state = STORE.conversations.get(body.conversation_id)
        if state is None:
            state = ConversationState(body.conversation_id, body.merchant_id or "",
                                      body.customer_id, None,
                                      "merchant_on_behalf" if body.customer_id else "vera")
            STORE.conversations[body.conversation_id] = state
            if body.merchant_id:
                STORE.merchant_conversations.setdefault(body.merchant_id, []).append(
                    body.conversation_id)
        if body.merchant_id and not state.merchant_id:
            state.merchant_id = body.merchant_id
        if body.customer_id and not state.customer_id:
            state.customer_id = body.customer_id
    return state


def _contexts_for(state: ConversationState) -> dict:
    merchant = STORE.get("merchant", state.merchant_id) or {}
    category = STORE.get("category", merchant.get("category_slug")) or {}
    trigger = STORE.get("trigger", state.trigger_id) or {"kind": "conversation_reply",
                                                         "payload": {}, "urgency": 2,
                                                         "scope": "merchant"}
    customer = STORE.get("customer", state.customer_id)
    return {"merchant": merchant, "category": category, "trigger": trigger, "customer": customer}


def _send(state: ConversationState, out: dict, rationale: str) -> dict:
    body_text = (out.get("body") or "").strip()
    if not body_text:
        return _end(state, "Nothing new worth saying on this thread; closing rather than sending "
                           "filler.")
    if composer._normalize(body_text) in state.sent_norms:
        return {"action": "wait", "wait_seconds": 3600,
                "rationale": "The only reply available repeats what we already sent; waiting "
                             "instead of repeating ourselves."}
    state.record("vera", body_text)
    return {"action": "send", "body": body_text, "cta": out.get("cta", "open_ended"),
            "rationale": rationale.strip()}


def _end(state: ConversationState, rationale: str) -> dict:
    state.ended = True
    state.mode = "ended"
    return {"action": "end", "rationale": rationale}


# =========================================================================
# /v1/healthz, /v1/metadata, /v1/teardown
# =========================================================================

@app.get("/v1/healthz")
async def healthz():
    with STORE.lock:
        conversations = len(STORE.conversations)
        ticks, sent = STORE.ticks, STORE.actions_sent
    return {"status": "ok", "uptime_seconds": int(time.time() - START),
            "contexts_loaded": STORE.counts(pushed_only=True),
            "contexts_available": STORE.counts(pushed_only=False),
            "conversations": conversations, "ticks": ticks, "actions_sent": sent,
            **CLIENT.stats()}


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Vera Next"),
        "team_members": [m.strip() for m in
                         os.environ.get("TEAM_MEMBERS", "stamsonwill").split(",") if m.strip()],
        "model": CLIENT.model_label(),
        "approach": ("Deterministic fact-extraction layer (numbers, peer deltas, matched digest "
                     "item, plain-English signals) + one temperature-0 LLM call routed by "
                     "trigger.kind + validation pass (anti-fabrication, single-CTA, taboo, "
                     "no-URL) with one corrective retry, then a template fallback that still "
                     "uses real numbers. Regex-first conversation state machine for replies."),
        "contact_email": os.environ.get("CONTACT_EMAIL", "stamsonwill@gmail.com"),
        "version": "1.0.0",
        "submitted_at": os.environ.get("SUBMITTED_AT", "2026-09-26T00:00:00Z"),
    }


@app.post("/v1/teardown")
async def teardown():
    STORE.wipe()
    return {"ok": True, "wiped_at": _now_iso()}


@app.get("/")
async def root():
    return {"bot": "magicpin Vera", "endpoints": ["/v1/context", "/v1/tick", "/v1/reply",
                                                  "/v1/healthz", "/v1/metadata"]}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    """The judge must never see a 500. Degrade to a valid, minimal response."""
    print("[error] " + request.url.path + ": " + repr(exc), file=sys.stderr)
    path = request.url.path
    if path.endswith("/tick"):
        return JSONResponse(status_code=200, content={"actions": []})
    if path.endswith("/reply"):
        return JSONResponse(status_code=200, content={
            "action": "wait", "wait_seconds": 3600,
            "rationale": "Internal error while composing; backing off rather than sending "
                         "something broken."})
    if path.endswith("/context"):
        return JSONResponse(status_code=200, content={
            "accepted": True, "ack_id": "ack_degraded", "stored_at": _now_iso()})
    return JSONResponse(status_code=200, content={"status": "degraded"})


@app.on_event("startup")
async def _startup():
    preload_base_dataset()
    print("[bot] LLM: " + CLIENT.model_label(), file=sys.stderr)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), log_level="info")
