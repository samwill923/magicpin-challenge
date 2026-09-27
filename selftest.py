"""Key-free contract test for the Vera bot.

Exercises the same call sequence as judge_simulator.py (warmup, versioned
context pushes, ticks, and the three replay scenarios) and checks the response
schemas, without needing an LLM key for the judge side.

    python selftest.py            # against http://localhost:8080
    BOT_URL=... python selftest.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BOT = os.environ.get("BOT_URL", "http://127.0.0.1:8080").rstrip("/")
DATA = Path(__file__).parent / "dataset"
FAILS: list[str] = []
URL_RE = re.compile(r"(https?://|www\.)", re.I)


def call(method: str, path: str, body=None, timeout=30):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BOT + path, data=data, method=method,
                                headers={"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), time.time() - start
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw), time.time() - start
        except json.JSONDecodeError:
            return exc.code, {"raw": raw[:200]}, time.time() - start


def check(cond: bool, label: str, detail: str = "") -> bool:
    print(("  PASS  " if cond else "  FAIL  ") + label + (("  -- " + detail) if detail and not cond else ""))
    if not cond:
        FAILS.append(label)
    return cond


def seed(name: str, container: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))[container]


def main() -> int:
    print("== warmup ==")
    code, health, _ = call("GET", "/v1/healthz", timeout=10)
    check(code == 200 and health.get("status") == "ok", "healthz returns ok", str(health))
    check(set(health.get("contexts_loaded", {})) == {"category", "merchant", "customer", "trigger"},
          "healthz.contexts_loaded has all four scopes", str(health.get("contexts_loaded")))
    check(all(v == 0 for v in health.get("contexts_loaded", {}).values()),
          "contexts_loaded is zero before any push", str(health.get("contexts_loaded")))

    code, meta, _ = call("GET", "/v1/metadata", timeout=10)
    need = {"team_name", "team_members", "model", "approach", "contact_email", "version",
            "submitted_at"}
    check(code == 200 and need <= set(meta), "metadata has every required field",
          str(sorted(need - set(meta))))
    print("  model: " + str(meta.get("model")))

    print("== context pushes ==")
    cats = {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((DATA / "categories").glob("*.json"))}
    for slug, payload in cats.items():
        code, resp, _ = call("POST", "/v1/context",
                             {"scope": "category", "context_id": slug, "version": 1,
                              "payload": payload, "delivered_at": "2026-04-26T09:45:00Z"})
        check(code == 200 and resp.get("accepted") is True and resp.get("ack_id"),
              "category/" + slug + " accepted", str(resp))

    merchants = seed("merchants_seed.json", "merchants")
    customers = seed("customers_seed.json", "customers")
    triggers = seed("triggers_seed.json", "triggers")
    for m in merchants:
        code, resp, _ = call("POST", "/v1/context", {"scope": "merchant",
                                                     "context_id": m["merchant_id"],
                                                     "version": 1, "payload": m,
                                                     "delivered_at": "2026-04-26T09:45:00Z"})
        if not (code == 200 and resp.get("accepted")):
            check(False, "merchant push " + m["merchant_id"], str(resp))
    print("  pushed " + str(len(merchants)) + " merchants")
    for c in customers:
        call("POST", "/v1/context", {"scope": "customer", "context_id": c["customer_id"],
                                     "version": 1, "payload": c})
    print("  pushed " + str(len(customers)) + " customers")

    m0 = merchants[0]
    code, resp, _ = call("POST", "/v1/context", {"scope": "merchant",
                                                 "context_id": m0["merchant_id"],
                                                 "version": 1, "payload": m0})
    check(code == 409 and resp.get("reason") == "stale_version" and
          resp.get("current_version") == 1, "re-pushing version 1 gives 409 stale_version",
          str(code) + " " + str(resp))

    bumped = json.loads(json.dumps(m0))
    bumped["performance"]["views"] = 2580
    code, resp, _ = call("POST", "/v1/context", {"scope": "merchant",
                                                 "context_id": m0["merchant_id"],
                                                 "version": 2, "payload": bumped})
    check(code == 200 and resp.get("accepted") is True, "version 2 replaces version 1", str(resp))

    code, resp, _ = call("POST", "/v1/context", {"scope": "nonsense", "context_id": "x",
                                                 "version": 1, "payload": {"a": 1}})
    check(code == 400 and resp.get("reason") == "invalid_scope", "bad scope gives 400", str(resp))

    code, health, _ = call("GET", "/v1/healthz", timeout=10)
    loaded = health.get("contexts_loaded", {})
    check(loaded.get("category") == len(cats) and loaded.get("merchant") == len(merchants) and
          loaded.get("customer") == len(customers) and loaded.get("trigger") == 0,
          "contexts_loaded matches what was pushed", str(loaded))

    print("== tick ==")
    for t in triggers:
        call("POST", "/v1/context", {"scope": "trigger", "context_id": t["id"], "version": 1,
                                     "payload": t})
    ids = [t["id"] for t in triggers]
    all_actions = []
    for i in range(0, len(ids), 5):
        batch = ids[i:i + 5]
        code, resp, secs = call("POST", "/v1/tick",
                                {"now": "2026-04-26T10:35:00Z", "available_triggers": batch})
        actions = resp.get("actions", []) if code == 200 else []
        check(code == 200 and isinstance(actions, list),
              "tick batch " + str(i // 5 + 1) + " returns actions[]", str(resp)[:200])
        check(secs < 15.0, "tick batch " + str(i // 5 + 1) + " under 15s",
              ("%.1f" % secs) + "s")
        print("    -> " + str(len(actions)) + " action(s) in " + ("%.1f" % secs) + "s")
        all_actions += actions

    required = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id",
                "template_name", "template_params", "body", "cta", "suppression_key", "rationale"}
    seen_bodies, seen_convs = set(), set()
    for a in all_actions:
        missing = sorted(required - set(a))
        check(not missing, "action has all required fields (" + str(a.get("trigger_id")) + ")",
              str(missing))
        check(bool((a.get("body") or "").strip()), "action body non-empty", str(a.get("trigger_id")))
        check(not URL_RE.search(a.get("body", "")), "action body has no URL",
              a.get("body", "")[:80])
        check(a.get("send_as") in ("vera", "merchant_on_behalf"), "send_as valid",
              str(a.get("send_as")))
        check(a.get("body") not in seen_bodies, "no verbatim repeat across actions",
              a.get("body", "")[:60])
        check(a.get("conversation_id") not in seen_convs, "conversation_id unique per action",
              str(a.get("conversation_id")))
        seen_bodies.add(a.get("body"))
        seen_convs.add(a.get("conversation_id"))
    print("  total actions: " + str(len(all_actions)))

    code, resp, _ = call("POST", "/v1/tick", {"now": "2026-04-26T10:40:00Z",
                                              "available_triggers": []})
    check(code == 200 and isinstance(resp.get("actions"), list),
          "tick with no triggers still returns actions[]", str(resp)[:120])

    print("== replay: auto-reply hell ==")
    auto = "Thank you for contacting us! Our team will respond shortly."
    mid = merchants[0]["merchant_id"]
    seq = []
    for i in range(1, 5):
        code, resp, secs = call("POST", "/v1/reply",
                                {"conversation_id": "conv_auto_" + str(i), "merchant_id": mid,
                                 "customer_id": None, "from_role": "merchant", "message": auto,
                                 "received_at": "2026-04-26T10:4" + str(i) + ":00Z",
                                 "turn_number": i + 1})
        seq.append(resp.get("action"))
        print("    turn " + str(i) + ": " + str(resp.get("action")) + "  " +
              str(resp.get("wait_seconds") or (resp.get("body") or "")[:60]))
        check(secs < 15.0, "auto-reply turn " + str(i) + " under 15s", ("%.1f" % secs) + "s")
    check(seq[0] == "send" and seq[1] == "wait" and "end" in seq[2:],
          "auto-reply: nudge once, back off, then end", str(seq))

    print("== replay: intent transition ==")
    code, resp, _ = call("POST", "/v1/reply",
                         {"conversation_id": "conv_intent_1", "merchant_id": mid,
                          "from_role": "merchant", "message": "Ok lets do it. Whats next?",
                          "received_at": "2026-04-26T10:50:00Z", "turn_number": 2})
    body = (resp.get("body") or "")
    print("    " + str(resp.get("action")) + ": " + body[:150])
    qualifying = ["would you", "do you", "can you tell", "what if", "how about"]
    actioning = ["done", "sending", "draft", "here", "confirm", "proceed", "next", "ready"]
    low = body.lower()
    check(resp.get("action") == "send", "commitment -> send", str(resp.get("action")))
    check(any(w in low for w in actioning), "reply is in action mode", body[:100])
    check(not any(w in low for w in qualifying), "reply asks no qualifying question", body[:100])

    print("== replay: hostile / off-topic ==")
    code, resp, _ = call("POST", "/v1/reply",
                         {"conversation_id": "conv_offtopic", "merchant_id": mid,
                          "from_role": "merchant",
                          "message": "Btw can you also help me with my GST filing this month?",
                          "received_at": "2026-04-26T10:52:00Z", "turn_number": 2})
    print("    " + str(resp.get("action")) + ": " + (resp.get("body") or "")[:150])
    check(resp.get("action") == "send", "off-topic -> send (stays on mission)", str(resp))
    check("gst" not in (resp.get("body") or "").lower() or
          re.search(r"(ca\b|accountant|outside|can'?t help|leave that)",
                    (resp.get("body") or "").lower()) is not None,
          "off-topic reply declines politely", (resp.get("body") or "")[:120])

    code, resp, _ = call("POST", "/v1/reply",
                         {"conversation_id": "conv_hostile", "merchant_id": mid,
                          "from_role": "merchant",
                          "message": "Stop messaging me. This is useless spam.",
                          "received_at": "2026-04-26T10:55:00Z", "turn_number": 2})
    print("    " + str(resp.get("action")) + ": " + str(resp.get("rationale"))[:120])
    check(resp.get("action") == "end", "hostile -> end", str(resp))

    code, resp, _ = call("POST", "/v1/reply",
                         {"conversation_id": "conv_after_optout", "merchant_id": mid,
                          "from_role": "merchant", "message": "hello?",
                          "received_at": "2026-04-26T11:00:00Z", "turn_number": 2})
    check(resp.get("action") == "end", "opted-out merchant stays closed on a new thread", str(resp))

    print("== summary ==")
    if FAILS:
        print("  " + str(len(FAILS)) + " check(s) failed:")
        for f in dict.fromkeys(FAILS):
            print("    - " + f)
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
