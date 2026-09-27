"""Pre-compose (and cache) the messages the judge harness will ask for.

Why: the Groq free tier allows ~8000 tokens/minute, and a judge run needs
tokens for BOTH our compositions and the judge's own scoring model. The
composition cache is keyed by a hash of the exact prompt and persisted to
.llm_cache.json, so composing ahead of time - at whatever pace the rate limit
allows - means the scored run itself spends no composition quota and answers
every tick instantly.

    python prewarm.py            # the 25 seed triggers judge_simulator pushes
    python prewarm.py --all      # every trigger in dataset/expanded (100)

Run this, then restart the bot (it loads the cache at startup), then run the
judge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import composer
from llm import CLIENT

HERE = Path(__file__).parent
DATA = HERE / "dataset"
EXPANDED = DATA / "expanded"


def load_seeds():
    categories = {}
    for path in sorted((DATA / "categories").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        categories[data.get("slug", path.stem)] = data
    merchants, customers, triggers = {}, {}, {}
    for name, container, key, store in (
            ("merchants_seed.json", "merchants", "merchant_id", merchants),
            ("customers_seed.json", "customers", "customer_id", customers),
            ("triggers_seed.json", "triggers", "id", triggers)):
        for item in json.loads((DATA / name).read_text(encoding="utf-8"))[container]:
            store[item[key]] = item
    return categories, merchants, customers, triggers


def load_expanded():
    def folder(name, key):
        out = {}
        for path in sorted((EXPANDED / name).glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            out[data.get(key, path.stem)] = data
        return out
    return (folder("categories", "slug"), folder("merchants", "merchant_id"),
            folder("customers", "customer_id"), folder("triggers", "id"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every expanded trigger, not just the seeds")
    args = ap.parse_args()

    if args.all and (EXPANDED / "triggers").is_dir():
        categories, merchants, customers, triggers = load_expanded()
    else:
        categories, merchants, customers, triggers = load_seeds()
        if EXPANDED.is_dir():  # customers the judge never pushes still need resolving
            _, _, exp_customers, _ = load_expanded()
            for cid, cust in exp_customers.items():
                customers.setdefault(cid, cust)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    print("prewarming " + str(len(triggers)) + " triggers at now=" + now_iso)
    print("llm: " + CLIENT.model_label())
    started = time.time()
    hits = fresh = skipped = 0

    for i, (tid, trigger) in enumerate(sorted(triggers.items()), 1):
        merchant = merchants.get(trigger.get("merchant_id"))
        if not merchant:
            skipped += 1
            continue
        category = categories.get(merchant.get("category_slug"))
        if not category:
            skipped += 1
            continue
        customer = customers.get(trigger.get("customer_id")) if trigger.get("customer_id") else None
        before = CLIENT.cache_hits
        result = composer.compose_message(category, merchant, trigger, customer,
                                          now_iso=now_iso,
                                          deadline=time.monotonic() + 300.0,
                                          cache_salt="prewarm")
        if CLIENT.cache_hits > before:
            hits += 1
        elif result["composed_by"].startswith("llm"):
            fresh += 1
        print("  [%2d/%2d] %-46s %-16s %s" % (i, len(triggers), tid[:46],
                                              result["composed_by"], result["body"][:60]))

    print("done in %.0fs — fresh: %d, cache hits: %d, skipped: %d" %
          (time.time() - started, fresh, hits, skipped))
    print("stats: " + json.dumps(CLIENT.stats()))
    print("Restart the bot so it loads the cache, then run: python run_judge.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
