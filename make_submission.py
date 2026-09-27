"""Generate submission.jsonl for the 30 canonical (merchant, trigger) test pairs.

The canonical pairs come from the challenge's own deterministic generator
(dataset/generate_dataset.py -> dataset/expanded/test_pairs.json, fixed seed
20260426), so this is the same set every participant is scored on.

Usage:
    python make_submission.py                  # all 30 pairs
    python make_submission.py --only T05,T21   # iterate on specific pairs
    python make_submission.py --workers 4      # parallel LLM calls

Writes:
    submission.jsonl        the deliverable (test_id, body, cta, send_as,
                            suppression_key, rationale)
    submission_debug.jsonl  dev-only: which pipeline produced each line, word
                            count, and any validation problems left over
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import composer

HERE = Path(__file__).parent
DATA = HERE / "dataset"
EXPANDED = DATA / "expanded"
# The dataset's own "today": keeps seasonal beats and date phrasing consistent.
ASOF = "2026-04-26T10:00:00Z"


def ensure_expanded() -> None:
    if (EXPANDED / "test_pairs.json").exists():
        return
    print("[dataset] expanding seeds with the challenge generator...")
    env = dict(os.environ, PYTHONUTF8="1")
    subprocess.run([sys.executable, str(DATA / "generate_dataset.py"),
                    "--seed-dir", str(DATA), "--out", str(EXPANDED)],
                   check=True, env=env)


def load(folder: str, key: str) -> dict:
    out = {}
    for path in sorted((EXPANDED / folder).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[data.get(key, path.stem)] = data
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated test_ids")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "submission.jsonl"))
    args = ap.parse_args()

    ensure_expanded()
    categories = load("categories", "slug")
    merchants = load("merchants", "merchant_id")
    customers = load("customers", "customer_id")
    triggers = load("triggers", "id")
    pairs = json.loads((EXPANDED / "test_pairs.json").read_text(encoding="utf-8"))["pairs"]
    wanted = {t.strip() for t in args.only.split(",") if t.strip()}
    if wanted:
        pairs = [p for p in pairs if p["test_id"] in wanted]

    print("[compose] " + str(len(pairs)) + " pairs, workers=" + str(args.workers))
    started = time.time()

    def work(pair: dict) -> dict:
        merchant = merchants.get(pair["merchant_id"], {})
        trigger = triggers.get(pair["trigger_id"], {})
        customer = customers.get(pair.get("customer_id")) if pair.get("customer_id") else None
        category = categories.get(merchant.get("category_slug"), {})
        result = composer.compose_message(category, merchant, trigger, customer,
                                          now_iso=ASOF, deadline=time.monotonic() + 300.0,
                                          cache_salt="submission")
        result["test_id"] = pair["test_id"]
        result["_merchant_id"] = pair["merchant_id"]
        result["_trigger_id"] = pair["trigger_id"]
        result["_customer_id"] = pair.get("customer_id")
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(work, pairs))
    results.sort(key=lambda r: r["test_id"])

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps({
                "test_id": r["test_id"],
                "body": r["body"],
                "cta": r["cta"],
                "send_as": r["send_as"],
                "suppression_key": r["suppression_key"],
                "rationale": r["rationale"],
            }, ensure_ascii=False) + "\n")

    debug_path = HERE / "submission_debug.jsonl"
    with debug_path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in results:
            fh.write(json.dumps({
                "test_id": r["test_id"], "merchant_id": r["_merchant_id"],
                "trigger_id": r["_trigger_id"], "customer_id": r["_customer_id"],
                "trigger_kind": r["trigger_kind"], "send_as": r["send_as"],
                "composed_by": r["composed_by"], "cta": r["cta"],
                "words": len(r["body"].split()), "chars": len(r["body"]),
                "body": r["body"],
            }, ensure_ascii=False) + "\n")

    by_source: dict[str, int] = {}
    for r in results:
        by_source[r["composed_by"]] = by_source.get(r["composed_by"], 0) + 1
    print("[done] " + str(len(results)) + " lines -> " + str(out_path) +
          "  (" + ", ".join(k + ": " + str(v) for k, v in sorted(by_source.items())) + ")")
    print("[time] " + ("%.1f" % (time.time() - started)) + "s")
    if by_source.get("template"):
        print("[warn] " + str(by_source["template"]) + " line(s) came from the deterministic "
              "fallback - set GROQ_API_KEY for full quality")
    return 0


if __name__ == "__main__":
    sys.exit(main())
