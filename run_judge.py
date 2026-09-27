"""Run magicpin's judge_simulator.py without editing it.

judge_simulator.py keeps its configuration in module-level constants; this
wrapper fills them from the environment (or .env) and calls its main().

It also swaps in a hardened provider for the judge's own scoring model, because
the stock one cannot reach Groq from here: Cloudflare rejects requests with the
default "Python-urllib" User-Agent (403 code 1010), and a 429 raises straight
through, which would silently turn every score into the heuristic fallback.
The replacement adds a User-Agent, paces itself against the tokens-per-minute
limit and retries on 429. Scoring is otherwise untouched: same prompt, same
model output, same parsing.

    python run_judge.py                      # scenario: all
    python run_judge.py phase2_short
    python run_judge.py full_evaluation

Reads: GROQ_API_KEY (or LLM_API_KEY / GEMINI_API_KEY), JUDGE_PROVIDER,
JUDGE_MODEL, BOT_URL.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error

import llm  # loads .env

import judge_simulator as js


class PacedJudgeProvider:
    """Duck-typed replacement for judge_simulator's LLMProvider."""

    def __init__(self, model: str, api_key: str, url: str, headers_fn, tpm: int):
        self.model = model
        self.api_key = api_key
        self.url = url
        self.headers_fn = headers_fn
        self.governor = llm.RateGovernor(tpm, 28)
        self._http = llm.Provider("judge", model, api_key)

    def name(self) -> str:
        return "groq(paced):" + self.model

    def complete(self, prompt: str, system: str = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages, "temperature": 0.2,
                "max_completion_tokens": 2500}
        est = (len(prompt) + len(system or "")) // 4 + 900
        for attempt in range(4):
            self.governor.wait_for_slot(est, time.monotonic() + 240)
            try:
                data = self._http._post(self.url, body, self.headers_fn(self.api_key), 60)
                usage = (data.get("usage") or {}).get("total_tokens") or 0
                if usage:
                    self.governor.settle(est, int(usage))
                return data["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:160]
                except Exception:
                    pass
                print("  [judge-llm] HTTP " + str(exc.code) + " " + detail)
                if exc.code == 429 and attempt < 3:
                    time.sleep(25)
                    continue
                raise
        raise RuntimeError("judge llm unavailable")


def main() -> int:
    scenario = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TEST_SCENARIO", "all")

    provider = (os.environ.get("JUDGE_PROVIDER") or "groq").lower()
    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY") or os.environ.get("LLM_API_KEY", "")
        model = os.environ.get("JUDGE_MODEL", "openai/gpt-oss-20b")
    elif provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY", "")
        model = os.environ.get("JUDGE_MODEL", "gemini-2.5-flash")
    else:
        key = os.environ.get("LLM_API_KEY", "")
        model = os.environ.get("JUDGE_MODEL", "")

    if not key:
        print("No judge API key found. Put GROQ_API_KEY=... in .env or the environment.")
        return 2

    js.BOT_URL = os.environ.get("BOT_URL", "http://127.0.0.1:8080")
    js.LLM_PROVIDER = provider
    js.LLM_API_KEY = key
    js.LLM_MODEL = model
    js.TEST_SCENARIO = scenario

    if provider == "groq":
        js.create_provider = lambda: PacedJudgeProvider(
            model, key, llm.GROQ_URL,
            lambda k: {"Authorization": "Bearer " + k},
            int(os.environ.get("JUDGE_TPM", "8000")))
    print("judge provider=" + provider + " model=" + model + " bot=" + js.BOT_URL +
          " scenario=" + scenario)
    try:
        js.main()
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
