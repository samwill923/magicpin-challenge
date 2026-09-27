"""LLM access layer: Groq primary, Gemini backup.

Design rules that matter for the challenge harness:
  * temperature 0 everywhere (submissions must be deterministic)
  * a hard wall-clock deadline per call, because /v1/tick and /v1/reply are
    scored on a 15-30s budget and a late answer is worth nothing
  * retry with fixed backoff on 429/5xx, then fail over to the backup provider
  * never raise to the caller: return None and let the caller fall back to its
    deterministic template
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Fixed (non-random) backoff so behaviour stays reproducible.
BACKOFF_SECONDS = (0.8, 2.0, 4.0)


def _load_dotenv() -> None:
    """Read KEY=value lines from ./.env into the environment (no overwrite).

    Keeps API keys out of the shell history and out of the repo (.env is
    gitignored); the process env still wins if it is already set.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and val and key not in os.environ:
                    os.environ[key] = val
    except OSError as exc:
        _log("could not read .env: " + repr(exc))


_load_dotenv()


def _env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return ""


class Provider:
    """One upstream chat endpoint."""

    def __init__(self, name: str, model: str, api_key: str):
        self.name = name
        self.model = model
        self.api_key = api_key
        self.last_tokens = 0

    def ready(self) -> bool:
        return bool(self.api_key)

    def _post(self, url: str, body: dict, headers: dict, timeout: float) -> dict:
        # An explicit User-Agent is required: Groq sits behind Cloudflare, which
        # rejects the default "Python-urllib/x.y" signature with 403 code 1010.
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "User-Agent": "vera-next-bot/1.0",
                     "Accept": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def complete(self, system: str, user: str, timeout: float, max_tokens: int) -> str:
        raise NotImplementedError


class GroqProvider(Provider):
    def __init__(self, model: str = "", api_key: str = ""):
        super().__init__("groq", model or _env("GROQ_MODEL") or DEFAULT_GROQ_MODEL,
                         api_key or _env("GROQ_API_KEY", "LLM_API_KEY"))

    def complete(self, system: str, user: str, timeout: float, max_tokens: int) -> str:
        data = self._post(
            GROQ_URL,
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": 0,
                "top_p": 1,
                "max_completion_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
            {"Authorization": "Bearer " + self.api_key},
            timeout,
        )
        self.last_tokens = int((data.get("usage") or {}).get("total_tokens") or 0)
        return data["choices"][0]["message"]["content"] or ""


class GeminiProvider(Provider):
    def __init__(self, model: str = "", api_key: str = ""):
        super().__init__("gemini", model or _env("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
                         api_key or _env("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY"))

    def complete(self, system: str, user: str, timeout: float, max_tokens: int) -> str:
        data = self._post(
            GEMINI_URL.format(model=self.model),
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": 0,
                    "topP": 1,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json",
                },
            },
            {"x-goog-api-key": self.api_key},
            timeout,
        )
        self.last_tokens = int((data.get("usageMetadata") or {}).get("totalTokenCount") or 0)
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


def _build_chain() -> list:
    """Primary first, backup second. LLM_PROVIDER picks which is primary."""
    primary = (_env("LLM_PROVIDER") or "groq").lower()
    model = _env("LLM_MODEL")
    groq = GroqProvider(model if primary == "groq" else "")
    gemini = GeminiProvider(model if primary == "gemini" else "")
    chain = [gemini, groq] if primary == "gemini" else [groq, gemini]
    return [p for p in chain if p.ready()]


class RateGovernor:
    """Self-pace against the provider's tokens-per-minute limit.

    Groq's free tier allows 8000 TPM on openai/gpt-oss-120b, which is about
    four full compositions per minute. Waiting for a slot beats firing the call
    and eating a 429, so we keep a 60-second rolling window of spent tokens and
    only proceed when there is room before the caller's deadline.
    """

    def __init__(self, tpm: int, rpm: int):
        self.tpm = tpm
        self.rpm = rpm
        self._spent = []  # (monotonic_ts, tokens)
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        self._spent = [(ts, tok) for ts, tok in self._spent if now - ts < 60.0]

    def wait_for_slot(self, est_tokens: int, deadline: float) -> bool:
        """True if a slot was secured (possibly after sleeping) before the deadline."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                used = sum(tok for _, tok in self._spent)
                if (used + est_tokens <= self.tpm * 0.92 and
                        len(self._spent) < self.rpm):
                    self._spent.append((now, est_tokens))
                    return True
                oldest = min(ts for ts, _ in self._spent) if self._spent else now
            sleep_for = max(0.5, 60.0 - (time.monotonic() - oldest) + 0.5)
            if time.monotonic() + sleep_for > deadline - 1.0:
                return False
            _log("rate governor: waiting %.1fs for token budget" % sleep_for)
            time.sleep(min(sleep_for, 20.0))

    def settle(self, est_tokens: int, actual_tokens: int) -> None:
        """Replace the estimate with what the call actually cost."""
        with self._lock:
            for i in range(len(self._spent) - 1, -1, -1):
                if self._spent[i][1] == est_tokens:
                    self._spent[i] = (self._spent[i][0], actual_tokens)
                    return


class LLMClient:
    """Thread-safe, cached, deadline-aware, self-throttling front end."""

    def __init__(self):
        self.chain = _build_chain()
        self._cache = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0
        self.throttled = 0
        self.governor = RateGovernor(int(_env("LLM_TPM") or 8000),
                                     int(_env("LLM_RPM") or 28))
        self._cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       ".llm_cache.json")
        self._load_cache()

    # -- disk cache ------------------------------------------------------
    def _load_cache(self) -> None:
        """Compositions are keyed by a hash of the exact prompt, so a prompt or
        context change invalidates them automatically. Persisting them means a
        pre-warmed run costs no quota during the scored test."""
        if _env("LLM_CACHE") == "0":
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                self._cache.update({k: v for k, v in data.items() if isinstance(v, str)})
                _log("loaded " + str(len(self._cache)) + " cached compositions from disk")
        except (OSError, ValueError):
            pass

    def _save_cache(self) -> None:
        if _env("LLM_CACHE") == "0":
            return
        try:
            tmp = self._cache_path + ".tmp"
            with self._lock:
                snapshot = dict(self._cache)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(snapshot, fh, ensure_ascii=False)
            os.replace(tmp, self._cache_path)
        except OSError as exc:
            _log("could not write cache: " + repr(exc))

    @property
    def enabled(self) -> bool:
        return bool(self.chain)

    def model_label(self) -> str:
        if not self.chain:
            return "deterministic-template-fallback (no API key configured)"
        return " -> ".join(p.name + ":" + p.model for p in self.chain)

    def complete_json(self, system: str, user: str, cache_key: str,
                      deadline: float, max_tokens: int = 2600):
        """Return the parsed JSON object, or None if every attempt failed.

        `deadline` is an absolute time.monotonic() value we never exceed.
        The cache key is a hash of the exact prompt (plus `cache_key` as a
        readable label), so identical requests are free and any change to the
        prompt or the underlying context misses the cache instead of serving a
        stale composition.
        """
        key = _prompt_hash(self.model_label(), system, user)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            return _parse_json(hit)

        raw = self._complete_raw(system, user, deadline, max_tokens)
        if raw is None:
            return None
        parsed = _parse_json(raw)
        if parsed is None:
            return None
        with self._lock:
            self._cache[key] = raw
        self._save_cache()
        return parsed

    def _complete_raw(self, system: str, user: str, deadline: float, max_tokens: int):
        for provider in self.chain:
            budget = max_tokens
            for attempt in range(len(BACKOFF_SECONDS) + 1):
                remaining = deadline - time.monotonic()
                if remaining < 1.5:
                    return None
                est = len(system) // 4 + len(user) // 4 + min(budget, 900)
                if not self.governor.wait_for_slot(est, deadline):
                    self.throttled += 1
                    _log("no token budget before deadline; using fallback")
                    return None
                try:
                    self.calls += 1
                    out = provider.complete(system, user, min(remaining, 25.0), budget)
                    if provider.last_tokens:
                        self.governor.settle(est, provider.last_tokens)
                    if out and out.strip():
                        return out
                    raise ValueError("empty completion")
                except urllib.error.HTTPError as exc:
                    detail = _peek(exc)
                    # A reasoning model can burn the whole budget before emitting
                    # the JSON; Groq reports that as a 400 json_validate_failed.
                    starved = exc.code == 400 and "json_validate_failed" in detail
                    retryable = exc.code == 429 or exc.code >= 500 or starved
                    if starved:
                        budget = min(budget * 2, 8000)
                    _log(provider.name + " HTTP " + str(exc.code) +
                         (" (retryable)" if retryable else "") + ": " + detail)
                    if not retryable:
                        break  # bad key / bad request: move to the next provider
                except Exception as exc:  # timeout, socket error, bad payload
                    _log(provider.name + " error: " + repr(exc))
                if attempt < len(BACKOFF_SECONDS):
                    sleep_for = BACKOFF_SECONDS[attempt]
                    if time.monotonic() + sleep_for > deadline - 1.5:
                        break
                    time.sleep(sleep_for)
        self.failures += 1
        return None

    def stats(self) -> dict:
        return {"llm_calls": self.calls, "cache_hits": self.cache_hits,
                "llm_failures": self.failures, "llm_throttled": self.throttled,
                "cached_compositions": len(self._cache)}


def _prompt_hash(model: str, system: str, user: str) -> str:
    return hashlib.sha1((model + "|" + system + "|" + user).encode("utf-8")).hexdigest()


def _parse_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _peek(exc) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:200]
    except Exception:
        return ""


def _log(msg: str) -> None:
    print("[llm] " + msg, file=sys.stderr, flush=True)


CLIENT = LLMClient()
