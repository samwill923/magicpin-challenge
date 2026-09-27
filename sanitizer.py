"""Outgoing-message sanitiser.

Applied to every live send and to submission.jsonl alike, so what the judge
reads is what the validator saw.

Two jobs:
  1. No dashes of any kind. Em and en dashes become commas, compound words are
     opened up ("click-through" -> "click through"), ranges become "to". ISO
     dates are protected, so 2026-12-15 survives intact.
  2. No Hindi phrase bolted onto an English message: either the whole message
     is Hinglish or none of it is.
"""
from __future__ import annotations

import re

DAYS = "Mon|Tue|Wed|Thu|Fri|Sat|Sun"
MONS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
DATE_SLOT = "\u0001"

# Hindi/Hinglish markers, used to tell a genuinely Hinglish message from an
# English one with a Hindi phrase bolted on the end.
HINGLISH_MARKER = re.compile(
    r"\b(aap|aapka|aapke|aapko|apka|apke|kya|hai|hain|mein|karein|kripya|yahan|"
    r"apna|humare|hum|nahi|liye|batao|chalega|theek|namaste|bhej|dun|doon|karu|"
    r"karun|sirf|abhi|hafte|baar|yaad|khatam|zaroori|dhyan|acha)\b", re.I)

# A closing Hindi ask tacked onto an otherwise English message.
TACKED_HINGLISH = re.compile(
    r"[,;:]?\s*(bhej\s*d(?:u|oo)n|chalega|batao|bata\s*d(?:i|ee)jiye|"
    r"kar\s*d(?:u|oo)n|karu[n]?)\s*\??\s*$", re.I)

INTERROGATIVE = re.compile(
    r"^(shall|want|would|can|could|should|do|does|may|which|what|how|is|are)\b", re.I)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

EXOTIC_HYPHENS = ("‐", "‑", "‒", "−")
COMMA_DASHES = ("—", "–")


def sanitize_body(text: str) -> str:
    if not text:
        return text
    t = text
    for exotic in EXOTIC_HYPHENS:
        t = t.replace(exotic, "-")
    for dash in COMMA_DASHES:
        t = t.replace(dash, ", ")
    # protect ISO dates before any hyphen rewriting
    t = re.sub(r"(\d{4})-(\d{2})-(\d{2})", lambda m: DATE_SLOT.join(m.groups()), t)
    t = re.sub(r"\b(" + DAYS + r")\s*-\s*(" + DAYS + r")\b", r"\g<1> to \g<2>", t)
    t = re.sub(r"\b(" + MONS + r")\s*-\s*(" + MONS + r")\b", r"\g<1> to \g<2>", t)
    t = re.sub(r"(\d(?:\.\d+)?)\s*-\s*(\d(?:\.\d+)?)", r"\g<1> to \g<2>", t)
    t = re.sub(r"(?<=[A-Za-z0-9])-(?=[A-Za-z])", " ", t)     # click-through
    t = re.sub(r"(?<=[A-Za-z])-(?=[0-9])", " ", t)
    t = re.sub(r"\s+-\s+", ", ", t)
    t = t.replace("-", " ")
    t = t.replace(DATE_SLOT, "-")

    t = drop_tacked_hinglish(t)

    t = re.sub(r"(,\s*){2,}", ", ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\g<1>", t)
    t = re.sub(r",\s*([.?!])", r"\g<1>", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def drop_tacked_hinglish(t: str) -> str:
    """Strip a trailing Hindi ask from a message that is otherwise English."""
    markers = {m.group(0).lower() for m in HINGLISH_MARKER.finditer(t)}
    if len(markers) >= 3:
        return t                      # a genuinely Hinglish message: leave it
    stripped = TACKED_HINGLISH.sub("", t).rstrip(" ,;:")
    if stripped == t.rstrip():
        return t
    parts = [p for p in SENTENCE_SPLIT.split(stripped) if p.strip()]
    tail = parts[-1] if parts else stripped
    if not stripped.endswith(("?", ".", "!")):
        stripped += "?" if INTERROGATIVE.match(tail.strip()) else "."
    if "?" not in stripped:
        stripped += " Shall I send it over?"
    return stripped


def has_dash(text: str) -> bool:
    """True if any dash survived (guards the live path after sanitising)."""
    if any(ch in text for ch in EXOTIC_HYPHENS + COMMA_DASHES):
        return True
    return bool(re.search(r"(?<=[A-Za-z])-(?=[A-Za-z])", text or ""))
