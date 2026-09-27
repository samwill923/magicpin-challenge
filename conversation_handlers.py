"""Multi-turn reply handling for /v1/reply.

Classification is deterministic regex work (fast, predictable, never times out);
the LLM is used only to write the body once the route is decided. Every branch
has a deterministic fallback, so /v1/reply always answers inside its budget.

Routes:
  opt_out / hostile      -> end
  auto-reply (canned or repeated verbatim)
                         -> nudge once, then wait, then end
  explicit commitment    -> ACTION mode immediately (never another qualifying question)
  question               -> answer from context, or decline off-topic and redirect
  defer ("call later")   -> wait
  no substance           -> at most 3 nudges, then end
"""
from __future__ import annotations

import re
import time

import composer
from llm import CLIENT
from sanitizer import sanitize_body

# --------------------------------------------------------------------------
# Signal detection
# --------------------------------------------------------------------------

OPT_OUT = re.compile(
    r"(not interested|no interest|don'?t (message|contact|call|send)|do not (message|contact|send)|"
    r"stop (messaging|sending|contacting|it|this)|stop\b.*\b(message|msg|spam)|unsubscribe|"
    r"remove (me|my number)|leave me alone|band karo|band kar do|mat bhejo|mat karo|"
    r"no thanks|not required|nahi chahiye|koi zarurat nahi)", re.I)

HOSTILE = re.compile(
    r"(useless|nonsense|rubbish|bekar|bakwas|spam|why are you bothering|stop bothering|"
    r"waste of time|irritating|pareshan|fed up|idiot|stupid|nalayak|bloody|shut up)", re.I)

AUTO_REPLY_MARKERS = (
    "thank you for contacting", "thanks for contacting", "thank you for reaching out",
    "thank you for your message", "we have received your message", "will get back to you",
    "will respond shortly", "our team will", "team will respond", "someone will get back",
    "this is an automated", "automated reply", "auto-reply", "auto reply",
    "automated assistant", "away from my phone", "outside our business hours",
    "our business hours", "our office hours", "we are currently closed",
    "aapki jaankari ke liye", "team tak pahuncha", "jaankari ke liye dhanyavaad",
    "sampark karne ke liye dhanyavaad", "hamari team", "kindly wait for",
    "we appreciate your message", "your message is important",
)

COMMITMENT = re.compile(
    r"^\s*(yes|yess|yep|yeah|ya|yup|haan|han|ji|ok|okay|okey|k|sure|done|great|"
    r"theek hai|thik hai|thike|sahi hai|chalega|go ahead|go for it|let'?s do it|lets do it|"
    r"do it|please do|karo|kar do|kar dijiye|bhejo|bhej do|bhej dijiye|send it|send|"
    r"proceed|start|begin|confirm|confirmed|yes please|absolutely|definitely|deal)\b",
    re.I)
COMMITMENT_ANY = re.compile(
    r"(let'?s do it|lets do it|go ahead|please (do|send|draft|proceed)|send (it|me|the)|"
    r"i want to join|want to join|mujhe jud|judna hai|join karna|sign me up|count me in|"
    r"sounds good|i'?m in|im in|yes interested|haan bhejo|kar do|kar dijiye|"
    r"ok lets|ok let'?s|what'?s next|whats next|aage kya)", re.I)

QUESTION = re.compile(
    r"(\?|^(what|what'?s|how|how'?s|why|when|where|which|who|can you|could you|will you|do you|"
    r"does it|is it|are you|kya|kaise|kitna|kitne|kab|kaun|kahan|batao|bata)\b)", re.I)

OFF_TOPIC = re.compile(
    r"\b(gst|income tax|itr|tds|tax filing|ca\b|chartered accountant|loan|mudra|insurance|"
    r"policy premium|electricity bill|rent agreement|passport|visa|shop licence|shop license|"
    r"labour law|pf\b|esi\b|trademark|court case|police|recruitment|hire staff|hiring staff)\b",
    re.I)

DEFER = re.compile(
    r"(call me (later|tomorrow|after)|later|busy|in a meeting|meeting mein|"
    r"abhi nahi|baad me|baad mein|kal|tomorrow|next week|after \d|"
    r"not now|some other time|give me (a|some) (day|time|week))", re.I)

ACK_ONLY = re.compile(r"^\s*(ok|okay|k|hmm+|hm|thanks|thank you|dhanyavaad|shukriya|ji|"
                      r"noted|got it|\U0001F44D|✅)\s*[.!]*\s*$", re.I)

QUALIFYING_PHRASES = ("would you", "do you", "can you tell", "what if", "how about",
                      "could you tell", "may i ask", "just to understand")
ACTION_WORDS = ("done", "sending", "sent", "draft", "drafted", "drafting", "here", "here's",
                "confirm", "proceed", "next", "ready", "setting up", "scheduled", "queued",
                "i'll do", "i will")


def classify(message: str, state, merchant_autoreplies: set) -> str:
    text = (message or "").strip()
    if not text:
        return "no_substance"
    low = text.lower()
    norm = composer._normalize(text)

    if OPT_OUT.search(low):
        return "opt_out"
    if HOSTILE.search(low):
        return "hostile"
    if any(m in low for m in AUTO_REPLY_MARKERS):
        return "auto_reply"
    # same text again, in this conversation or anywhere from this merchant
    if norm and (norm in state.inbound_norms or norm in merchant_autoreplies):
        return "auto_reply"
    if len(text.split()) > 22 and re.search(r"(dear (customer|sir)|for any (query|queries)|"
                                            r"visit us|timings?:)", low):
        return "auto_reply"
    if COMMITMENT.match(low) or COMMITMENT_ANY.search(low):
        return "commitment"
    if QUESTION.search(low):
        return "off_topic" if OFF_TOPIC.search(low) else "question"
    if DEFER.search(low):
        return "defer"
    if ACK_ONLY.match(low):
        return "no_substance"
    return "engaged"


def defer_seconds(message: str) -> int:
    low = (message or "").lower()
    if re.search(r"next week", low):
        return 259200
    if re.search(r"(tomorrow|kal\b)", low):
        return 86400
    if re.search(r"(meeting|busy|abhi nahi|not now)", low):
        return 7200
    return 3600


# --------------------------------------------------------------------------
# Reply composition
# --------------------------------------------------------------------------

SYSTEM_REPLY = """You are Vera, magicpin's marketing assistant, replying to a merchant on WhatsApp mid-conversation. You already introduced yourself earlier in this thread.

HARD RULES:
1. Every number, date, price, name and source you write must appear in the FACTS block. Never invent anything.
2. No preamble, no re-introduction, no "thanks for your reply" filler. Answer or act in the first sentence.
3. Exactly ONE ask, and it is the last sentence. At most one question mark in the whole message.
4. Obey the category voice and never use a forbidden word or claim.
5. No URLs, no phone numbers, no email addresses.
6. 25-70 words. Shorter is better mid-conversation.
7. Never repeat a sentence you already sent in this thread.

OUTPUT: one JSON object only:
{"body": "<message>", "cta": "<open_ended|binary_yes_no|binary_confirm_cancel|multi_choice_slot|none>", "rationale": "<1-2 sentences>"}"""

SYSTEM_REPLY_CUSTOMER = """You are replying on WhatsApp on behalf of an Indian local business, from the business's own number, to one of its own customers. You are the business - never mention magicpin, Vera, an assistant or a bot.

HARD RULES:
1. Every number, date, price, slot and name must appear in the FACTS block. Never invent anything.
2. No preamble. Answer or confirm in the first sentence.
3. Exactly ONE ask, in the last sentence. At most one question mark.
4. Obey the category voice; no medical promises, no pressure, no guilt.
5. No URLs, no phone numbers, no email addresses.
6. 25-60 words, warm and clear.

OUTPUT: one JSON object only:
{"body": "<message>", "cta": "<open_ended|binary_yes_no|binary_confirm_cancel|multi_choice_slot|none>", "rationale": "<1-2 sentences>"}"""

BRANCH_RULES = {
    "commitment": (
        "The merchant has just committed ('yes / let's do it'). Switch to ACTION mode NOW. "
        "Do NOT ask any qualifying question. State what you are doing this minute (name the exact "
        "artefact and where it goes), scope it with a real number from FACTS, and close by asking "
        "only for the go-ahead to execute the final step (e.g. 'Reply CONFIRM and I'll publish it'). "
        "Never begin a sentence with 'Do you' or 'Would you'."),
    "question": (
        "Answer the merchant's question directly and only from FACTS. If FACTS does not contain the "
        "answer, say plainly what you can check or do instead - never guess. Then return to the "
        "open thread with one concrete next step."),
    "off_topic": (
        "The merchant asked about something outside your remit. Decline that in one short sentence "
        "without lecturing (point them to the right professional if obvious), then bring the thread "
        "back to the pending item with a single concrete next step."),
    "engaged": (
        "The merchant replied with something substantive. Acknowledge the specific thing they said "
        "in a few words, add one piece of value from FACTS they did not have, and close with one "
        "low-friction next step."),
    "auto_reply_nudge": (
        "The reply was the business's automated WhatsApp greeting, not a person. Say in one light, "
        "non-judgemental line that this looks like an auto-reply, restate the single benefit in "
        "under ten words, and tell the owner exactly what one word to reply when they see it. Keep "
        "it under 35 words."),
    "nudge": (
        "There has been no real answer yet. Do not repeat your earlier message or complain about "
        "silence. Lead with ONE new fact from FACTS they have not seen, and ask a single question "
        "they can answer in three words."),
}


def compose_reply(branch: str, state, contexts: dict, message: str,
                  deadline: float, from_role: str = "merchant") -> dict:
    """Returns {'body','cta','rationale'} for a 'send' action."""
    category = contexts.get("category") or {}
    merchant = contexts.get("merchant") or {}
    trigger = contexts.get("trigger") or {"kind": "conversation_reply", "payload": {}}
    customer = contexts.get("customer")
    customer_facing = from_role == "customer" or (state.send_as == "merchant_on_behalf")

    route = {
        "kind": trigger.get("kind", "conversation_reply"),
        "frame": BRANCH_RULES.get(branch, BRANCH_RULES["engaged"]),
        "levers": "momentum + effort externalisation",
        "cta": "binary_confirm_cancel" if branch == "commitment" else "open_ended",
        "template": "n/a",
        "send_as": "merchant_on_behalf" if customer_facing else "vera",
    }
    f = composer.build_facts(category, merchant, trigger, customer if customer_facing else None,
                            route["send_as"], state.last_now_iso)

    transcript = "\n".join(("merchant: " if t["role"] != "vera" else "you: ") + t["text"]
                           for t in state.turns[-6:])
    system = SYSTEM_REPLY_CUSTOMER if customer_facing else SYSTEM_REPLY
    extra = []
    body = cta = rationale = ""

    for attempt in range(2):
        if not CLIENT.enabled or time.monotonic() > deadline - 2.0:
            break
        prompt = "\n".join([
            "=== FACTS (the only information that exists) ===",
            f.text(),
            "",
            "=== CONVERSATION SO FAR (do not repeat any of your own lines) ===",
            transcript or "(this is the first reply in the thread)",
            "",
            "=== THEIR LATEST MESSAGE ===",
            message.strip(),
            "",
            "=== WHAT TO DO ===",
            route["frame"],
            "Language: " + f.language_rule,
            "Emoji: " + f.emoji_rule,
        ] + (["", "=== YOUR PREVIOUS ATTEMPT WAS REJECTED - FIX AND REWRITE ==="] +
             ["  - " + e for e in extra] if extra else []) +
            ["", "Write the reply now. Return only the JSON object."])
        out = CLIENT.complete_json(system, prompt,
                                   "reply|" + state.conversation_id + "|" + branch + "|" +
                                   str(state.turn_count) + "|a" + str(attempt), deadline)
        if not out:
            break
        cand = sanitize_body(str(out.get("body", "")).strip())
        cand_cta = str(out.get("cta", route["cta"])).strip() or route["cta"]
        extra = composer.validate(cand, cand_cta, f, route, state.sent_norms)
        if branch == "commitment":
            extra += _action_mode_problems(cand)
        if not extra:
            return {"body": cand, "cta": cand_cta,
                    "rationale": str(out.get("rationale", "")).strip() or
                                 "Advancing the thread with a concrete next step."}
        if attempt == 0 and not composer._hard_fail(extra):
            body, cta = cand, cand_cta
            rationale = str(out.get("rationale", "")).strip()

    if body:
        return {"body": body, "cta": cta,
                "rationale": rationale or "Advancing the thread with a concrete next step."}
    return _fallback_reply(branch, state, f, merchant, customer if customer_facing else None)


def _action_mode_problems(body: str) -> list:
    low = (body or "").lower()
    bad = [p for p in QUALIFYING_PHRASES if p in low]
    problems = []
    if bad:
        problems.append('the merchant already said yes - remove the qualifying phrase "' +
                        bad[0] + '" and state what you are doing now')
    if not any(w in low for w in ACTION_WORDS):
        problems.append("name the concrete thing you are doing right now (drafting/sending/"
                        "scheduling) instead of discussing it")
    return problems


def _fallback_reply(branch: str, state, f, merchant: dict, customer) -> dict:
    out = _fallback_reply_raw(branch, state, f, merchant, customer)
    out["body"] = sanitize_body(out.get("body", ""))
    return out


def _fallback_reply_raw(branch: str, state, f, merchant: dict, customer) -> dict:
    name = f.salutation
    offer = f.active_offers[0] if f.active_offers else ""
    anchor = f.anchors[0].split(" = ", 1)[-1] if f.anchors else ""

    if branch == "commitment":
        body = (name + ", starting now. I'm drafting it and will queue it for your approval" +
                (" using your " + offer if offer else "") +
                ". Reply CONFIRM and I'll publish it today.")
        return {"body": body, "cta": "binary_confirm_cancel",
                "rationale": "Merchant committed; switched straight to execution with a single "
                             "confirm step (deterministic fallback)."}
    if branch == "auto_reply_nudge":
        return {"body": ("Looks like an auto-reply. When the owner sees this: it's a 2-minute "
                         "change on your Google profile that costs nothing. Just reply YES and "
                         "I'll set it up."),
                "cta": "binary_yes_no",
                "rationale": "Auto-reply detected; one explicit prompt addressed to the owner."}
    if branch == "off_topic":
        return {"body": (name + ", that one is for your CA - outside what I can do. Back to your "
                         "profile: shall I send the draft for your approval?"),
                "cta": "binary_yes_no",
                "rationale": "Declined the out-of-scope ask in one line and returned to the "
                             "original thread."}
    if branch == "question":
        detail = anchor or "your latest numbers"
        return {"body": (name + ", here's what your data shows: " + detail +
                         ". If you want the rest, say the word and I'll pull it together for you."),
                "cta": "open_ended",
                "rationale": "Answered from the merchant's own context without guessing."}
    return {"body": (name + ", one thing worth a look: " + (anchor or "your 30-day numbers") +
                     ". Want me to put a short plan together for it?"),
            "cta": "open_ended",
            "rationale": "Re-engaged with a new fact from the merchant's own data and a single "
                         "low-friction ask."}
