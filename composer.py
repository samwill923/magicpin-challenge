"""Composition engine for the magicpin Vera bot.

Pipeline for every message:

    resolve contexts -> build_facts()  (deterministic, Python)
                     -> route(trigger.kind)
                     -> ONE LLM call (temperature 0, JSON out)
                     -> validate() -> one corrective retry
                     -> template fallback if the LLM is unreachable/invalid

build_facts() is where specificity comes from: every number the model is
allowed to use is pre-computed here with its provenance, so the model never
does arithmetic and never has to guess. validate() then checks the output only
contains numbers that appear in the facts block, which is our anti-fabrication
guard.
"""
from __future__ import annotations

import re
import time

from llm import CLIENT
from sanitizer import has_dash, sanitize_body

# --------------------------------------------------------------------------
# Static knowledge
# --------------------------------------------------------------------------

# Internal signal strings must never reach the merchant verbatim (-1 penalty
# for exposing internal jargon), so they are translated to plain English here.
SIGNAL_TEXT = {
    "stale_posts": "no Google post published in {v} days",
    "ctr_below_peer_median": "click-through rate is below the category peer median",
    "above_peer_ctr": "click-through rate is above the category peer average",
    "above_peer_median_calls": "calls are above the category peer average",
    "above_peer_calls": "calls are above the category peer average",
    "high_risk_adult_cohort": "a high-risk adult patient cohort in the roster",
    "engaged_in_last_48h": "merchant replied to Vera within the last 48 hours",
    "engaged_in_last_24h": "merchant replied to Vera within the last 24 hours",
    "renewal_due_soon": "subscription renewal due in {v} days",
    "perf_dip_severe": "a severe drop in performance this week",
    "perf_dip_post_expiry": "performance dropped after the subscription expired",
    "unverified_gbp": "the Google Business Profile is not verified yet",
    "dormant_with_vera": "no merchant reply to Vera for {v} days",
    "no_active_offers": "no active offer running right now",
    "no_recent_conversation": "no conversation with Vera yet",
    "no_recent_post": "no recent Google post",
    "high_engagement": "consistently replies to Vera",
    "growing_views_7d": "profile views are growing week-over-week",
    "winback_eligible": "eligible for a win-back (lapsed subscription)",
    "new_merchant": "recently joined magicpin",
    "trial_ending_soon": "the free trial is ending soon",
    "ipl_eligible_locality": "located in an IPL match-night catchment",
    "high_volume": "high order volume",
    "stable_growth": "steady month-over-month growth",
    "seasonal_dip_apr_may": "in the normal April-May seasonal dip",
    "high_retention": "unusually high member retention",
    "active_planning": "currently planning a new programme with Vera",
    "boutique_segment": "boutique-sized studio",
    "compliance_aware": "responds to compliance alerts",
    "high_repeat_rate": "high repeat-customer rate",
    "delivery_not_set_up": "home delivery is not set up yet",
}

# trigger.kind -> digest.kind, used to pick the category item to cite.
KIND_TO_DIGEST_KIND = {
    "research_digest": ("research",),
    "regulation_change": ("compliance",),
    "compliance_alert": ("compliance",),
    "cde_opportunity": ("cde",),
    "category_trend_movement": ("trend",),
    "competitor_opened": ("compete", "trend"),
    "supply_alert": ("alert", "supply"),
    "category_seasonal": ("seasonal",),
    "festival_upcoming": ("seasonal",),
    "seasonal_perf_dip": ("seasonal",),
    "review_theme_emerged": ("tech", "trend"),
    "gbp_unverified": ("trend", "tech"),
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Which slices of context a trigger kind actually needs. Sending the whole
# merchant record costs tokens (Groq free tier: 8000/min) and dilutes the
# model's attention, so each kind gets only what it can legitimately cite.
# The CTR-vs-peer headline and the trigger payload always go in.
PERF_KINDS = {"perf_dip", "perf_spike", "seasonal_perf_dip", "renewal_due", "winback_eligible",
              "gbp_unverified", "dormant_with_vera", "curious_ask_due", "milestone_reached",
              "competitor_opened", "review_theme_emerged", "ipl_match_today"}
REVIEW_KINDS = {"review_theme_emerged", "perf_dip", "milestone_reached"}
SEASON_KINDS = {"festival_upcoming", "category_seasonal", "seasonal_perf_dip", "curious_ask_due",
                "dormant_with_vera", "category_trend_movement", "competitor_opened",
                "ipl_match_today", "gbp_unverified", "perf_spike"}
AGGREGATE_KINDS = {"supply_alert", "research_digest", "winback_eligible", "seasonal_perf_dip",
                   "renewal_due", "regulation_change", "cde_opportunity", "dormant_with_vera"}
PEER_EXTRA_KINDS = {"milestone_reached", "review_theme_emerged", "gbp_unverified"}

# --------------------------------------------------------------------------
# Trigger routing: how each kind should be framed
# --------------------------------------------------------------------------

DEFAULT_ROUTE = {
    "frame": ("Say what changed and why it matters to this business today, using one "
              "number from FACTS. Offer to do the next step for them."),
    "levers": "reciprocity + curiosity",
    "cta": "open_ended",
    "template": "vera_generic_v1",
}

ROUTES = {
    "research_digest": {
        "frame": ("Lead with the journal finding and its trial size, tie it to the part of "
                  "THIS practice's roster it affects, and cite the source at the end. "
                  "Offer to pull the abstract and draft patient-facing copy."),
        "levers": "source credibility + reciprocity + curiosity",
        "cta": "open_ended", "template": "vera_research_digest_v1",
    },
    "regulation_change": {
        "frame": ("State the rule change, the exact deadline and what passes/fails under it. "
                  "Be matter-of-fact, not alarmist. Offer to run the audit checklist with them."),
        "levers": "loss aversion (deadline) + source credibility + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_compliance_alert_v1",
    },
    "supply_alert": {
        "frame": ("Open with the alert and the exact batch/molecule identifiers, say plainly how "
                  "serious it is, then quantify how many of THEIR customers it touches using the "
                  "count in FACTS. Offer the customer note plus the replacement workflow."),
        "levers": "urgency + specificity + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_supply_alert_v1",
    },
    "cde_opportunity": {
        "frame": ("Name the session, speaker, date, credits and fee from FACTS. One line on why "
                  "it is worth their evening. Offer to block the slot or send a reminder."),
        "levers": "peer social proof + low friction",
        "cta": "binary_yes_no", "template": "vera_cde_invite_v1",
    },
    "competitor_opened": {
        "frame": ("Name the competitor, distance and their offer exactly as in FACTS. Do not "
                  "advise a price war; compare on what this merchant is already stronger at, "
                  "using their own numbers. Offer one concrete counter-move."),
        "levers": "loss aversion + social proof + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_competitor_alert_v1",
    },
    "perf_dip": {
        "frame": ("Open with the exact metric and drop from FACTS, give the single most likely "
                  "cause visible in FACTS (not a guess), and propose one fix you will execute."),
        "levers": "loss aversion + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_perf_dip_v1",
    },
    "seasonal_perf_dip": {
        "frame": ("Reassure first: this dip is the expected seasonal pattern. Describe that pattern "
                  "from the seasonal note in FACTS using ONLY the month window that note gives - "
                  "never name a single month inside it, never mention another season or festival, "
                  "never merge two notes. Prove it is not a listing problem with their "
                  "click-through rate against the peer average from FACTS. Then say what not to "
                  "spend on right now and redirect to retention, quoting AT MOST ONE roster number "
                  "(member or customer count) - stacking several makes it read like filler. Name "
                  "their locality. Offer to draft that retention play."),
        "levers": "anxiety pre-emption + contrarian judgement + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_seasonal_context_v1",
    },
    "perf_spike": {
        "frame": ("Name the metric, the rise and the likely driver from FACTS. Tell them how to "
                  "compound it this week. Congratulate in half a sentence, not more."),
        "levers": "momentum + curiosity + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_perf_spike_v1",
    },
    "milestone_reached": {
        "frame": ("State exactly where they are versus the milestone number from FACTS and how "
                  "small the remaining gap is. Offer the one play that closes it."),
        "levers": "goal gradient + social proof + low friction",
        "cta": "binary_yes_no", "template": "vera_milestone_v1",
    },
    "renewal_due": {
        "frame": ("Lead with what their subscription produced in the last 30 days (real numbers "
                  "from FACTS), then the days remaining and the amount. Renewal is the CTA; do "
                  "not bundle other asks."),
        "levers": "loss aversion + value proof",
        "cta": "binary_yes_no", "template": "vera_renewal_v1",
    },
    "winback_eligible": {
        "frame": ("Quantify what lapsed since expiry using FACTS (days since expiry, performance "
                  "drop, customers added meanwhile). No guilt. One low-friction restart step."),
        "levers": "loss aversion + specificity",
        "cta": "binary_yes_no", "template": "vera_winback_v1",
    },
    "dormant_with_vera": {
        "frame": ("Do not mention their silence or send a reminder-shaped message. Bring one new "
                  "useful fact from FACTS (their numbers or a category item) and ask one short "
                  "question they can answer in a few words."),
        "levers": "asking the merchant + reciprocity + curiosity",
        "cta": "open_ended", "template": "vera_reengage_v1",
    },
    "curious_ask_due": {
        "frame": ("Ask ONE short, direct question about what is actually happening in their shop "
                  "this week, and state up front exactly what you will build from their answer. "
                  "Do NOT guess at the answer or speculate about their week: open with one real "
                  "number of theirs from FACTS, then ask."),
        "levers": "asking the merchant + reciprocity + low effort",
        "cta": "open_ended", "template": "vera_curious_ask_v1",
    },
    "active_planning_intent": {
        "frame": ("They already said yes to this idea. Do NOT ask another qualifying question. "
                  "Deliver a concrete first draft now, 3 or 4 short lines, built on their existing "
                  "offer in FACTS. EVERY price must already appear in FACTS: if the plan needs a "
                  "rate FACTS does not give you (a bulk rate, a tier, a discount), do not invent "
                  "one - lay out the structure with the real price you do have and ask them to set "
                  "that one number. Close by asking only whether to proceed with the step you "
                  "name."),
        "levers": "effort externalisation + momentum",
        "cta": "binary_confirm_cancel", "template": "vera_planning_draft_v1",
    },
    "review_theme_emerged": {
        "frame": ("Quote the recurring review theme with its occurrence count and the customer "
                  "quote from FACTS. Tie the cost to a number that is actually in FACTS (their "
                  "calls, orders, views or rating against the peer average) - do not predict a "
                  "consequence you cannot support. Name the operational cause it points to, and "
                  "offer one fix plus a reply template for those reviews."),
        "levers": "loss aversion + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_review_theme_v1",
    },
    "festival_upcoming": {
        "frame": ("Anchor on the festival date and the days remaining from FACTS, plus the "
                  "category seasonal note. Recommend the one thing worth preparing now, tied to "
                  "an offer in FACTS. Keep it useful, not festive fluff."),
        "levers": "planning window + reciprocity",
        "cta": "binary_yes_no", "template": "vera_festival_prep_v1",
    },
    "ipl_match_today": {
        "frame": ("Name the fixture, venue and start time from FACTS. Then give the judgement "
                  "call the category data supports (weeknight vs weekend match behaviour) even "
                  "if it means advising against a promo. Leverage their ACTIVE offer, do not "
                  "invent one. Offer the assets you will produce."),
        "levers": "contrarian data + loss aversion + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_matchday_v1",
    },
    "category_seasonal": {
        "frame": ("Lead with the specific demand shifts from FACTS (up and down), translate them "
                  "into one shelf/menu/schedule action this week. Offer to draft it."),
        "levers": "specificity + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_seasonal_demand_v1",
    },
    "gbp_unverified": {
        "frame": ("State that the profile is unverified and put it next to their own views figure "
                  "and locality from FACTS, so the gap is concrete. Phrase the uplift "
                  "comparatively - verified listings in this category see about that much more "
                  "discovery - never as 'costs you +X% uplift'. Name the verification path exactly "
                  "as FACTS gives it, and offer to walk them through it; you cannot issue or send "
                  "Google's postcard yourself."),
        "levers": "loss aversion + effort externalisation",
        "cta": "binary_yes_no", "template": "vera_gbp_verify_v1",
    },
    # ---- customer-facing kinds (send_as = merchant_on_behalf) ----
    "recall_due": {
        "frame": ("Write as the clinic/studio to their own patient. Name the service due and the "
                  "interval from FACTS, offer the exact slots in FACTS honouring their slot "
                  "preference, and state the real catalogue price. No medical claims."),
        "levers": "personal relevance + concrete slots + low friction",
        "cta": "multi_choice_slot", "template": "merchant_recall_reminder_v1",
    },
    "chronic_refill_due": {
        "frame": ("Write as the pharmacy to the customer (or their carer, per the channel in "
                  "FACTS). List the exact molecules and the run-out date, apply the real discount "
                  "offer from FACTS, confirm the saved delivery address. Respectful, precise."),
        "levers": "timing precision + trust + low friction",
        "cta": "binary_confirm_cancel", "template": "merchant_refill_reminder_v1",
    },
    "customer_lapsed_hard": {
        "frame": ("Write as the business to a member who stopped coming. Use the gap in FACTS "
                  "with no shame or guilt, reference what they were working on, and offer one "
                  "concrete no-commitment restart from the real offers in FACTS."),
        "levers": "warmth + barrier removal + single binary CTA",
        "cta": "binary_yes_no", "template": "merchant_winback_v1",
    },
    "customer_lapsed_soft": {
        "frame": ("Write as the business to a customer whose usual return window has passed. Use "
                  "their visit history from FACTS, keep it short and warm, and make the return "
                  "step one tap."),
        "levers": "personal relevance + low friction",
        "cta": "binary_yes_no", "template": "merchant_softlapse_v1",
    },
    "appointment_tomorrow": {
        "frame": ("Write as the business confirming tomorrow's booking. Include whatever of the "
                  "service, time and stylist/doctor is in FACTS, one line of prep if useful, and "
                  "a one-tap confirm or reschedule."),
        "levers": "no-show prevention + convenience",
        "cta": "binary_confirm_cancel", "template": "merchant_appointment_confirm_v1",
    },
    "trial_followup": {
        "frame": ("Write as the business to someone who just finished a trial session. Reference "
                  "the trial date from FACTS, name the next session option exactly, and keep the "
                  "commitment small."),
        "levers": "momentum + low friction",
        "cta": "binary_yes_no", "template": "merchant_trial_followup_v1",
    },
    "wedding_package_followup": {
        "frame": ("Write as the salon to a bride-to-be. Anchor on the days-to-wedding number and "
                  "the trial date in FACTS, explain why this is the right prep window, quote the "
                  "real package price, honour her slot preference."),
        "levers": "timing window + personal relevance + single CTA",
        "cta": "binary_yes_no", "template": "merchant_bridal_followup_v1",
    },
}

# Customer-scoped kinds that we downgrade to a merchant-facing approval ask when
# the CustomerContext has not been pushed to us.
CUSTOMER_KINDS = {"recall_due", "chronic_refill_due", "customer_lapsed_hard",
                  "customer_lapsed_soft", "appointment_tomorrow", "trial_followup",
                  "wedding_package_followup", "customer_lapsed", "booking_reminder"}


def route_for(trigger: dict, has_customer: bool) -> dict:
    kind = (trigger or {}).get("kind", "")
    route = dict(ROUTES.get(kind, DEFAULT_ROUTE))
    route["kind"] = kind
    payload = (trigger or {}).get("payload") or {}
    try:
        days_until = int(payload.get("days_until") or payload.get("days_to_wedding") or 0)
    except (TypeError, ValueError):
        days_until = 0
    if kind == "festival_upcoming" and days_until > 45:
        # Diwali in 188 days is not a reason to message anyone today. Lead with
        # something the merchant can act on this week instead.
        route["frame"] = (
            "The festival in the payload is " + str(days_until) + " days away, far too distant to "
            "act on, so do NOT lead with it and do not build the message around it. Lead with what "
            "is live in their numbers right now (their views, calls or click-through against the "
            "peer average) or the current season note in FACTS, propose the one thing worth doing "
            "this week, and mention the festival at most as a short planning aside, or not at all.")
        route["levers"] = "current performance + reciprocity"
        route["cta"] = "binary_yes_no"
    if payload.get("placeholder"):
        # The event fired but carries no detail. The kind-specific frame would
        # ask for names, numbers and deadlines we do not have, which is how a
        # model ends up inventing a competitor - so swap in a detail-free frame.
        route["frame"] = (
            "This " + kind.replace("_", " ") + " event fired but carries NO details: no names, "
            "numbers, dates or amounts. Do not state or imply any specific event fact, and never "
            "name a competitor, product, fixture or festival. Instead lead with the strongest real "
            "number from FACTS about this business, say what you would do about it, and ask one "
            "short question they can answer in a few words.")
        route["levers"] = "asking the merchant + reciprocity + specificity from their own numbers"
        route["cta"] = "open_ended"
    customer_scoped = (trigger or {}).get("scope") == "customer" or kind in CUSTOMER_KINDS
    if customer_scoped and not has_customer:
        # We know a customer event fired but were never given the profile.
        # Ask the merchant for approval instead of inventing customer details.
        route = {
            "kind": kind,
            "frame": ("A customer-level event fired but we have no profile for that customer, so "
                      "write to the MERCHANT. Name the event, say you have the draft ready, and "
                      "ask for a go-ahead to send it on their behalf. Never invent the "
                      "customer's name or history."),
            "levers": "effort externalisation + reciprocity",
            "cta": "binary_yes_no",
            "template": "vera_customer_outreach_approval_v1",
            "send_as": "vera",
        }
        return route
    route["send_as"] = "merchant_on_behalf" if customer_scoped else "vera"
    return route


# --------------------------------------------------------------------------
# Facts extraction
# --------------------------------------------------------------------------

class Facts:
    def __init__(self):
        self.sections: list[tuple[str, list[str]]] = []
        self.anchors: list[str] = []
        self.salutation = "Hi"
        self.language_rule = "Write in English."
        self.emoji_rule = "Do not use emoji."
        self.merchant_name = ""
        self.owner = ""
        self.locality = ""
        self.active_offers: list[str] = []
        self.catalogue_offers: list[str] = []
        self.customer_name = ""
        self.digest_item: dict | None = None
        self.numbers: set[str] = set()

    def add(self, title: str, lines: list[str]) -> None:
        lines = [ln for ln in lines if ln]
        if lines:
            self.sections.append((title, lines))

    def text(self) -> str:
        out = []
        for title, lines in self.sections:
            out.append(title)
            out.extend("  - " + ln for ln in lines)
        return "\n".join(out)


def _pct(x) -> str:
    """0.18 -> '+18%'   -0.05 -> '-5%'"""
    try:
        v = float(x) * 100
    except (TypeError, ValueError):
        return ""
    sign = "+" if v >= 0 else "-"
    return sign + _num(abs(v)) + "%"


def _num(x) -> str:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(f - round(f)) < 0.05:
        return str(int(round(f)))
    return ("%.1f" % f).rstrip("0").rstrip(".")


def _rel_gap(mine, peer) -> str:
    """'31% below the peer average' for ctr 0.021 vs 0.030."""
    try:
        mine, peer = float(mine), float(peer)
        if peer <= 0:
            return ""
        gap = (mine - peer) / peer * 100
    except (TypeError, ValueError):
        return ""
    if abs(gap) < 5:
        return "in line with the peer average"
    return _num(abs(gap)) + "% " + ("above" if gap > 0 else "below") + " the peer average"


def _human_signal(sig: str) -> str:
    if not isinstance(sig, str):
        return ""
    base, _, val = sig.partition(":")
    val = val.rstrip("d")
    tpl = SIGNAL_TEXT.get(base)
    if tpl:
        return tpl.format(v=val) if "{v}" in tpl else tpl
    for key, tpl in SIGNAL_TEXT.items():
        if base.startswith(key):
            digits = re.findall(r"\d+", base)
            return tpl.format(v=digits[0] if digits else val) if "{v}" in tpl else tpl
    return base.replace("_", " ")


def _pick_digest(category: dict, trigger: dict) -> dict | None:
    digest = (category or {}).get("digest") or []
    if not digest:
        return None
    payload = (trigger or {}).get("payload") or {}
    wanted_id = payload.get("top_item_id") or payload.get("digest_item_id") or payload.get("alert_id")
    if wanted_id:
        for item in digest:
            if item.get("id") == wanted_id:
                return item
    for dk in KIND_TO_DIGEST_KIND.get((trigger or {}).get("kind", ""), ()):
        for item in digest:
            if item.get("kind") == dk:
                return item
    return None


def _seasonal_beat(category: dict, now_month: str) -> dict | None:
    for beat in (category or {}).get("seasonal_beats") or []:
        rng = beat.get("month_range", "")
        if now_month in rng:
            return beat
        if "-" in rng:
            a, _, b = rng.partition("-")
            if a.strip() in MONTHS and b.strip() in MONTHS:
                i, j, k = MONTHS.index(a.strip()), MONTHS.index(b.strip()), MONTHS.index(now_month)
                if (i <= k <= j) or (i > j and (k >= i or k <= j)):
                    return beat
    return None


def _language_rule(merchant: dict, customer: dict | None, category: dict) -> str:
    local = {"ta": "Tamil", "te": "Telugu", "kn": "Kannada", "mr": "Marathi", "bn": "Bengali"}
    if customer:
        pref = str(((customer.get("identity") or {}).get("language_pref") or "en")).lower()
        if pref in ("hi", "hindi"):
            return ("Write in simple Roman-script Hindi (Hinglish) — the customer's stated "
                    "preference is Hindi. No Devanagari script.")
        if "hi-en" in pref or pref in ("hi-en mix", "hinglish"):
            return ("Write in a natural Hindi-English mix in Roman script, the way Indians "
                    "actually message on WhatsApp. Keep prices and dates in English/numerals.")
        for code, name in local.items():
            if pref.startswith(code):
                return ("Write mainly in English; one or two everyday " + name + " words in Roman "
                        "script are welcome (a greeting or sign-off). No " + name + " script.")
        return "Write in clear, simple English."
    langs = [str(x).lower() for x in ((merchant.get("identity") or {}).get("languages") or ["en"])]
    code_mix = ((category or {}).get("voice") or {}).get("code_mix", "")
    extra = [local[c] for c in local if c in langs]
    if "hi" in langs and code_mix == "hindi_english_natural":
        # All in, or not at all: a Hindi phrase bolted onto an English sentence
        # reads like a mistake, which is what the judge sees too.
        rule = ("This merchant speaks Hindi, so write the WHOLE message in natural Hinglish - "
                "Roman script, the way an Indian business owner actually types on WhatsApp. Every "
                "sentence code-mixed, not one Hindi phrase stapled to English prose. Numbers, "
                "prices, dates and technical terms stay in English/numerals.")
    else:
        rule = ("Write in English only, start to finish. Do not insert Hindi or any other "
                "language: a part-Hindi sentence reads like a typo here.")
    if extra and "hi" not in langs:
        rule += " The merchant also speaks " + "/".join(extra) + ", but keep this message English."
    return rule


def _salutation(category: dict, owner: str, merchant_name: str) -> str:
    """Use the category's own salutation pattern, e.g. dentists get 'Dr. Meera'."""
    if not owner:
        return merchant_name
    examples = ((category or {}).get("voice") or {}).get("salutation_examples") or []
    for ex in examples:
        if "{first_name}" in ex:
            title = ex.split("{first_name}")[0].strip()
            if title and title.lower() not in ("hi", "hello") and not owner.lower().startswith(
                    title.lower().rstrip(".")):
                return title + " " + owner
            return owner
    return owner


def _emoji_rule(category: dict, send_as: str) -> str:
    slug = (category or {}).get("slug", "")
    if send_as == "merchant_on_behalf":
        return "At most one emoji, near the start, only if it fits the business. Never more than one."
    if slug in ("dentists", "pharmacies"):
        return "No emoji — this is a clinical/professional peer conversation."
    return "At most one emoji, and only if it genuinely fits. Zero is fine."


def build_facts(category: dict, merchant: dict, trigger: dict,
                customer: dict | None, send_as: str, now_iso: str = "") -> Facts:
    f = Facts()
    category = category or {}
    merchant = merchant or {}
    trigger = trigger or {}
    ident = merchant.get("identity") or {}
    perf = merchant.get("performance") or {}
    peer = category.get("peer_stats") or {}
    sub = merchant.get("subscription") or {}
    payload = trigger.get("payload") or {}
    kind = str(trigger.get("kind", ""))

    f.merchant_name = ident.get("name", "the business")
    f.owner = (ident.get("owner_first_name") or "").strip()
    f.locality = ident.get("locality", "")
    f.salutation = _salutation(category, f.owner, f.merchant_name)
    f.language_rule = _language_rule(merchant, customer, category)
    f.emoji_rule = _emoji_rule(category, send_as)
    # A customer-facing message is written by the business to its own customer:
    # the merchant's dashboard numbers are irrelevant there and must not leak.
    customer_facing = send_as == "merchant_on_behalf"

    # -- merchant identity / subscription -----------------------------------
    id_lines = [
        "business name: " + str(f.merchant_name),
        ("owner/staff first name (sign off as this person if it fits): " if customer_facing else
         "owner/contact first name (use this in the greeting): ") + (f.owner or "not known"),
        "locality and city: " + ", ".join(x for x in [ident.get("locality"), ident.get("city")] if x),
        "category: " + str(category.get("display_name") or merchant.get("category_slug", "")),
    ]
    if not customer_facing:
        id_lines.append("Google profile verified: " + ("yes" if ident.get("verified") else "no"))
        if ident.get("established_year"):
            id_lines.append("operating since: " + str(ident["established_year"]))
    status = sub.get("status")
    if status and not customer_facing:
        bits = "subscription: " + str(status) + " (" + str(sub.get("plan", "")) + ")"
        if sub.get("days_remaining"):
            bits += ", " + str(sub["days_remaining"]) + " days remaining"
        if sub.get("days_since_expiry"):
            bits += ", expired " + str(sub["days_since_expiry"]) + " days ago"
        id_lines.append(bits)
    f.add("MERCHANT" if not customer_facing else "THE BUSINESS YOU ARE WRITING AS", id_lines)

    # -- performance: the CTR-vs-peer headline always, the rest only when the
    # trigger is about performance ------------------------------------------
    perf_lines = []
    win = perf.get("window_days", 30)
    wide = kind in PERF_KINDS
    metric = str((payload or {}).get("metric") or "")
    if perf.get("ctr") is not None:
        line = "click-through rate: " + _num(float(perf["ctr"]) * 100) + "%"
        if peer.get("avg_ctr"):
            line += " vs category peer average " + _num(float(peer["avg_ctr"]) * 100) + "%"
            gap = _rel_gap(perf["ctr"], peer["avg_ctr"])
            if gap:
                line += " (" + gap + ")"
        perf_lines.append(line)
    for key, label in (("views", "profile views"), ("calls", "calls"),
                       ("directions", "direction requests"), ("leads", "leads")):
        if perf.get(key) is None:
            continue
        if key in ("directions", "leads") and metric != key:
            continue
        if not (wide or metric == key or key == "views"):
            continue
        line = label + " last " + str(win) + " days: " + _num(perf[key])
        peer_key = "avg_" + key + "_30d"
        if peer.get(peer_key):
            line += " vs peer average " + _num(peer[peer_key]) + " (" +                     _rel_gap(perf[key], peer[peer_key]) + ")"
        perf_lines.append(line)
    d7 = perf.get("delta_7d") or {}
    for key, label in (("views_pct", "views"), ("calls_pct", "calls"),
                       ("ctr_pct", "click-through rate")):
        if d7.get(key) is None:
            continue
        if wide or metric + "_pct" == key:
            perf_lines.append(label + " week-over-week: " + _pct(d7[key]))
    if kind in PEER_EXTRA_KINDS:
        for k in ("avg_rating", "avg_review_count", "avg_post_freq_days"):
            if peer.get(k):
                perf_lines.append("category peer " +
                                  k.replace("avg_", "average ").replace("_", " ") +
                                  ": " + _num(peer[k]))
    if not customer_facing:
        f.add("PERFORMANCE (last " + str(win) + " days, with peer benchmarks)", perf_lines)

    # -- offers -------------------------------------------------------------
    offers = merchant.get("offers") or []
    f.active_offers = [o.get("title", "") for o in offers if o.get("status") == "active"]
    other = [(o.get("title", ""), o.get("status", "")) for o in offers if o.get("status") != "active"]
    offer_lines = []
    if f.active_offers:
        offer_lines.append("ACTIVE offers (safe to promote verbatim): " + "; ".join(f.active_offers))
    else:
        offer_lines.append("ACTIVE offers: none running right now")
    if other:
        offer_lines.append("inactive offers (do NOT promote as live): " +
                           "; ".join(t + " [" + s + "]" for t, s in other))
    f.catalogue_offers = [o.get("title", "") for o in (category.get("offer_catalog") or [])]
    if not f.active_offers:
        cat_offers = f.catalogue_offers[:5]
        if cat_offers:
            offer_lines.append("category catalogue templates (propose one as a NEW offer, never as "
                               "already live): " + "; ".join(cat_offers))
    f.add("OFFERS", offer_lines)

    # -- signals, customer base, reviews ------------------------------------
    sig_lines = [_human_signal(s) for s in (merchant.get("signals") or [])]
    agg = merchant.get("customer_aggregate") or {}
    agg_labels = {
        "total_unique_ytd": "unique customers year-to-date",
        "total_active_members": "active members",
        "lapsed_180d_plus": "customers lapsed more than 180 days",
        "lapsed_90d_plus": "customers lapsed more than 90 days",
        "retention_6mo_pct": "6-month retention",
        "retention_3mo_pct": "3-month retention",
        "high_risk_adult_count": "high-risk adult patients in the roster",
        "chronic_rx_count": "chronic-prescription customers",
        "repeat_customer_pct": "repeat-customer share",
        "monthly_churn_pct": "monthly churn",
        "trial_to_paid_pct": "trial-to-paid conversion",
        "delivery_orders_30d": "delivery orders (30d)",
        "dine_in_orders_30d": "dine-in orders (30d)",
        "delivery_share_pct": "delivery share of orders",
    }
    if kind not in AGGREGATE_KINDS and not customer_facing:
        agg_labels = {k: v for k, v in agg_labels.items()
                      if k in ("total_unique_ytd", "total_active_members", "chronic_rx_count",
                               "high_risk_adult_count")}
    for key, label in agg_labels.items():
        if agg.get(key) not in (None, 0):
            val = agg[key]
            sig_lines.append(label + ": " + (_num(float(val) * 100) + "%" if key.endswith("_pct")
                                             else _num(val)))
    if not customer_facing:
        f.add("MERCHANT SIGNALS AND CUSTOMER BASE", sig_lines)

    rev_lines = []
    for theme in merchant.get("review_themes") or []:
        line = (str(theme.get("theme", "")).replace("_", " ") + ": " +
                str(theme.get("occurrences_30d", "")) + " mentions in 30 days (" +
                str(theme.get("sentiment", "")) + ")")
        if theme.get("common_quote"):
            line += ', typical quote: "' + str(theme["common_quote"]) + '"'
        rev_lines.append(line)
    if not customer_facing and kind in REVIEW_KINDS:
        f.add("REVIEW THEMES", rev_lines)

    # -- conversation history ----------------------------------------------
    hist = merchant.get("conversation_history") or []
    hist_lines = []
    for turn in hist[-2:]:
        who = "Vera" if turn.get("from") == "vera" else "merchant"
        body = str(turn.get("body", ""))
        hist_lines.append(who + " (" + str(turn.get("ts", ""))[:10] + "): " +
                          (body[:120] + "..." if len(body) > 120 else body))
    if hist_lines:
        hist_lines.append("You have messaged this merchant before: do NOT re-introduce yourself "
                          "and do NOT repeat any line above.")
    else:
        hist_lines.append("No prior conversation on record.")
    if not customer_facing:
        f.add("RECENT CONVERSATION WITH VERA", hist_lines)

    # -- category voice -----------------------------------------------------
    voice = category.get("voice") or {}
    voice_lines = [
        "tone: " + str(voice.get("tone", "")) + " / register: " + str(voice.get("register", "")),
        "vocabulary you may use: " + ", ".join((voice.get("vocab_allowed") or [])[:6]),
        "FORBIDDEN words and claims: " + ", ".join(voice.get("vocab_taboo") or []),
    ]
    if voice.get("tone_examples"):
        voice_lines.append('tone examples: "' + '" / "'.join(voice["tone_examples"][:2]) + '"')
    f.add("CATEGORY VOICE", voice_lines)

    # -- the citable category item ------------------------------------------
    item = None if customer_facing else _pick_digest(category, trigger)
    f.digest_item = item
    if item:
        item_lines = ["title: " + str(item.get("title", ""))]
        for key in ("source", "summary", "actionable", "date", "credits", "trial_n",
                    "patient_segment"):
            if item.get(key):
                item_lines.append(key.replace("_", " ") + ": " + str(item[key]))
        f.add("CATEGORY ITEM TO CITE (this is the only research/news you may reference)", item_lines)

    month = (now_iso[5:7] if len(now_iso) >= 7 else "")
    beat = _seasonal_beat(category, MONTHS[int(month) - 1] if month.isdigit() and 1 <= int(month) <= 12
                          else "")
    season_lines = []
    if beat:
        season_lines.append("current season note (" + str(beat.get("month_range", "")) + "): " +
                            str(beat.get("note", "")))
    trends = category.get("trend_signals") or []
    if trends:
        top = max(trends, key=lambda t: t.get("delta_yoy", 0))
        season_lines.append('search trend: "' + str(top.get("query", "")) + '" ' +
                            _pct(top.get("delta_yoy")) + " year-on-year" +
                            (", age band " + str(top.get("segment_age")) if top.get("segment_age") else ""))
    if not customer_facing and kind in SEASON_KINDS:
        f.add("SEASON AND SEARCH TRENDS", season_lines)

    # -- the trigger --------------------------------------------------------
    trg_lines = [
        "kind: " + str(trigger.get("kind", "")),
        "source: " + str(trigger.get("source", "")) + ", urgency " + str(trigger.get("urgency", "")) +
        "/5, scope " + str(trigger.get("scope", "merchant")),
    ]
    for key, val in payload.items():
        # internal ids add nothing the model may say out loud
        if key in ("placeholder", "metric_or_topic", "top_item_id", "digest_item_id", "alert_id",
                   "merchant_id", "customer_id") or val in (None, "", [], {}):
            continue
        trg_lines.append("payload." + key + ": " + _flat(val))
    if payload.get("placeholder"):
        trg_lines.append("NOTE: this trigger carries no detail payload. Anchor the message on " +
                         ("this customer's own history and the active offer above"
                          if customer_facing else
                          "the merchant's own numbers above and on the category item") +
                         ", and do not invent event details (no made-up times, batches or amounts).")
    f.add("TRIGGER (the reason you are messaging right now)", trg_lines)

    # -- customer -----------------------------------------------------------
    if customer:
        c_ident = customer.get("identity") or {}
        rel = customer.get("relationship") or {}
        prefs = customer.get("preferences") or {}
        f.customer_name = str(c_ident.get("name", "")).split(" (")[0].strip()
        cust_lines = [
            "customer first name (use it): " + str(c_ident.get("name", "")),
            "language preference: " + str(c_ident.get("language_pref", "")),
            "relationship state: " + str(customer.get("state", "")),
        ]
        if c_ident.get("age_band"):
            cust_lines.append("age band: " + str(c_ident["age_band"]))
        if c_ident.get("senior_citizen"):
            cust_lines.append("senior citizen: yes (be respectful, unhurried)")
        for key, label in (("visits_total", "total visits"), ("first_visit", "first visit"),
                           ("last_visit", "last visit"), ("lifetime_value", "lifetime value (Rs)")):
            if rel.get(key):
                cust_lines.append(label + ": " + str(rel[key]))
        if rel.get("services_received"):
            cust_lines.append("services received: " + ", ".join(str(s) for s in rel["services_received"][:6]))
        if rel.get("favourite_dish"):
            cust_lines.append("favourite dish: " + str(rel["favourite_dish"]))
        if rel.get("chronic_conditions"):
            cust_lines.append("chronic conditions on file: " + ", ".join(rel["chronic_conditions"]))
        for key, val in prefs.items():
            if val not in (None, "", []):
                cust_lines.append("preference " + key.replace("_", " ") + ": " + _flat(val))
        consent = customer.get("consent") or {}
        if consent.get("scope"):
            cust_lines.append("consented message types (stay inside these): " +
                              ", ".join(consent["scope"]))
        f.add("CUSTOMER (you are writing to this person as the business)", cust_lines)

    # numbers the model is allowed to use + strongest anchors
    f.numbers = _allowed_numbers(f.text())
    f.anchors = _anchor_hints(f, trigger, item, {} if customer_facing else perf,
                              {} if customer_facing else peer, customer)
    return f


def _flat(val) -> str:
    if isinstance(val, dict):
        if "label" in val:
            return str(val["label"])
        return ", ".join(k + "=" + _flat(v) for k, v in val.items())
    if isinstance(val, list):
        return "; ".join(_flat(v) for v in val)
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, float) and 0 < abs(val) < 1:
        return _pct(val)
    return str(val)


def _anchor_hints(f: Facts, trigger: dict, item, perf, peer, customer) -> list[str]:
    """The 2-3 strongest verifiable anchors, so the model leads with one."""
    out = []
    payload = (trigger or {}).get("payload") or {}
    for key in ("affected_batches", "molecule_list", "match", "festival", "deadline_iso",
                "days_until", "days_to_wedding", "days_remaining", "days_since_last_visit",
                "delta_pct", "value_now", "milestone_value", "their_offer", "competitor_name",
                "available_slots", "stock_runs_out_iso", "estimated_uplift_pct", "trends",
                "occurrences_30d", "renewal_amount", "days_since_expiry"):
        if payload.get(key) not in (None, "", [], {}):
            out.append("trigger " + key + " = " + _flat(payload[key]))
    if item and item.get("source"):
        out.append("citation: " + str(item.get("source")))
    if perf.get("ctr") is not None and peer.get("avg_ctr"):
        out.append("their CTR " + _num(float(perf["ctr"]) * 100) + "% vs peer " +
                   _num(float(peer["avg_ctr"]) * 100) + "%")
    if perf.get("views") is not None:
        out.append("their 30-day views " + _num(perf["views"]))
    if customer:
        rel = customer.get("relationship") or {}
        if rel.get("last_visit"):
            out.append("their last visit " + str(rel["last_visit"]))
        if rel.get("visits_total"):
            out.append("visits so far " + str(rel["visits_total"]))
        if f.active_offers:
            out.append("real offer to quote: " + f.active_offers[0])
    return out[:3]


NUM_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _norm_num(tok: str) -> str:
    tok = tok.replace(",", "").rstrip(".")
    if tok.endswith(".0"):
        tok = tok[:-2]
    return tok


def _allowed_numbers(facts_text: str) -> set:
    allowed = set()
    for tok in NUM_TOKEN.findall(facts_text):
        n = _norm_num(tok)
        allowed.add(n)
        try:
            val = float(n)
        except ValueError:
            continue
        # let the model render 0.38 as 38%, 4999 as 4,999, 2.1 as 2
        if 0 < val < 1:
            allowed.add(_norm_num(_num(val * 100)))
        if val == int(val):
            allowed.add(str(int(val)))
            allowed.add("{:,}".format(int(val)).replace(",", ""))
        allowed.add(_norm_num(_num(round(val))))
    return allowed


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

SYSTEM_MERCHANT = """You are Vera, magicpin's marketing assistant, messaging an Indian local-business owner on WhatsApp like a sharp colleague who already did the homework.

RULES (breaking any one rejects the message):
1. Every number, date, price, source, name and count must appear in FACTS. Nothing else exists: never invent research, competitors, offers, slots or statistics.
2. First sentence carries a verifiable anchor from FACTS (number, date, source or named item). No generic "grow your sales" opener.
2b. Then name at least one more thing specific to THIS business from FACTS: prefer their locality, their business name, their own views/calls/click-through figure, or their active offer by its exact title; their roster counts (members, patients, chronic-Rx customers) work as extra colour. A message that only repeats the trigger number is not personalised.
2bb. When you quote a number from their own records rather than their Google profile (members, patients, chronic-Rx customers, retention, churn, orders), introduce it as theirs - "your dashboard shows 245 active members", "your roster has 124 high-risk adults" - so the provenance is obvious.
2c. Never attribute a fact to a source, study, dashboard or dataset unless that exact source is named in FACTS. If FACTS gives no source, state the fact plainly with no attribution. Do not claim a consequence you cannot support from FACTS ("you will lose footfall") - say what the number shows instead.
3. No preamble, no "hope you're well", no re-introducing yourself. Start with their name, then the fact.
4. Exactly ONE ask, as the final sentence. Never two or three options. At most one question mark in the message.
5. Obey the category voice; never use a forbidden word or claim. Peer-to-peer, no hype, no ALL CAPS.
6. No URLs, domains, phone numbers, emails or {{1}} placeholders.
7. Offer the concrete artefact you will produce (draft, post, list, audit) and cap the effort you need from them.
8. 40-75 words, 2-4 short sentences. Cut every word that carries no fact.
9. Never expose internal jargon, field names or trigger names.
10. No dashes of any kind - no em dash, no en dash, no hyphenated compounds. Use commas and full stops, and write "click through", "week on week", "3 month recall".
11. Never guess, hedge or speculate. No "I'm guessing", "probably", "perhaps". State what FACTS supports; if you want their input, ask a direct question.

Output one JSON object only:
{"body": "<message>", "cta": "<open_ended|binary_yes_no|binary_confirm_cancel|multi_choice_slot|none>", "rationale": "<one sentence: anchor, lever, expected reply>"}"""

SYSTEM_CUSTOMER = """You are writing a WhatsApp message ON BEHALF OF an Indian local business, from the business's own number, to one of ITS OWN customers. You are the business - never mention magicpin, Vera, an assistant or a bot.

RULES (breaking any one rejects the message):
1. Every number, date, price, service, slot and name must appear in FACTS. Never invent slots, prices, offers or history.
2. Open with the customer's first name and the business name, then the concrete reason you are writing (their last visit, their booking, a due date).
3. Exactly ONE ask, as the final sentence, one tap to act. Two named appointment slots are allowed; nothing else. At most one question mark.
4. Obey the category voice and the customer's language preference. No medical promises, no pressure, no guilt.
5. Stay inside the customer's consented message types.
6. No URLs, phone numbers, emails or {{1}} placeholders.
7. 35-65 words, warm and easy to read.
8. No dashes of any kind - no em dash, no en dash, no hyphenated compounds. Use commas, and write "6 month cleaning", "walk in".
9. Never guess or speculate about them. Only what FACTS says.

Output one JSON object only:
{"body": "<message>", "cta": "<open_ended|binary_yes_no|binary_confirm_cancel|multi_choice_slot|none>", "rationale": "<one sentence: anchor, lever, expected reply>"}"""


def build_prompt(f: Facts, route: dict, extra_rules: list[str] | None = None) -> str:
    anchors = "\n".join("  - " + a for a in f.anchors) or "  - (use the strongest number in FACTS)"
    if route.get("send_as") == "merchant_on_behalf":
        greeting = ("open with the CUSTOMER's first name from FACTS and identify the business ("
                    + f.merchant_name + "). You are the business, not a platform.")
    else:
        greeting = "address the merchant as " + f.salutation + " (exactly as given in FACTS)."
    parts = [
        "=== FACTS (the only information that exists; everything you write must trace back here) ===",
        f.text(),
        "",
        "=== STRONGEST ANCHORS (lead with one of these) ===",
        anchors,
        "",
        "=== THIS MESSAGE ===",
        "Why you are messaging now: " + str(route.get("kind", "")).replace("_", " "),
        "How to frame it: " + route["frame"],
        "Compulsion levers to use: " + route["levers"],
        "Suggested CTA shape: " + route["cta"],
        "Greeting: " + greeting,
        "Language: " + f.language_rule,
        "Emoji: " + f.emoji_rule,
    ]
    if extra_rules:
        parts += ["", "=== YOUR PREVIOUS ATTEMPT WAS REJECTED - FIX THESE AND REWRITE ==="]
        parts += ["  - " + r for r in extra_rules]
    parts += ["", "Write the message now. Return only the JSON object."]
    return "\n".join(parts)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

URL_RE = re.compile(r"(https?://|www\.|\b[a-z0-9-]+\.(com|in|org|net|co|io)\b)", re.I)
PHONE_RE = re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PREAMBLE_RE = re.compile(
    r"(hope (you|this) (are|is|find)|hope you('| a)re doing|i am reaching out|i'm reaching out|"
    r"just wanted to (reach|check in with you)|my name is|this is vera (here|from)|i am vera|"
    r"i'm vera|greetings|dear (sir|madam)|trust this (message|finds))", re.I)
TIME_UNIT_RE = re.compile(r"^\s*(-|to|and)?\s*(min|mins|minute|minutes|sec|second|seconds|hour|"
                          r"hours|hrs|ghante|ghanta|din|day|days|week|weeks|month|months)", re.I)
CTA_WORDS = ("reply", "say", "send", "confirm", "tell me", "want me", "shall i", "should i",
             "let me know", "batao", "bhej", "bata", "chalega", "karu", "karun", "kar dun",
             "pick", "choose", "book", "just say", "yes", "ok")

# "noted in magicpin gym data, Apr 2026" - an attribution the contexts never made.
CITATION_RE = re.compile(r"(?:per|as per|according to|noted in|noted by|reported in|reported by|"
                         r"cited in|source:|sourced from)\s+([^.;?!\n]{4,70})", re.I)

ALWAYS_SAFE_NUMBERS = {"1", "2", "3", "4", "5", "10", "15", "20", "24", "30", "45", "60", "90", "2"}


def sentences(body: str) -> list[str]:
    parts = re.split(r"(?<=[.!?—])\s+|\n+", body.strip())
    return [p.strip() for p in parts if p.strip()]


def validate(body: str, cta: str, f: Facts, route: dict,
             banned_bodies: set | None = None) -> list[str]:
    """Return a list of human-readable violations ('' == clean)."""
    problems = []
    text = (body or "").strip()
    if not text:
        return ["the body was empty"]

    if URL_RE.search(text):
        problems.append("remove the URL/web address - links are rejected by WhatsApp templates")
    if PHONE_RE.search(text) or EMAIL_RE.search(text):
        problems.append("remove the phone number/email address")
    if "{{" in text or "}}" in text:
        problems.append("remove the template placeholder braces")
    if PREAMBLE_RE.search(text):
        problems.append("delete the preamble/self-introduction and open with the name plus the fact")

    words = len(text.split())
    if words > 95 or len(text) > 620:
        problems.append("cut it to 75 words or fewer - it is too long at " + str(words) + " words")
    if words < 18:
        problems.append("too short: add the verifiable anchor and the concrete next step")

    taboo = [t for t in ((f_voice(f) or {}).get("vocab_taboo") or [])]
    lower = text.lower()
    for word in taboo:
        w = str(word).split("(")[0].strip().lower()
        if w and w in lower:
            problems.append('remove the forbidden phrase "' + w + '" (category taboo)')

    if text.count("?") > 1:
        problems.append("only one question mark allowed - keep a single CTA")
    if cta != "none":
        last = sentences(text)[-1].lower()
        if "?" not in last and not any(w in last for w in CTA_WORDS):
            problems.append("move the call-to-action into the final sentence")
    if re.search(r"reply\s+\w+\s+for\b.*\breply\s+\w+\s+for\b", lower) and cta != "multi_choice_slot":
        problems.append("only one reply option - drop the multi-choice menu")
    if re.search(r"\b(?:flat\s+)?\d+%\s*(off|discount)\b", lower):
        active = " ".join(f.active_offers).lower()
        if "%" not in active:
            problems.append("replace the generic percentage discount with a service-at-price offer "
                            "from FACTS")
    if not f.owner and re.search(r"^(hi|hello|namaste)\s+(there|sir|madam)", lower):
        problems.append("greet with the business name from FACTS instead of a generic greeting")
    if route.get("send_as") == "merchant_on_behalf":
        if f.customer_name and f.customer_name.lower() not in lower:
            problems.append("greet the customer by their first name (" + f.customer_name + ")")
    elif f.owner and f.owner.split()[-1].lower() not in lower and f.merchant_name.lower() not in lower:
        problems.append("address the merchant by name (" + f.salutation + ")")

    # anti-fabrication: every substantial number must trace back to FACTS
    for match in NUM_TOKEN.finditer(text):
        tok = _norm_num(match.group())
        if tok in f.numbers or tok in ALWAYS_SAFE_NUMBERS:
            continue
        tail = text[match.end():match.end() + 14]
        if TIME_UNIT_RE.match(tail) and _safe_small(tok):
            continue
        if match.group().endswith("%") or "%" in text[match.end():match.end() + 2]:
            problems.append('the figure "' + match.group() + '%" is not in FACTS - use only numbers '
                            'from FACTS')
        else:
            problems.append('the number "' + match.group() + '" is not in FACTS - use only numbers '
                            'from FACTS')

    problems += _citation_problems(text, f.text())

    if has_dash(text):
        problems.append("remove every dash: use commas, or open the compound up "
                        '("click through", "week on week")')
    if re.search(r"\b(i'?m guessing|i am guessing|my guess|i suspect|probably|perhaps|maybe)\b",
                 lower):
        problems.append("no guessing or hedging - state only what FACTS supports, and ask a "
                        "direct question if you want their input")

    if banned_bodies and _normalize(text) in banned_bodies:
        problems.append("this exact message was already sent - say something new")
    if route.get("send_as") == "merchant_on_behalf" and re.search(r"\b(vera|magicpin)\b", lower):
        problems.append("this message goes out from the business's own number - never mention "
                        "magicpin or Vera")
    return problems[:6]


def _citation_problems(text: str, facts_text: str) -> list:
    """Flag any source attribution that FACTS does not actually make."""
    out = []
    low_facts = facts_text.lower()
    for match in CITATION_RE.finditer(text):
        span = match.group(1).strip(" .,;:-")
        # only treat it as a citation if it looks like a source: a year or a
        # capitalised name. "per day", "per plate", "per your reply" are not.
        if not (re.search(r"\d{4}", span) or re.search(r"[A-Z][A-Za-z]{2,}", span)):
            continue
        tokens = [t.lower() for t in re.findall(r"[A-Za-z]{4,}", span)]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in low_facts)
        if hits / len(tokens) < 0.6:
            out.append('the source "' + span[:40] + '" is not in FACTS - cite only a source named '
                       'in FACTS, or state the fact with no attribution')
    return out[:1]


def _safe_small(tok: str) -> bool:
    try:
        return float(tok) <= 120
    except ValueError:
        return False


def f_voice(f: Facts) -> dict:
    for title, lines in f.sections:
        if title == "CATEGORY VOICE":
            for ln in lines:
                if ln.startswith("FORBIDDEN"):
                    return {"vocab_taboo": [x.strip() for x in ln.split(":", 1)[1].split(",")]}
    return {}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


# --------------------------------------------------------------------------
# Deterministic template fallback (used when the LLM is unavailable)
# --------------------------------------------------------------------------

def _first_anchor(f: Facts) -> str:
    return f.anchors[0].split(" = ", 1)[-1] if f.anchors else ""


def template_body(f: Facts, route: dict, customer: dict | None, trigger: dict) -> tuple[str, str]:
    """A real-numbers-only fallback message. Never generic, never fabricated."""
    payload = (trigger or {}).get("payload") or {}
    kind = route.get("kind", "")
    item = f.digest_item or {}
    offer = f.active_offers[0] if f.active_offers else ""
    name = f.salutation

    if route.get("send_as") == "merchant_on_behalf" and customer:
        c_name = ((customer.get("identity") or {}).get("name") or "there").split(" (")[0]
        rel = customer.get("relationship") or {}
        biz = f.merchant_name
        slots = payload.get("available_slots") or payload.get("next_session_options") or []
        slot_labels = [s.get("label") for s in slots if isinstance(s, dict) and s.get("label")]
        bits = ["Hi " + c_name + ", " + biz + " here."]
        if kind == "chronic_refill_due" and payload.get("molecule_list"):
            bits.append("Your monthly " + ", ".join(payload["molecule_list"]) +
                        " stock runs out on " + str(payload.get("stock_runs_out_iso", ""))[:10] + ".")
            if offer:
                bits.append("Same pack ready, with " + offer + " applied.")
            bits.append("Reply CONFIRM and we will dispatch to your saved address.")
            return " ".join(bits), "binary_confirm_cancel"
        if rel.get("last_visit"):
            bits.append("Our records show your last visit on " + str(rel["last_visit"]) + ".")
        if payload.get("service_due"):
            bits.append("Your " + str(payload["service_due"]).replace("_", " ") + " is now due.")
        if offer:
            bits.append(offer + " applies.")
        if slot_labels:
            bits.append("We have " + " or ".join(slot_labels[:2]) +
                        " open — reply with the one that suits you.")
            return " ".join(bits), "multi_choice_slot"
        bits.append("Reply YES and we will hold a slot that suits you.")
        return " ".join(bits), "binary_yes_no"

    bits = [name + ","]
    if item.get("title"):
        bits.append(str(item["title"]) +
                    (" (" + str(item["source"]) + ")" if item.get("source") else "") + ".")
    elif payload.get("delta_pct") is not None:
        bits.append("your " + str(payload.get("metric", "numbers")) + " moved " +
                    _pct(payload["delta_pct"]) + " week-over-week.")
    elif payload.get("competitor_name"):
        bits.append(str(payload["competitor_name"]) + " opened " +
                    _num(payload.get("distance_km", "")) + " km away" +
                    (" with " + str(payload["their_offer"]) if payload.get("their_offer") else "") + ".")
    elif payload.get("festival"):
        bits.append(str(payload["festival"]) + " lands on " + str(payload.get("date", "")) + ".")
    elif payload.get("days_since_expiry"):
        bits.append("your plan lapsed " + _num(payload["days_since_expiry"]) +
                    " days ago and the profile work has been paused since.")
    elif payload.get("days_since_last_merchant_message"):
        bits.append("we last spoke " + _num(payload["days_since_last_merchant_message"]) +
                    " days ago about " +
                    str(payload.get("last_topic", "your profile")).replace("_", " ") + ".")
    elif payload.get("milestone_value"):
        bits.append("you are at " + _num(payload.get("value_now", "")) + " reviews, " +
                    _num(float(payload["milestone_value"]) - float(payload.get("value_now", 0))) +
                    " short of " + _num(payload["milestone_value"]) + ".")
    elif payload.get("renewal_amount"):
        bits.append("your " + str(payload.get("plan", "")) + " plan renews in " +
                    _num(payload.get("days_remaining", "")) + " days at Rs " +
                    _num(payload["renewal_amount"]) + ".")
    elif payload.get("estimated_uplift_pct"):
        bits.append("your Google listing is still unverified — verified listings in this category "
                    "see about " + _pct(payload["estimated_uplift_pct"]).lstrip("+") + " more "
                    "discovery.")
    elif payload.get("theme"):
        bits.append(str(payload["theme"]).replace("_", " ") + " came up in " +
                    _num(payload.get("occurrences_30d", "")) + " reviews this month" +
                    (' — "' + str(payload["common_quote"]) + '"' if payload.get("common_quote")
                     else "") + ".")
    elif payload.get("trends"):
        bits.append("demand is shifting this season: " +
                    ", ".join(str(t).replace("_", " ") for t in payload["trends"][:3]) + ".")
    elif payload.get("intent_topic"):
        bits.append("picking up the " + str(payload["intent_topic"]).replace("_", " ") +
                    " you asked about.")
    elif payload.get("ask_template"):
        bits.append("one quick question about this week.")
    elif payload.get("credits"):
        bits.append("a CDE session worth " + _num(payload["credits"]) +
                    " credits is open for booking.")
    elif payload.get("season_note"):
        bits.append("this week's dip is the " + str(payload["season_note"]).replace("_", " ") +
                    ", not a problem with your listing.")
    else:
        bits.append("one " + str(kind).replace("_", " ") + " item worth two minutes.")

    perf = _perf_sentence(f)
    if perf:
        bits.append(perf)
    if offer:
        bits.append("Your " + offer + " is the strongest thing to put behind it.")
    bits.append("Want me to draft it for your approval?")
    return " ".join(bits), "open_ended"


def _perf_sentence(f: Facts) -> str:
    """A readable performance sentence instead of a raw fact line."""
    ctr = views = ""
    for title, lines in f.sections:
        if not title.startswith("PERFORMANCE"):
            continue
        for ln in lines:
            if ln.startswith("click-through rate") and not ctr:
                body = ln.split(":", 1)[1].strip()
                body = body.replace(" vs category peer average ", " against a peer average of ")
                ctr = "Your click-through rate is " + body + "."
            if ln.startswith("profile views") and not views:
                views = "You had " + ln.split(":", 1)[1].strip() + " profile views in the last 30 days."
    return ctr or views


# --------------------------------------------------------------------------
# The composition entry point
# --------------------------------------------------------------------------

def compose_message(category: dict, merchant: dict, trigger: dict, customer: dict | None = None,
                    now_iso: str = "", deadline: float | None = None,
                    banned_bodies: set | None = None, cache_salt: str = "") -> dict:
    """Compose one message. Returns body/cta/send_as/suppression_key/rationale (+ template info)."""
    deadline = deadline if deadline is not None else time.monotonic() + 20.0
    route = route_for(trigger, bool(customer))
    send_as = route["send_as"]
    f = build_facts(category, merchant, trigger, customer, send_as, now_iso)
    system = SYSTEM_CUSTOMER if send_as == "merchant_on_behalf" else SYSTEM_MERCHANT

    cache_key = "|".join([
        (merchant or {}).get("merchant_id", "?"), (trigger or {}).get("id", "?"),
        (customer or {}).get("customer_id", "-"), cache_salt,
    ])

    body = cta = rationale = ""
    source = "template"
    problems: list[str] = []
    for attempt in range(2):
        if not CLIENT.enabled or time.monotonic() > deadline - 2.0:
            break
        prompt = build_prompt(f, route, problems if attempt else None)
        out = CLIENT.complete_json(system, prompt, cache_key + "|a" + str(attempt), deadline)
        if not out:
            break
        cand_body = sanitize_body(str(out.get("body", "")).strip())
        cand_cta = str(out.get("cta", route["cta"])).strip() or route["cta"]
        cand_rat = str(out.get("rationale", "")).strip()
        problems = validate(cand_body, cand_cta, f, route, banned_bodies)
        if not problems:
            body, cta, rationale, source = cand_body, cand_cta, cand_rat, "llm"
            break
        # keep the best-effort candidate in case the retry also fails
        if attempt == 0 and not _hard_fail(problems):
            body, cta, rationale, source = cand_body, cand_cta, cand_rat, "llm_retry_soft"

    if not body:
        body, cta = template_body(f, route, customer, trigger)
        body = sanitize_body(body)
        rationale = ("Deterministic fallback composed from this merchant's own figures and the "
                     "matched category item; no LLM output was available.")
        source = "template"
    elif source != "llm":
        rationale = rationale or "Composed from the merchant's own figures and the trigger."

    if not rationale:
        rationale = ("Anchored on " + (_first_anchor(f) or "the merchant's own numbers") +
                     "; single low-friction CTA for a fast reply.")

    return {
        "body": body,
        "cta": cta or route["cta"],
        "send_as": send_as,
        "suppression_key": (trigger or {}).get("suppression_key") or
                           (str((trigger or {}).get("kind", "msg")) + ":" +
                            str((merchant or {}).get("merchant_id", ""))),
        "rationale": rationale,
        "template_name": route["template"],
        "template_params": _template_params(body, f),
        "composed_by": source,
        "trigger_kind": route.get("kind", ""),
    }


def _hard_fail(problems: list[str]) -> bool:
    hard = ("URL", "phone number", "forbidden phrase", "empty", "never mention magicpin",
            "already sent", "not in FACTS")
    return any(any(h in p for h in hard) for p in problems)


def _template_params(body: str, f: Facts) -> list:
    """Params for the pre-approved first-outbound WhatsApp template."""
    sents = sentences(body)
    return [f.salutation, " ".join(sents[:-1]) if len(sents) > 1 else body,
            sents[-1] if sents else ""]
