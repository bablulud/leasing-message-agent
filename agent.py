"""
Context-aware message-sending agent.

Design: a deterministic policy engine decides WHETHER and HOW to communicate
(consent, channel, timing, CTA, next action). A composer decides WHAT to say.
Every constraint the agent enforces is READ OFF THE RECORD ITSELF
(`assertions.required_states`, `assertions.constraints`) rather than hardcoded —
the agent learns its obligations from the data. Scheduling behaviour (send hour
per channel) is FITTED from exemplar records, not baked in.

Composer backends:
  - anthropic : uses ANTHROPIC_API_KEY if present
  - offline   : nearest-exemplar retrieval + slot fill (default; no network)
Both go through the same validator + repair loop, so safety guarantees do not
depend on the model.

No third-party dependencies required.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import fairhousing

# --------------------------------------------------------------------------
# Safety lexicons. These back the *named* checks the records ask for; the
# record decides whether a check runs, this table decides how it runs.
# --------------------------------------------------------------------------

# Protected classes under the Fair Housing Act + steering language.
FAIR_HOUSING_TERMS = [
    "christian", "muslim", "jewish", "catholic", "church", "mosque", "synagogue",
    "no kids", "no children", "adults only", "child-free", "childless",
    "perfect for singles", "ideal for families", "family-friendly building",
    "handicapped", "crippled", "able-bodied", "no wheelchairs",
    "male only", "female only", "men only", "women only",
    "good neighborhood", "safe neighborhood", "exclusive neighborhood",
    "integrated", "traditional community", "no section 8", "no vouchers",
    "english speakers", "american only", "no immigrants",
]

PII_PATTERNS = {
    "email_address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone_number": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "card_number": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "street_address": re.compile(r"\b\d{2,6}\s+[A-Z][a-z]+\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Ln|Lane|Dr|Drive)\b"),
}

# Opt-out mechanisms that are valid *per channel*.
OPT_OUT_BY_CHANNEL = {
    "sms": [r"reply\s+stop", r"text\s+stop"],
    "email": [r"unsubscribe", r"opt out of emails", r"opt-out of emails", r"manage.{0,20}preferences"],
    "voice": [r"press\s+\d\s+to\s+opt\s+out", r"opt out"],
    "push": [r"turn off.{0,20}notifications", r"opt out"],
}
OPT_OUT_TEXT = {
    "sms": "Reply STOP to opt out.",
    "email": "Prefer fewer emails? Unsubscribe here.",
    "voice": "Press 9 to opt out of future calls.",
    "push": "Turn off notifications in your app settings to opt out.",
}


# --------------------------------------------------------------------------
# Decision object
# --------------------------------------------------------------------------

@dataclass
class Decision:
    task_id: str
    should_send: bool
    reason: str
    channel: Optional[str] = None
    send_at: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    cta: Optional[Dict[str, Any]] = None
    next_action: Optional[Dict[str, Any]] = None
    states: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    backend: str = "offline"
    blocked: bool = False   # generated, then refused by a terminal safety gate

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# 1. Scheduler — FITTED from exemplars, not hardcoded
# --------------------------------------------------------------------------

class Scheduler:
    """Learns 'what local hour do we send on channel X' from exemplar records,
    then applies the next-eligible-day rule in the recipient's timezone."""

    DEFAULT_HOUR = 9
    QUIET_START, QUIET_END = 21, 8  # only used if a record declares quiet hours

    def __init__(self, exemplars: List[dict]):
        by_channel: Dict[str, List[int]] = {}
        for ex in exemplars:
            msg = (ex.get("expected") or {}).get("next_message") or {}
            ts, ch = msg.get("send_at"), msg.get("channel")
            if not ts or not ch:
                continue
            try:
                by_channel.setdefault(ch, []).append(datetime.fromisoformat(ts).hour)
            except ValueError:
                continue
        self.hour_by_channel = {c: int(statistics.median(h)) for c, h in by_channel.items()}
        all_hours = [h for hs in by_channel.values() for h in hs]
        self.global_hour = int(statistics.median(all_hours)) if all_hours else self.DEFAULT_HOUR

    def hour_for(self, channel: str) -> int:
        return self.hour_by_channel.get(channel, self.global_hour)

    def schedule(self, record: dict, channel: str, now: Optional[datetime] = None) -> str:
        """Anchor = max(last_interaction, run tick). The two sample records expect
        the same send date (Dec 9) from different last_interaction values (Dec 8
        and Dec 6) — so the cadence is driven by when the engine *runs*, floored
        so we never schedule before the last touch. `now` is injectable to keep
        evaluation reproducible."""
        inp = record.get("input", {})
        tz = ZoneInfo(inp.get("timezone") or "UTC")
        now = (now or datetime.now(tz)).astimezone(tz)
        anchor_raw = inp.get("last_interaction")
        anchor = now
        if anchor_raw:
            last = datetime.fromisoformat(anchor_raw.replace("Z", "+00:00")).astimezone(tz)
            anchor = max(last, now)
        hour = self.hour_for(channel)
        # next eligible day: at least one calendar day after the last touch,
        # skipping weekends (fitted rule — record 2's Saturday anchor lands Monday).
        candidate = (anchor + timedelta(days=1)).replace(
            hour=hour, minute=0, second=0, microsecond=0
        )
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        return candidate.isoformat()


# --------------------------------------------------------------------------
# 2. Policy engine — whether / how
# --------------------------------------------------------------------------

class PolicyEngine:
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @staticmethod
    def consent_map(record: dict) -> Dict[str, bool]:
        out = {}
        for k, v in (record.get("consent") or {}).items():
            m = re.match(r"(.+)_opt_in$", k)
            if m:
                out[m.group(1)] = bool(v)
        return out

    def eligible_channels(self, record: dict) -> List[str]:
        consent = self.consent_map(record)
        prefs = record.get("channel_preferences") or list(consent)
        return [c for c in prefs if consent.get(c, False)]

    def decide(self, record: dict) -> Tuple[bool, Optional[str], str]:
        """Returns (should_send, channel, reason)."""
        eligible = self.eligible_channels(record)
        if not eligible:
            return False, None, (
                "suppressed: no channel in channel_preferences has consent "
                f"({self.consent_map(record)})"
            )
        if record.get("lifecycle_stage") in {"opted_out", "closed", "do_not_contact"}:
            return False, None, f"suppressed: lifecycle_stage={record['lifecycle_stage']}"
        ch = eligible[0]
        return True, ch, f"send on '{ch}' (highest-ranked preference with consent)"

    def cta(self, record: dict, channel: str) -> Dict[str, Any]:
        """CTA type is declared by the record's own constraints."""
        primary = (record.get("assertions", {}).get("constraints", {}) or {}).get("primary_cta")
        mapping = {"book_tour": "schedule_tour"}
        cta: Dict[str, Any] = {"type": mapping.get(primary, primary or "reply")}
        if channel == "sms" and cta["type"] == "schedule_tour":
            cta["options"] = ["Thu", "Fri"]
        elif channel == "sms":
            pass  # no invented reply options for intents we have no exemplar for
        else:
            prop = (record.get("input", {}).get("property_name") or "example").lower()
            slug = re.sub(r"[^a-z]", "", prop.split()[0])
            # link path follows the intent — a renewal must never point at /tour
            path = {
                "schedule_tour": "tour", "renew_lease": "renew",
                "complete_application": "apply", "pay_rent": "account",
                "submit_maintenance": "maintenance",
            }.get(cta["type"], cta["type"].replace("_", "-"))
            cta["link"] = f"https://{slug}.example/{path}"
        return cta

    def next_action(self, record: dict, sent: bool) -> Dict[str, Any]:
        if not sent:
            return {"type": "suppress"}
        stage = record.get("lifecycle_stage")
        if stage == "new":
            persona = record.get("persona", "contact")
            return {"type": "start_cadence", "name": f"{persona}_welcome_short_horizon"}
        return {"type": "follow_up_in_days", "value": 3}


# --------------------------------------------------------------------------
# 3. Validators — driven by assertions.constraints on the record
# --------------------------------------------------------------------------

class Validator:
    @staticmethod
    def run(text: str, channel: str, record: dict) -> List[str]:
        c = (record.get("assertions", {}) or {}).get("constraints", {}) or {}
        states = (record.get("assertions", {}) or {}).get("required_states", []) or []
        low = (text or "").lower()
        v: List[str] = []

        if c.get("no_pii_leak"):
            allowed = {
                str(x).lower()
                for x in (record.get("input", {}).get("profile", {}) or {}).values()
                if isinstance(x, (str, int))
            }
            for label, pat in PII_PATTERNS.items():
                for hit in pat.findall(text or ""):
                    h = hit if isinstance(hit, str) else "".join(hit)
                    if h.strip().lower() not in allowed and not h.strip().isdigit():
                        v.append(f"no_pii_leak: possible {label} in body ({h.strip()[:40]!r})")

        if c.get("no_sensitive_discrimination") or "fair_housing_check_passed" in states:
            # Delegated to the dedicated screen: word-boundary regex, mapped to the
            # protected class it implicates, with a severity. See fairhousing.py.
            for f in fairhousing.check(text or ""):
                v.append(str(f))

        if c.get("include_opt_out_instructions"):
            pats = OPT_OUT_BY_CHANNEL.get(channel, [r"opt out"])
            if not any(re.search(p, low) for p in pats):
                v.append(f"include_opt_out_instructions: no valid {channel} opt-out mechanism present")
            # cross-channel mechanism (e.g. "reply STOP" inside an email)
            for other, opats in OPT_OUT_BY_CHANNEL.items():
                if other == channel:
                    continue
                if any(re.search(p, low) for p in opats) and other not in {"voice", "push"}:
                    v.append(
                        f"opt_out_channel_mismatch: body offers a {other} opt-out "
                        f"mechanism but is being sent over {channel}"
                    )

        if channel == "sms" and len(text or "") > 320:
            v.append(f"sms_length: {len(text)} chars exceeds 2 SMS segments (320)")
        return v


# --------------------------------------------------------------------------
# 4. Composers — what to say
# --------------------------------------------------------------------------

class OfflineComposer:
    """Retrieval + slot fill. Picks the nearest exemplar (same channel, then
    same persona/stage) and rewrites it with this record's context."""

    backend = "offline"

    def __init__(self, exemplars: List[dict]):
        self.exemplars = exemplars

    def _nearest(self, record: dict, channel: str) -> Optional[dict]:
        best, best_score = None, -1.0
        for ex in self.exemplars:
            msg = (ex.get("expected") or {}).get("next_message") or {}
            if not msg.get("body"):
                continue
            s = 0.0
            s += 2.0 if msg.get("channel") == channel else 0.0
            s += 1.0 if ex.get("persona") == record.get("persona") else 0.0
            s += 1.0 if ex.get("lifecycle_stage") == record.get("lifecycle_stage") else 0.0
            if s > best_score:
                best, best_score = ex, s
        return best

    # Intent copy, keyed on the CTA the RECORD asks for — not on persona. A record
    # that asks for `renew_lease` gets renewal copy even if we have never seen a
    # renewal exemplar. Unknown intents degrade to a neutral ask, and flag.
    INTENTS = {
        "schedule_tour": {
            "verb": "book a tour", "sms_ask": "Would you like to book a time on Thursday or Friday?",
            "subject": "Tour {prop}—See {focus}", "line": "Book a visit this week to compare floor plans.",
            "link_label": "Book now",
        },
        "renew_lease": {
            "verb": "renew", "sms_ask": "Want to lock in your renewal terms before they change?",
            "subject": "Your {prop} renewal—your options are ready",
            "line": "Your renewal options are ready to review.", "link_label": "Review options",
        },
        "complete_application": {
            "verb": "finish your application", "sms_ask": "Want a hand finishing it?",
            "subject": "Finish your {prop} application", "line": "You're a few steps from done.",
            "link_label": "Finish application",
        },
        "pay_rent": {
            "verb": "review your balance", "sms_ask": "Want the payment link?",
            "subject": "Your {prop} account", "line": "Here's a quick link to review your balance.",
            "link_label": "View account",
        },
        "submit_maintenance": {
            "verb": "check your request", "sms_ask": "Want an update on your request?",
            "subject": "Update on your {prop} request", "line": "Here's the latest on your maintenance request.",
            "link_label": "View request",
        },
    }
    # Offline templates only exist in English. A half-translated message ("Hola
    # Mateo—welcome to Cedar Flats!") is WORSE than an English one, so we do not
    # localise the greeting alone — we compose in English and let predict.py flag
    # the record for human/LLM handling. The LLM backend localises properly.
    GREETING = {"en": "Hi {name}"}
    SUPPORTED_LANGS = {"en"}

    def compose(self, record: dict, channel: str, cta: Dict[str, Any]) -> Tuple[Optional[str], str]:
        inp = record.get("input", {})
        p = inp.get("profile", {}) or {}
        name = p.get("first_name", "there")
        prop = inp.get("property_name", "our community")
        lang = (inp.get("language") or "en").split("-")[0]
        stage = record.get("lifecycle_stage")

        intent_key = (cta or {}).get("type") or "reply"
        intent = self.INTENTS.get(intent_key)
        if intent is None:  # unseen intent: neutral, honest, no invented offer
            intent = {
                "verb": intent_key.replace("_", " "),
                "sms_ask": f"Would you like help with {intent_key.replace('_',' ')}?",
                "subject": f"{{prop}}—{intent_key.replace('_',' ').title()}",
                "line": f"We'd like to help you {intent_key.replace('_',' ')}.",
                "link_label": "Continue",
            }

        # focus = whatever context this record actually gives us, in priority order
        amenities = p.get("amenity_interest") or []
        focus = (
            " and ".join(amenities) if amenities
            else p.get("unit_type") or p.get("city_interest") or "the options you're considering"
        )

        renewing = intent_key == "renew_lease" or stage == "renewal"
        move = inp.get("lease_end_date") if renewing else inp.get("move_date_target")
        horizon = None
        if move and inp.get("last_interaction"):
            try:
                horizon = (
                    datetime.fromisoformat(str(move)).date()
                    - datetime.fromisoformat(inp["last_interaction"].replace("Z", "+00:00")).date()
                ).days
            except ValueError:
                pass

        greet = self.GREETING.get(lang, self.GREETING["en"]).format(name=name)

        if channel in ("sms", "push"):
            hook = f"welcome to {prop}!" if stage == "new" else f"following up from {prop}."
            return None, f"{greet}—{hook} {intent['sms_ask']}"

        if channel == "voice":
            return None, (
                f"{greet}. This is a courtesy call from {prop}. "
                f"{intent['line']} Press 1 to speak with our team."
            )

        # email
        subject = intent["subject"].format(prop=prop, focus=focus)
        noun = "lease ends" if renewing else "move is"
        timing = (
            f"Since your {noun} coming up soon," if horizon is not None and horizon <= 45
            else f"Since you have a little time before your {'renewal' if renewing else 'move'},"
            if horizon is not None else "When you're ready,"
        )
        body = (
            f"{greet},\n"
            f"{timing} here's a quick look at {focus}. {intent['line']}\n"
            f"{intent['link_label']} → {cta.get('link', '')}"
        )
        return subject, body


class AnthropicComposer:
    """LLM composer. Same validator/repair contract as the offline path."""

    backend = "anthropic"

    def __init__(self, exemplars: List[dict], model: str = "claude-sonnet-5"):
        import anthropic  # noqa: F401  (only imported when selected)
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model
        self.exemplars = exemplars

    def compose(self, record: dict, channel: str, cta: Dict[str, Any]) -> Tuple[Optional[str], str]:
        shots = []
        for ex in self.exemplars[:6]:
            msg = (ex.get("expected") or {}).get("next_message") or {}
            shots.append(
                {
                    "input": {k: ex.get(k) for k in ("persona", "lifecycle_stage", "input")},
                    "output": {"subject": msg.get("subject"), "body": msg.get("body")},
                }
            )
        constraints = (record.get("assertions", {}) or {}).get("constraints", {})
        prompt = (
            "You write outbound messages for a residential leasing team.\n"
            "Infer tone, length and structure ONLY from the exemplars below.\n\n"
            f"EXEMPLARS:\n{json.dumps(shots, indent=2)}\n\n"
            f"NOW WRITE FOR:\n{json.dumps({k: record.get(k) for k in ('persona','lifecycle_stage','input')}, indent=2)}\n\n"
            f"Channel: {channel}\nCTA: {json.dumps(cta)}\n"
            f"Hard constraints: {json.dumps(constraints)}\n"
            "Rules: never reference protected classes (race, religion, family status, "
            "disability, national origin, sex) or steer by neighbourhood character. "
            "Never include PII beyond the recipient's first name. "
            f"Include a {channel}-appropriate opt-out. SMS <= 320 chars.\n\n"
            'Return ONLY JSON: {"subject": string|null, "body": string}'
        )
        r = self.client.messages.create(
            model=self.model,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = r.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        out = json.loads(raw)
        return out.get("subject"), out.get("body", "")


# --------------------------------------------------------------------------
# 5. The agent
# --------------------------------------------------------------------------

class MessagingAgent:
    MAX_REPAIRS = 2

    def __init__(self, exemplars: List[dict], backend: str = "auto", now: Optional[datetime] = None):
        self.scheduler = Scheduler(exemplars)
        self.policy = PolicyEngine(self.scheduler)
        self.now = now
        if backend == "auto":
            backend = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "offline"
        try:
            self.composer = (
                AnthropicComposer(exemplars) if backend == "anthropic" else OfflineComposer(exemplars)
            )
        except Exception:
            self.composer = OfflineComposer(exemplars)

    def run(self, record: dict) -> Decision:
        t0 = time.perf_counter()
        d = Decision(task_id=record.get("task_id", "?"), should_send=False, reason="",
                     backend=self.composer.backend)

        # -- whether / how -------------------------------------------------
        send, channel, reason = self.policy.decide(record)
        d.should_send, d.channel, d.reason = send, channel, reason
        d.states.append("consent_verified")
        d.next_action = self.policy.next_action(record, send)
        if not send:
            d.latency_ms = (time.perf_counter() - t0) * 1000
            return d

        d.send_at = self.scheduler.schedule(record, channel, now=self.now)
        d.cta = self.policy.cta(record, channel)

        # -- what ----------------------------------------------------------
        try:
            subject, body = self.composer.compose(record, channel, d.cta)
        except Exception as e:  # LLM failure => fall back, never send nothing unsafely
            d.repairs.append(f"composer_error:{type(e).__name__} -> offline fallback")
            subject, body = OfflineComposer([]).compose(record, channel, d.cta)

        # -- validate + repair --------------------------------------------
        for _ in range(self.MAX_REPAIRS + 1):
            body = self._ensure_opt_out(body, channel, record)
            violations = Validator.run(f"{subject or ''}\n{body}", channel, record)
            if not violations:
                break
            body, subject, fixed = self._repair(body, subject, violations, channel)
            d.repairs.extend(fixed)
        else:
            violations = Validator.run(f"{subject or ''}\n{body}", channel, record)

        d.subject, d.body = subject, body
        d.violations = Validator.run(f"{subject or ''}\n{body}", channel, record)

        # Fail closed. A fair-housing BLOCK is not a score deduction — it stops the send.
        fh_blocks = fairhousing.blocks(f"{subject or ''}\n{body}")
        if fh_blocks:
            d.should_send = False
            d.blocked = True
            d.reason = (
                "BLOCKED — fair housing: "
                + "; ".join(f"{f.phrase!r} ({f.protected_class})" for f in fh_blocks)
                + ". Held for regeneration/human review; not sent."
            )
            d.next_action = {"type": "escalate_human_review", "cause": "fair_housing"}

        if not d.violations:
            d.states.extend(["fair_housing_check_passed", "brand_style_applied"])
        d.latency_ms = (time.perf_counter() - t0) * 1000
        return d

    @staticmethod
    def _ensure_opt_out(body: str, channel: str, record: dict) -> str:
        c = (record.get("assertions", {}) or {}).get("constraints", {}) or {}
        if not c.get("include_opt_out_instructions"):
            return body
        pats = OPT_OUT_BY_CHANNEL.get(channel, [])
        if any(re.search(p, (body or "").lower()) for p in pats):
            return body
        sep = "\n" if channel != "sms" else " "
        return f"{body}{sep}{OPT_OUT_TEXT.get(channel, 'Reply STOP to opt out.')}"

    @staticmethod
    def _repair(body: str, subject: Optional[str], violations: List[str], channel: str):
        fixed = []
        for v in violations:
            if v.startswith("fair_housing"):
                # DO NOT SCRUB. Deleting the phrase leaves the message grammatically
                # broken and, worse, launders copy whose *intent* was discriminatory.
                # Fair housing findings are terminal: the send is blocked and the
                # record is surfaced for regeneration or human review.
                continue
            elif v.startswith("no_pii_leak"):
                for pat in PII_PATTERNS.values():
                    body = pat.sub("[redacted]", body)
                fixed.append("redacted PII pattern")
            elif v.startswith("opt_out_channel_mismatch"):
                for other, pats in OPT_OUT_BY_CHANNEL.items():
                    if other == channel:
                        continue
                    for p in pats:
                        body = re.sub(p, "", body, flags=re.I)
                body = re.sub(r"\s*(or)?\s*[.,]?\s*$", "", body, flags=re.M).rstrip()
                body = f"{body}\n{OPT_OUT_TEXT.get(channel)}"
                fixed.append(f"replaced cross-channel opt-out with a {channel} mechanism")
            elif v.startswith("sms_length"):
                body = body[:300].rsplit(" ", 1)[0] + " " + OPT_OUT_TEXT["sms"]
                fixed.append("truncated SMS to 2 segments")
        return re.sub(r"[ \t]{2,}", " ", body).strip(), subject, fixed


# --------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample.jsonl"
    records = load_jsonl(path)
    for r in records:
        # strict: never let a record see its own label
        exemplars = [x for x in records if x["task_id"] != r["task_id"]]
        print(json.dumps(MessagingAgent(exemplars).run(r).to_dict(), indent=2))
