"""
Fair Housing screen for generated leasing copy.

WHY THIS IS A SEPARATE, DETERMINISTIC MODULE
--------------------------------------------
The Fair Housing Act (42 U.S.C. §3604(c)) makes it unlawful to "make, print, or
publish ... any notice, statement, or advertisement" indicating a preference,
limitation, or discrimination based on a protected class. That applies to
MARKETING COPY, not just leasing decisions — the liability attaches to the
sentence, not to whether anyone was actually denied a unit.

An LLM writing leasing copy will produce "perfect for young professionals"
unprompted, because that phrasing saturates real estate listings in its training
data. So the screen cannot live in the prompt. It runs AFTER generation, it is
deterministic, and the model cannot talk it out of a finding.

FAIL-CLOSED, DON'T SCRUB
------------------------
An earlier version deleted the offending phrase and shipped the rest. That's the
wrong instinct. If the generator produced steering language, the *intent* of the
message is suspect, not just the wording — and a deleted phrase leaves broken
grammar that reads as sloppy at best. Findings at BLOCK severity stop the send
and demand regeneration or human review. Nothing gets silently laundered.

Protected classes (federal): race, color, religion, sex (incl. sexual orientation
and gender identity per HUD 2021), familial status, national origin, disability.
State/local law commonly adds source of income, age, marital status, veteran
status — configurable below.

This lexicon is a floor, not a ceiling. It catches known-bad phrasings; it cannot
catch novel steering. Pair it with human review of a sample and, ideally, an
LLM-based second opinion that is allowed to flag but never to clear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Severity(str, Enum):
    BLOCK = "block"    # do not send; regenerate or escalate to human
    REVIEW = "review"  # context-dependent; hold for a human
    WARN = "warn"      # legal but sloppy; log it


@dataclass(frozen=True)
class Rule:
    pattern: str              # regex, matched case-insensitively on word boundaries
    protected_class: str
    severity: Severity
    why: str                  # plain English, shown to a human reviewer
    suggest: Optional[str] = None


# --------------------------------------------------------------------------
# The lexicon. Grouped by protected class so a finding can name the exposure.
# --------------------------------------------------------------------------

RULES: List[Rule] = [
    # ---- familial status (the most common violation in leasing copy) -------
    Rule(r"\bno (?:kids|children)\b", "familial status", Severity.BLOCK,
         "States a limitation on families with children.", "Describe the unit, not who may live in it."),
    Rule(r"\b(?:adults?[- ]only|child[- ]free|childless)\b", "familial status", Severity.BLOCK,
         "Excludes families with children.",
         "Only lawful for verified HOPA 55+/62+ housing — and then it must be stated as such."),
    Rule(r"\b(?:perfect|ideal|great|suited) for (?:singles|couples|young professionals|bachelors?|empty nesters)\b",
         "familial status", Severity.BLOCK,
         "Indicates a preference for a household type — classic steering language.",
         "Describe the amenity: 'studio layout with a dedicated work nook'."),
    Rule(r"\b(?:family[- ]friendly|ideal for families|great for families)\b", "familial status", Severity.BLOCK,
         "Preference for families is unlawful even though it sounds positive. "
         "The Act prohibits preference, not just exclusion.",
         "Name the feature instead: 'playground and fenced courtyard'."),
    Rule(r"\bmature (?:tenants?|residents?|community)\b", "familial status / age", Severity.BLOCK,
         "Proxy for excluding children or younger renters."),
    Rule(r"\b(?:no more than|maximum of|limit of) \w+ (?:people|persons|occupants) per (?:bedroom|unit)\b",
         "familial status", Severity.REVIEW,
         "Occupancy limits are lawful only if they track a legitimate local code. "
         "Overly restrictive limits function as a ban on children."),

    # ---- religion ---------------------------------------------------------
    Rule(r"\b(?:christian|catholic|muslim|jewish|hindu|buddhist)\b", "religion", Severity.BLOCK,
         "Names a religion in marketing copy."),
    Rule(r"\b(?:near|close to|walk(?:ing distance)? to|steps from|minutes from|next to)\s+"
         r"(?:the\s+)?(?:church|mosque|synagogue|temple)\b",
         "religion", Severity.REVIEW,
         "Naming houses of worship as a selling point signals religious preference. "
         "Listing all nearby landmarks neutrally is defensible; cherry-picking one is not."),
    Rule(r"\b(?:christmas|easter|ramadan|hanukkah) (?:special|move[- ]in)\b", "religion", Severity.REVIEW,
         "Holiday-specific promotions can signal religious preference.",
         "Use a neutral date range: 'December move-in special'."),

    # ---- disability -------------------------------------------------------
    Rule(r"\b(?:handicapped?|crippled?|invalid)\b", "disability", Severity.BLOCK,
         "Outdated, derogatory framing.", "Use 'accessible' — e.g. 'accessible parking'."),
    Rule(r"\b(?:able[- ]bodied|must be able to|no wheelchairs?)\b", "disability", Severity.BLOCK,
         "Imposes a physical-ability requirement."),
    Rule(r"\b(?:no (?:service|emotional support) animals?|no assistance animals?)\b", "disability", Severity.BLOCK,
         "Service and assistance animals are not pets. A blanket no-pets policy "
         "cannot be applied to them, and saying so in copy is a per-se violation."),
    Rule(r"\b(?:walking distance|must climb|stairs only)\b", "disability", Severity.WARN,
         "Ability-assuming phrasing.", "'0.3 miles from the station' states the fact without the assumption."),

    # ---- race / color / national origin ------------------------------------
    Rule(r"\b(?:no section 8|no vouchers?|no housing assistance)\b", "source of income / race (proxy)",
         Severity.BLOCK,
         "Source-of-income discrimination is banned outright in many states and cities, "
         "and has disparate impact by race — a live federal theory."),
    Rule(r"\b(?:english[- ]speaking|english speakers only|no immigrants|american(?:s)? only)\b",
         "national origin", Severity.BLOCK, "Language or origin requirement."),
    Rule(r"\b(?:integrated|traditional|exclusive|desirable) (?:neighborhood|community|area)\b",
         "race / color", Severity.BLOCK,
         "Coded neighborhood language. 'Exclusive' and 'traditional' are long-recognised "
         "racial steering proxies in HUD enforcement."),
    Rule(r"\b(?:safe|good|nice|quiet) (?:neighborhood|neighbourhood|area|part of town)\b",
         "race / color (proxy)", Severity.BLOCK,
         "Characterising the surrounding neighborhood steers by demographics rather than "
         "describing the property. This is the single most common steering phrase in leasing copy.",
         "Say nothing about the neighborhood's character. Describe the property, or state "
         "verifiable facts ('gated garage, on-site staff 7am–7pm')."),

    # ---- sex / gender ------------------------------------------------------
    Rule(r"\b(?:male|female|men|women|bachelor)s? only\b", "sex", Severity.BLOCK,
         "Sex-restricted housing. Narrow exceptions exist for shared living spaces; "
         "marketing copy is not where you rely on them."),

    # ---- catch-all preference framing --------------------------------------
    Rule(r"\b(?:we (?:prefer|are looking for)|looking for the right (?:type|kind) of)\b",
         "any", Severity.REVIEW,
         "Any sentence about the kind of PERSON wanted, rather than the property offered, "
         "is a preference statement. Screen it."),
]

COMPILED = [(re.compile(r.pattern, re.I), r) for r in RULES]


@dataclass
class Finding:
    phrase: str
    protected_class: str
    severity: Severity
    why: str
    suggest: Optional[str]
    span: tuple

    def __str__(self) -> str:
        s = (f"fair_housing[{self.severity.value}] {self.protected_class}: "
             f"{self.phrase!r} — {self.why}")
        if self.suggest:
            s += f" Suggested: {self.suggest}"
        return s


def check(text: str) -> List[Finding]:
    """Screen a message. Returns every finding; empty list means it passed."""
    out: List[Finding] = []
    for rx, rule in COMPILED:
        for m in rx.finditer(text or ""):
            out.append(Finding(
                phrase=m.group(0), protected_class=rule.protected_class,
                severity=rule.severity, why=rule.why, suggest=rule.suggest,
                span=m.span(),
            ))
    return out


def blocks(text: str) -> List[Finding]:
    """Findings severe enough to stop the send."""
    return [f for f in check(text) if f.severity is Severity.BLOCK]


def is_sendable(text: str) -> bool:
    return not blocks(text)


def report(text: str) -> str:
    """Human-readable screen result — what a reviewer sees."""
    fs = check(text)
    if not fs:
        return "PASS — no fair housing findings."
    lines = [f"{'BLOCKED' if blocks(text) else 'REVIEW'} — {len(fs)} finding(s):"]
    for f in fs:
        lines.append(f"  [{f.severity.value.upper():6}] {f.protected_class:32} {f.phrase!r}")
        lines.append(f"           {f.why}")
        if f.suggest:
            lines.append(f"           → {f.suggest}")
    return "\n".join(lines)


if __name__ == "__main__":
    samples = [
        "Hi Taylor—welcome to Oak Ridge! Book a tour Thursday or Friday. Reply STOP to opt out.",
        "Beautiful 2BR in a safe neighborhood, perfect for young professionals. No kids.",
        "Newly renovated units. Walking distance to the church. No section 8.",
        "Accessible parking, elevator to all floors, service animals welcome.",
    ]
    for s in samples:
        print(f"\n> {s}\n{report(s)}")
