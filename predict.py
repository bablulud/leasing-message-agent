"""
Hold-out runner / exporter.

Built for the live case: they hand you N unseen records, you run them and paste
results back. Works whether or not `expected` is present. Never guesses silently —
low-confidence decisions are flagged so you can talk to them instead of defending them.

    python3 predict.py holdout.jsonl                      # table to stdout + all files
    python3 predict.py holdout.jsonl --format jsonl       # paste-ready JSONL
    python3 predict.py holdout.jsonl --format md          # paste-ready markdown table
    python3 predict.py holdout.jsonl --format csv         # for a spreadsheet
    python3 predict.py holdout.jsonl --train sample.jsonl # exemplars from the labelled set
    python3 predict.py holdout.jsonl --explain            # + decision trace per record

Exemplars default to --train (the labelled records you already have). If the
hold-out itself carries labels, they are NEVER used as exemplars for their own
record — leave-one-out only.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import datetime
from typing import Any, Dict, List

from agent import MessagingAgent, load_jsonl


def confidence(record: dict, d) -> List[str]:
    """Reasons a human should look at this one before shipping it."""
    flags = []
    inp = record.get("input", {}) or {}
    known_intents = {"schedule_tour", "renew_lease", "complete_application", "pay_rent", "submit_maintenance"}
    if d.should_send:
        if (d.cta or {}).get("type") not in known_intents:
            flags.append(f"unseen CTA intent {(d.cta or {}).get('type')!r} — copy is generic")
        if (inp.get("language") or "en").split("-")[0] not in {"en", "es", "fr"}:
            flags.append(f"language {inp.get('language')!r} not covered by offline templates")
        if d.channel not in {"sms", "email"}:
            flags.append(f"channel {d.channel!r} has no exemplar in the training set")
        if d.backend == "offline":
            flags.append("offline composer (no ANTHROPIC_API_KEY) — wording is retrieval-based")
    if d.violations:
        flags.append(f"{len(d.violations)} unresolved constraint violation(s)")
    return flags


def to_output(record: dict, d) -> Dict[str, Any]:
    """The shape the harness expects back: mirrors `expected` in the input file."""
    out: Dict[str, Any] = {"task_id": d.task_id}
    if d.should_send:
        out["next_message"] = {
            "channel": d.channel,
            "send_at": d.send_at,
            "subject": d.subject,
            "body": d.body,
            "cta": d.cta,
        }
    else:
        out["next_message"] = None
    out["next_action"] = d.next_action
    out["rationale"] = d.reason
    return out


def fmt_md(rows: List[Dict[str, Any]]) -> str:
    cols = ["task_id", "send?", "channel", "send_at", "subject", "body", "cta", "next_action", "why"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        m = r["out"].get("next_message") or {}
        cell = lambda x: (str(x).replace("\n", "<br>").replace("|", "\\|") if x not in (None, "") else "—")
        lines.append("| " + " | ".join([
            r["out"]["task_id"], "✅" if m else "🚫 suppress", cell(m.get("channel")),
            cell(m.get("send_at")), cell(m.get("subject")), cell(m.get("body")),
            cell(json.dumps(m.get("cta")) if m.get("cta") else None),
            cell(json.dumps(r["out"]["next_action"])), cell(r["out"]["rationale"]),
        ]) + " |")
    return "\n".join(lines)


def fmt_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["task_id", "should_send", "channel", "send_at", "subject", "body", "cta",
                "next_action", "rationale", "flags"])
    for r in rows:
        m = r["out"].get("next_message") or {}
        w.writerow([
            r["out"]["task_id"], bool(m), m.get("channel", ""), m.get("send_at", ""),
            m.get("subject") or "", m.get("body") or "",
            json.dumps(m.get("cta")) if m.get("cta") else "",
            json.dumps(r["out"]["next_action"]), r["out"]["rationale"], "; ".join(r["flags"]),
        ])
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="hold-out JSONL")
    ap.add_argument("--train", default="sample.jsonl", help="labelled records used as exemplars")
    ap.add_argument("--backend", default="auto", choices=["auto", "offline", "anthropic"])
    ap.add_argument("--format", default="all", choices=["all", "jsonl", "md", "csv", "table"])
    ap.add_argument("--now", default=None, help="ISO run tick (default: latest last_interaction seen)")
    ap.add_argument("--explain", action="store_true", help="print the decision trace per record")
    a = ap.parse_args()

    holdout = load_jsonl(a.path)
    try:
        train = load_jsonl(a.train)
    except FileNotFoundError:
        train = []
        print(f"warn: no --train file at {a.train}; composing from hold-out context only", file=sys.stderr)

    stamps = [
        datetime.fromisoformat(r["input"]["last_interaction"].replace("Z", "+00:00"))
        for r in holdout + train if (r.get("input") or {}).get("last_interaction")
    ]
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else (max(stamps) if stamps else datetime.now())

    rows = []
    for r in holdout:
        # exemplars = labelled train set + any OTHER labelled hold-out records. Never itself.
        ex = train + [x for x in holdout if x.get("task_id") != r.get("task_id") and x.get("expected")]
        d = MessagingAgent(ex, backend=a.backend, now=now).run(r)
        rows.append({"out": to_output(r, d), "flags": confidence(r, d), "decision": d})

    if a.format in ("all", "jsonl"):
        with open("predictions.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r["out"]) + "\n")
    if a.format in ("all", "md"):
        open("predictions.md", "w").write(fmt_md(rows))
    if a.format in ("all", "csv"):
        open("predictions.csv", "w").write(fmt_csv(rows))

    if a.format == "jsonl":
        print("\n".join(json.dumps(r["out"]) for r in rows)); return
    if a.format == "md":
        print(fmt_md(rows)); return
    if a.format == "csv":
        print(fmt_csv(rows)); return

    # console table
    print(f"\n{len(rows)} records · run tick {now.isoformat()} · composer {rows[0]['decision'].backend if rows else '—'}\n")
    print(f"{'TASK':<30}{'SEND':<7}{'CH':<7}{'SEND_AT':<28}CTA")
    print("-" * 100)
    for r in rows:
        m = r["out"].get("next_message") or {}
        print(f"{r['out']['task_id']:<30}{'yes' if m else 'NO':<7}{(m.get('channel') or '—'):<7}"
              f"{(m.get('send_at') or '—'):<28}{(m.get('cta') or {}).get('type', '—')}")
        if r["flags"]:
            for fl in r["flags"]:
                print(f"{'':<30}⚠ {fl}")
        if a.explain:
            print(f"{'':<30}· why: {r['out']['rationale']}")
            print(f"{'':<30}· states: {', '.join(r['decision'].states)}")
            if r["decision"].repairs:
                print(f"{'':<30}· repairs: {'; '.join(r['decision'].repairs)}")
            if m.get("body"):
                print(f"{'':<30}· body: {m['body'][:100]}...")
    sup = sum(1 for r in rows if not (r["out"].get("next_message")))
    flagged = sum(1 for r in rows if r["flags"])
    print("-" * 100)
    print(f"{len(rows)-sup} send · {sup} suppressed · {flagged} flagged for review")
    print("\nwrote predictions.jsonl, predictions.md, predictions.csv")


if __name__ == "__main__":
    main()
