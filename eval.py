"""
Eval harness: runs the agent over a JSONL set and scores each decision against
`expected` — structurally (channel, timing, CTA, next action, required states,
constraints) and semantically (subject/body similarity).

Label leakage: by default each record is evaluated with LEAVE-ONE-OUT exemplars,
so the agent never sees its own expected output. Use --fewshot all to fit on the
whole labelled set (calibration mode).

    python3 eval.py sample.jsonl                 # strict, offline composer
    python3 eval.py sample.jsonl --fewshot all
    python3 eval.py sample.jsonl --backend anthropic   # needs ANTHROPIC_API_KEY

Writes results.json and report.html. Stdlib only.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent import MessagingAgent, Validator, load_jsonl

# ---------------------------------------------------------------- similarity

def _ngrams(s: str, n: int = 3) -> Counter:
    s = re.sub(r"\s+", " ", (s or "").lower().strip())
    return Counter(s[i : i + n] for i in range(max(len(s) - n + 1, 0)))


def cosine(a: str, b: str) -> float:
    """Char-trigram cosine. Model-free proxy for semantic match; swap in an
    embedding or LLM judge here if you want a stronger signal."""
    A, B = _ngrams(a), _ngrams(b)
    if not A or not B:
        return 1.0 if not A and not B else 0.0
    dot = sum(A[k] * B[k] for k in A.keys() & B.keys())
    na = math.sqrt(sum(v * v for v in A.values()))
    nb = math.sqrt(sum(v * v for v in B.values()))
    return dot / (na * nb) if na and nb else 0.0


def _coverage(text: str, record: dict) -> float:
    """Fraction of this record's context tokens present in the text."""
    inp = record.get("input", {}) or {}
    prof = inp.get("profile", {}) or {}
    tokens: List[str] = []
    for v in prof.values():
        tokens += [str(x) for x in v] if isinstance(v, list) else [str(v)]
    if inp.get("property_name"):
        tokens.append(str(inp["property_name"]))
    tokens = [t for t in tokens if len(t) > 2]
    if not tokens:
        return 1.0
    low = (text or "").lower()

    def hit(t: str) -> bool:
        t = t.lower()
        return t in low or any(w in low for w in t.split(", ")[0].split() if len(w) > 3)

    return sum(hit(t) for t in tokens) / len(tokens)


def personalization(body: str, subject: Optional[str], record: dict) -> float:
    """Context coverage, normalised against the reference message's own coverage.

    Absolute coverage is the wrong bar: the gold message for record 1 never
    mentions `city_interest` either, so an absolute threshold of 0.85 would fail
    the ground truth. We score the agent *relative to the reference* — 1.0 means
    "personalised at least as richly as the expected message".
    """
    got = _coverage(f"{subject or ''} {body or ''}", record)
    emsg = (record.get("expected") or {}).get("next_message") or {}
    gold = _coverage(f"{emsg.get('subject') or ''} {emsg.get('body') or ''}", record) if emsg else 0.0
    return min(got / gold, 1.0) if gold > 0 else got


# ---------------------------------------------------------------- scoring

def same_instant(a: Optional[str], b: Optional[str]) -> bool:
    try:
        return datetime.fromisoformat(a) == datetime.fromisoformat(b)
    except Exception:
        return a == b


def score_record(record: dict, d) -> Dict[str, Any]:
    exp = record.get("expected", {}) or {}
    emsg = exp.get("next_message") or {}
    eact = exp.get("next_action") or {}
    thr = record.get("thresholds", {}) or {}
    checks: List[Dict[str, Any]] = []

    def check(name, ok, got, want, weight=1.0, partial=None):
        checks.append(
            {
                "name": name,
                "ok": bool(ok),
                "score": 1.0 if ok else (partial or 0.0),
                "got": got,
                "want": want,
                "weight": weight,
            }
        )

    expected_send = bool(emsg)
    check("decision.should_send", d.should_send == expected_send, d.should_send, expected_send, 2.0)

    if expected_send and d.should_send:
        check("channel", d.channel == emsg.get("channel"), d.channel, emsg.get("channel"), 2.0)

        ok_t = same_instant(d.send_at, emsg.get("send_at"))
        partial = 0.0
        if not ok_t and d.send_at and emsg.get("send_at"):
            try:
                delta = abs(
                    (datetime.fromisoformat(d.send_at) - datetime.fromisoformat(emsg["send_at"])).total_seconds()
                )
                partial = 0.5 if delta <= 6 * 3600 else (0.25 if delta <= 24 * 3600 else 0.0)
            except Exception:
                pass
        check("send_at", ok_t, d.send_at, emsg.get("send_at"), 1.5, partial)

        ecta = emsg.get("cta") or {}
        check("cta.type", (d.cta or {}).get("type") == ecta.get("type"),
              (d.cta or {}).get("type"), ecta.get("type"), 1.5)

        sim_body = cosine(d.body or "", emsg.get("body") or "")
        checks.append({"name": "semantic.body", "ok": sim_body >= 0.60, "score": sim_body,
                       "got": round(sim_body, 3), "want": ">= 0.60", "weight": 2.0})

        if emsg.get("subject") or d.subject:
            sim_sub = cosine(d.subject or "", emsg.get("subject") or "")
            checks.append({"name": "semantic.subject", "ok": sim_sub >= 0.50, "score": sim_sub,
                           "got": round(sim_sub, 3), "want": ">= 0.50", "weight": 1.0})

    check("next_action.type", (d.next_action or {}).get("type") == eact.get("type"),
          (d.next_action or {}).get("type"), eact.get("type"), 1.5)

    # required states + constraints, re-verified independently of the agent
    req = (record.get("assertions", {}) or {}).get("required_states", []) or []
    missing = [s for s in req if s not in d.states]
    check("required_states", not missing, d.states, req, 2.0)

    indep = Validator.run(f"{d.subject or ''}\n{d.body or ''}", d.channel or "sms", record) if d.should_send else []
    check("constraints", not indep, indep or "clean", "no violations", 3.0)

    # thresholds declared by the record itself
    pscore = personalization(d.body or "", d.subject, record) if d.should_send else 1.0
    pmin = thr.get("personalization_score_min", 0.0)
    lat_max = thr.get("p95_latency_ms", float("inf"))
    smax = thr.get("safety_violations_max", 0)
    thresholds = [
        {"name": "personalization_score", "got": round(pscore, 3), "want": f">= {pmin}", "ok": pscore >= pmin},
        {"name": "latency_ms", "got": round(d.latency_ms, 1), "want": f"<= {lat_max}", "ok": d.latency_ms <= lat_max},
        {"name": "safety_violations", "got": len(indep), "want": f"<= {smax}", "ok": len(indep) <= smax},
    ]

    tot_w = sum(c["weight"] for c in checks)
    score = sum(min(c["score"], 1.0) * c["weight"] for c in checks) / tot_w if tot_w else 0.0
    hard = {"decision.should_send", "channel", "constraints", "required_states"}
    passed = (
        all(c["ok"] for c in checks if c["name"] in hard)
        and all(t["ok"] for t in thresholds)
        and score >= 0.75
    )
    return {
        "task_id": record.get("task_id"),
        "persona": record.get("persona"),
        "lifecycle_stage": record.get("lifecycle_stage"),
        "passed": passed,
        "score": round(score, 3),
        "checks": checks,
        "thresholds": thresholds,
        "decision": d.to_dict(),
        "expected": exp,
    }


# ---------------------------------------------------------------- dashboard

CSS = """
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e8ec;--mut:#8b93a3;
--ok:#3fb950;--bad:#f85149;--warn:#d29922;--acc:#6ea8fe}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;padding:32px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:0}
.sub{color:var(--mut);margin-bottom:24px;font-size:13px}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:28px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px;min-width:150px}
.kpi .v{font-size:26px;font-weight:600}.kpi .l{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
.hd{display:flex;align-items:center;gap:10px;margin-bottom:14px}
.pill{font-size:11px;padding:2px 8px;border-radius:20px;font-weight:600}
.pass{background:rgba(63,185,80,.15);color:var(--ok)}.fail{background:rgba(248,81,73,.15);color:var(--bad)}
.tag{background:#20242e;color:var(--mut);font-size:11px;padding:2px 8px;border-radius:20px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--mut);font-weight:500;font-size:11px;text-transform:uppercase;
letter-spacing:.05em;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:6px 8px;border-bottom:1px solid #1d2129;vertical-align:top}
tr:last-child td{border-bottom:none}
.ok{color:var(--ok)}.no{color:var(--bad)}
code{background:#0b0d11;padding:1px 5px;border-radius:4px;font-size:12px;color:#c9d1d9;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}
.msg{background:#0b0d11;border:1px solid var(--line);border-radius:8px;padding:12px}
.msg .l{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.msg pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:inherit;font-size:13px}
.bar{height:5px;background:#20242e;border-radius:3px;overflow:hidden;margin-top:8px}
.bar>div{height:100%;background:var(--acc)}
.rep{color:var(--warn);font-size:12px;margin-top:10px}
"""


def render(results: List[dict], meta: dict) -> str:
    n = len(results)
    p = sum(r["passed"] for r in results)
    avg = sum(r["score"] for r in results) / n if n else 0
    viol = sum(len(r["decision"]["violations"]) for r in results)
    lat = sorted(r["decision"]["latency_ms"] for r in results)
    p95 = lat[min(int(0.95 * len(lat)), len(lat) - 1)] if lat else 0

    def esc(x):
        return html.escape(str(x)) if x is not None else "<span style='color:#8b93a3'>—</span>"

    cards = []
    for r in results:
        d = r["decision"]
        emsg = (r["expected"] or {}).get("next_message") or {}
        rows = "".join(
            f"<tr><td><code>{esc(c['name'])}</code></td>"
            f"<td class='{'ok' if c['ok'] else 'no'}'>{'PASS' if c['ok'] else 'FAIL'}</td>"
            f"<td>{esc(c['got'])}</td><td>{esc(c['want'])}</td></tr>"
            for c in r["checks"]
        )
        trows = "".join(
            f"<tr><td><code>{esc(t['name'])}</code></td>"
            f"<td class='{'ok' if t['ok'] else 'no'}'>{'PASS' if t['ok'] else 'FAIL'}</td>"
            f"<td>{esc(t['got'])}</td><td>{esc(t['want'])}</td></tr>"
            for t in r["thresholds"]
        )
        reps = (
            f"<div class='rep'>⟳ repairs applied: {esc('; '.join(d['repairs']))}</div>"
            if d["repairs"] else ""
        )
        cards.append(f"""
<div class="card">
  <div class="hd">
    <h2>{esc(r['task_id'])}</h2>
    <span class="pill {'pass' if r['passed'] else 'fail'}">{'PASS' if r['passed'] else 'FAIL'}</span>
    <span class="tag">{esc(r['persona'])} · {esc(r['lifecycle_stage'])}</span>
    <span class="tag">{esc(d['channel'] or 'suppressed')}</span>
    <span class="tag">{esc(d['backend'])}</span>
    <span style="margin-left:auto;color:#8b93a3">score {r['score']:.2f}</span>
  </div>
  <div class="bar"><div style="width:{r['score']*100:.0f}%"></div></div>
  <div class="cols">
    <div class="msg"><div class="l">Agent output</div>
      {f"<div style='font-weight:600;margin-bottom:8px'>{esc(d['subject'])}</div>" if d['subject'] else ''}
      <pre>{esc(d['body'] or d['reason'])}</pre>
      <div style="color:#8b93a3;font-size:12px;margin-top:8px">send_at: <code>{esc(d['send_at'])}</code></div>
    </div>
    <div class="msg"><div class="l">Expected</div>
      {f"<div style='font-weight:600;margin-bottom:8px'>{esc(emsg.get('subject'))}</div>" if emsg.get('subject') else ''}
      <pre>{esc(emsg.get('body') or '(no send)')}</pre>
      <div style="color:#8b93a3;font-size:12px;margin-top:8px">send_at: <code>{esc(emsg.get('send_at'))}</code></div>
    </div>
  </div>
  {reps}
  <table style="margin-top:14px"><tr><th>Check</th><th>Result</th><th>Got</th><th>Want</th></tr>{rows}</table>
  <table style="margin-top:10px"><tr><th>Threshold</th><th>Result</th><th>Got</th><th>Want</th></tr>{trows}</table>
</div>""")

    return f"""<!doctype html><meta charset="utf-8"><title>Messaging agent — eval</title>
<style>{CSS}</style>
<h1>Messaging agent — evaluation report</h1>
<div class="sub">{esc(meta['path'])} · composer <code>{esc(meta['backend'])}</code> ·
few-shot <code>{esc(meta['fewshot'])}</code> · run tick <code>{esc(meta.get('now'))}</code></div>
<div class="kpis">
  <div class="kpi"><div class="l">Pass rate</div><div class="v">{p}/{n}</div></div>
  <div class="kpi"><div class="l">Mean score</div><div class="v">{avg:.2f}</div></div>
  <div class="kpi"><div class="l">Safety violations</div><div class="v {'no' if viol else 'ok'}">{viol}</div></div>
  <div class="kpi"><div class="l">p95 latency</div><div class="v">{p95:.0f}<span style="font-size:14px;color:#8b93a3">ms</span></div></div>
</div>
{''.join(cards)}"""


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="sample.jsonl")
    ap.add_argument("--backend", default="auto", choices=["auto", "offline", "anthropic"])
    ap.add_argument("--fewshot", default="loo", choices=["loo", "all"],
                    help="loo = leave-one-out (no label leakage); all = fit on full labelled set")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--now", default=None,
                    help="ISO run tick. Default: latest last_interaction in the set "
                         "(treats the file as one batch run, keeps results reproducible).")
    a = ap.parse_args()

    records = load_jsonl(a.path)

    if a.now:
        now = datetime.fromisoformat(a.now.replace("Z", "+00:00"))
    else:
        stamps = [
            datetime.fromisoformat(r["input"]["last_interaction"].replace("Z", "+00:00"))
            for r in records
            if (r.get("input") or {}).get("last_interaction")
        ]
        now = max(stamps) if stamps else datetime.now()

    results = []
    for r in records:
        ex = records if a.fewshot == "all" else [x for x in records if x["task_id"] != r["task_id"]]
        agent = MessagingAgent(ex, backend=a.backend, now=now)
        results.append(score_record(r, agent.run(r)))

    meta = {"path": a.path, "backend": results[0]["decision"]["backend"] if results else a.backend,
            "fewshot": a.fewshot, "now": now.isoformat()}
    with open("results.json", "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    with open(a.out, "w") as f:
        f.write(render(results, meta))

    print(f"{'TASK':<32}{'PASS':<7}{'SCORE':<8}FAILING CHECKS")
    print("-" * 92)
    for r in results:
        bad = ", ".join(c["name"] for c in r["checks"] if not c["ok"]) or "—"
        print(f"{r['task_id']:<32}{'✓' if r['passed'] else '✗':<7}{r['score']:<8.2f}{bad}")
    n = len(results)
    print("-" * 92)
    print(f"{sum(r['passed'] for r in results)}/{n} passed · "
          f"mean score {sum(r['score'] for r in results)/n:.2f} · "
          f"{sum(len(r['decision']['violations']) for r in results)} safety violations")
    print(f"\nwrote {a.out} and results.json")


if __name__ == "__main__":
    main()
