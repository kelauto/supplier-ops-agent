#!/usr/bin/env python3
"""
Supplier Ops Agent  —  e-commerce operations triage + reporting
==============================================================
Reads a supplier inbox and an order book, then produces:
  * classified email triage (quote / delay / invoice / shipping / complaint)
  * overdue-order detection against promised ship dates
  * drafted supplier follow-ups (per supplier, ready to send)
  * an operational report (HTML) + an action list (CSV)

Runs with the Python standard library only - no install, no API key.

    python3 agent.py --seed-demo        # create sample inbox + order book
    python3 agent.py                    # run the triage
    python3 agent.py --use-llm          # route classification through an LLM

The LLM path is OPTIONAL and deliberately pluggable: set OPENAI_API_KEY and
pass --use-llm. Without a key the agent uses the deterministic rule engine,
so the demo is honest - it never pretends a stub call was real inference.
"""
import argparse
import csv
import html
import json
import os
import re
import sys
from datetime import date, datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
INBOX = os.path.join(ROOT, "inbox")
ORDERS = os.path.join(ROOT, "orders.csv")

# ---------------------------------------------------------------- classification

RULES = [
    ("quote",     r"\b(quote|quotation|price|pricing|cost per|rate for|proforma)\b"),
    ("delay",     r"\b(delay|delayed|postpone|late|backlog|shortage|will not make it)\b"),
    ("invoice",   r"\b(invoice|payment|remittance|outstanding|balance due|payable)\b"),
    ("shipping",  r"\b(ship|shipped|tracking|waybill|awb|container|etd|eta|port)\b"),
    ("complaint", r"\b(damaged|defect|faulty|broken|wrong item|refund|compensation)\b"),
]

URGENT = r"\b(urgent|asap|immediately|today|escalate|deadline)\b"


def classify(text, use_llm=False):
    """Return (intent, confidence, evidence). LLM path falls back to rules."""
    if use_llm:
        try:
            return classify_llm(text)
        except Exception as e:
            print(f"  ! LLM unavailable ({e.__class__.__name__}) - using rule engine")
    low = text.lower()
    for intent, pat in RULES:
        m = re.search(pat, low)
        if m:
            return intent, 0.78, m.group(0)
    return "other", 0.40, ""


def classify_llm(text):
    """Real LLM call - only runs if OPENAI_API_KEY is set and --use-llm passed."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    import urllib.request
    body = json.dumps({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "Classify the supplier email into exactly one of: "
             "quote, delay, invoice, shipping, complaint, other. "
             'Reply with JSON only: {"intent": "...", "confidence": 0.0}'},
            {"role": "user", "content": text[:4000]},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                headers={"Authorization": f"Bearer {key}",
                                         "Content-Type": "application/json"})
    import urllib.request as u
    with u.urlopen(req, timeout=40) as r:
        out = json.load(r)
    parsed = json.loads(out["choices"][0]["message"]["content"])
    return parsed["intent"], float(parsed.get("confidence", 0.7)), "llm"


# ---------------------------------------------------------------- follow-up drafts

DRAFTS = {
    "quote": "Thanks for your email. Could you confirm your best unit price, MOQ and lead "
             "time for the quantities on our last order? We're ready to place the next one "
             "this week and will confirm on receipt.",
    "delay": "Noted on the delay - I need a firm revised ship date in writing today, plus "
             "the reason. If the new date slips our window we'll need to split the shipment "
             "and air-freight the shortfall at your cost. Please confirm by end of day.",
    "invoice": "Thanks - we have the invoice. Our finance runs payments on the 15th and 30th; "
               "please confirm the PO number matches so it clears on the next run without a query.",
    "shipping": "Please send the AWB, vessel/container number and the ETD/ETA for this "
                "shipment so we can update our tracking sheet and notify the customer.",
    "complaint": "Thank you for flagging this. Please send photos of the affected units, the "
                 "batch/lot number, and your proposal for credit or replacement. We'll confirm "
                 "within 24 hours.",
    "other": "Thanks for the update - noted. I'll come back to you if anything needs action.",
}


def detect_urgency(text):
    return bool(re.search(URGENT, text.lower()))


# ---------------------------------------------------------------- order tracking

def load_orders(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def overdue(orders, today=None):
    today = today or date.today()
    out = []
    for o in orders:
        try:
            promised = datetime.strptime(o["promised_ship"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if o.get("status", "").lower() in ("delivered", "cancelled"):
            continue
        days = (today - promised).days
        if days > 0:
            o = dict(o)
            o["days_late"] = days
            o["value_at_risk"] = float(o.get("qty", 0) or 0) * float(o.get("unit_cost", 0) or 0)
            out.append(o)
    out.sort(key=lambda r: -r["days_late"])
    return out


# ---------------------------------------------------------------- reporting

def write_actions(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["priority", "supplier", "intent", "subject", "action", "days_late", "value_at_risk"])
        for r in rows:
            w.writerow([r["priority"], r["supplier"], r["intent"], r["subject"],
                        r["action"], r.get("days_late", ""), r.get("value_at_risk", "")])
    return path


def write_report(rows, late, path):
    def prio_class(p):
        return {"HIGH": "hi", "MEDIUM": "md", "LOW": "lo"}.get(p, "lo")

    mail = "".join(f"""
      <tr>
        <td><span class="pill {prio_class(r['priority'])}">{r['priority']}</span></td>
        <td><b>{html.escape(r['supplier'])}</b><div class="mut">{html.escape(r['subject'])}</div></td>
        <td>{html.escape(r['intent'])}
            <div class="mut">{int(r['confidence']*100)}% · {html.escape(r['evidence'] or 'no keyword')}</div></td>
        <td class="draft">{html.escape(r['action'])}</td>
      </tr>""" for r in rows)

    late_rows = "".join(f"""
      <tr>
        <td>{html.escape(l['order_id'])}</td>
        <td>{html.escape(l['supplier'])}</td>
        <td>{html.escape(l['promised_ship'])}</td>
        <td class="bad">+{l['days_late']}d</td>
        <td>{html.escape(l.get('qty',''))}</td>
        <td>${l['value_at_risk']:,.0f}</td>
      </tr>""" for l in late)

    total_risk = sum(l["value_at_risk"] for l in late)
    hi = sum(1 for r in rows if r["priority"] == "HIGH")

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ops report</title><style>
 body{{margin:0;padding:26px 16px;background:#f5f6f8;color:#111827;
   font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
 .w{{max-width:1000px;margin:0 auto}}
 h1{{font-size:22px;margin:0 0 2px}} .sub{{color:#6b7280;font-size:13px;margin-bottom:20px}}
 .cards{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:22px}}
 .card{{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:14px 18px;flex:1;min-width:150px}}
 .card .n{{font-size:24px;font-weight:700}} .card .l{{color:#6b7280;font-size:12px;text-transform:uppercase;letter-spacing:.5px}}
 h2{{font-size:16px;margin:26px 0 8px}}
 table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #e5e7eb;border-radius:10px;overflow:hidden}}
 th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:#6b7280;
   padding:9px 10px;border-bottom:1px solid #e5e7eb;background:#fafbfc}}
 td{{padding:10px;border-bottom:1px solid #f1f2f4;vertical-align:top;font-size:14px}}
 .mut{{color:#6b7280;font-size:12px}} .bad{{color:#dc2626;font-weight:700}}
 .pill{{display:inline-block;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:700}}
 .pill.hi{{background:#fee2e2;color:#991b1b}} .pill.md{{background:#fef3c7;color:#92400e}}
 .pill.lo{{background:#e0f2fe;color:#075985}}
 .draft{{color:#374151;max-width:380px}}
 footer{{color:#9ca3af;font-size:12px;text-align:center;margin-top:24px}}
</style></head><body><div class="w">
 <h1>Supplier operations report</h1>
 <div class="sub">generated {date.today().isoformat()} · {len(rows)} emails triaged · {len(late)} late orders</div>
 <div class="cards">
   <div class="card"><div class="n">{len(rows)}</div><div class="l">Emails triaged</div></div>
   <div class="card"><div class="n" style="color:#dc2626">{hi}</div><div class="l">High priority</div></div>
   <div class="card"><div class="n" style="color:#dc2626">{len(late)}</div><div class="l">Overdue orders</div></div>
   <div class="card"><div class="n">${total_risk:,.0f}</div><div class="l">Value at risk</div></div>
 </div>
 <h2>Inbox triage &amp; drafted replies</h2>
 <table><thead><tr><th>Priority</th><th>Supplier</th><th>Intent</th><th>Drafted follow-up</th></tr></thead>
 <tbody>{mail}</tbody></table>
 <h2>Overdue orders</h2>
 <table><thead><tr><th>Order</th><th>Supplier</th><th>Promised</th><th>Late by</th><th>Qty</th><th>Value</th></tr></thead>
 <tbody>{late_rows or '<tr><td colspan="6" class="mut">Nothing overdue.</td></tr>'}</tbody></table>
 <footer>Supplier Ops Agent — automation sample</footer>
</div></body></html>"""


# ---------------------------------------------------------------- demo data

def seed_demo():
    os.makedirs(INBOX, exist_ok=True)
    mails = [
        ("supplier_shenzen.txt", "Shenzhen Yuan Trading",
         "Re: PO-1042 quotation request\n\nHi, please find our updated pricing. Unit cost is "
         "$4.20 at MOQ 500. Lead time 25 days. Payment 30% deposit. This quote is valid 14 days."),
        ("supplier_istanbul.txt", "Istanbul Textiles",
         "Shipment update\n\nWe have shipped 3 cartons today. AWB 176-44219955. ETD 12 Oct, "
         "ETA Lagos 19 Oct. Please confirm the consignee address is unchanged."),
        ("supplier_mumbai.txt", "Mumbai Components",
         "URGENT - production delay\n\nDue to a raw material shortage our line is delayed by "
         "9 days. We may not make the 15 Oct deadline. Please advise urgently."),
        ("supplier_guangzhou.txt", "Guangzhou Print Co",
         "Invoice INV-88213 attached\n\nOutstanding balance USD 11,480 against PO-1019. "
         "Kindly arrange remittance this week."),
        ("supplier_ankara.txt", "Ankara Hardware",
         "Damaged units in last delivery\n\nCustomer reports 12 of 60 units faulty on arrival. "
         "We can offer credit note or replacement. Photos attached."),
    ]
    for fn, sender, body in mails:
        with open(os.path.join(INBOX, fn), "w", encoding="utf-8") as f:
            f.write(f"FROM: {sender}\nSUBJECT: {body.splitlines()[0]}\n\n{body}")

    today = date.today()
    rows = [
        ["PO-1042", "Shenzhen Yuan Trading", (today - timedelta(days=6)).isoformat(), "in_transit", 500, 4.20],
        ["PO-1019", "Guangzhou Print Co",   (today - timedelta(days=2)).isoformat(), "awaiting",   2400, 4.78],
        ["PO-1031", "Istanbul Textiles",    (today + timedelta(days=4)).isoformat(), "in_transit", 300, 9.10],
        ["PO-0988", "Nairobi Supplies",     (today - timedelta(days=14)).isoformat(), "awaiting",  150, 22.00],
        ["PO-1055", "Mumbai Components",    (today + timedelta(days=2)).isoformat(), "awaiting",   800, 6.35],
        ["PO-0975", "Lagos Packaging",      (today - timedelta(days=30)).isoformat(), "delivered", 1000, 1.10],
    ]
    with open(ORDERS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "supplier", "promised_ship", "status", "qty", "unit_cost"])
        w.writerows(rows)
    print(f"seeded {len(mails)} emails into inbox/ and {len(rows)} orders into orders.csv")


# ---------------------------------------------------------------- main

def priority_for(intent, urgent, days_late):
    if urgent or days_late > 3 or intent == "complaint":
        return "HIGH"
    if intent in ("delay", "quote") or days_late > 0:
        return "MEDIUM"
    return "LOW"


def main():
    ap = argparse.ArgumentParser(description="Supplier inbox triage + ops reporting")
    ap.add_argument("--seed-demo", action="store_true", help="write sample inbox + orders.csv")
    ap.add_argument("--use-llm", action="store_true", help="classify via OpenAI (needs OPENAI_API_KEY)")
    a = ap.parse_args()

    if a.seed_demo:
        seed_demo()
        return 0

    if not os.path.isdir(INBOX):
        print("! no inbox/ found - run:  python3 agent.py --seed-demo", file=sys.stderr)
        return 1

    orders = load_orders(ORDERS)
    late = overdue(orders)
    late_by_supplier = {l["supplier"]: l["days_late"] for l in late}

    rows = []
    files = sorted(f for f in os.listdir(INBOX) if f.endswith(".txt"))
    print(f"→ triaging {len(files)} emails{'  (LLM)' if a.use_llm else '  (rule engine)'}")
    for fn in files:
        raw = open(os.path.join(INBOX, fn), encoding="utf-8").read()
        sender = ""
        m = re.search(r"^FROM:\s*(.+)$", raw, re.M)
        if m:
            sender = m.group(1).strip()
        else:
            sender = fn.replace(".txt", "").replace("_", " ").title()
        subj = raw.splitlines()[1].replace("SUBJECT:", "").strip() if len(raw.splitlines()) > 1 else fn
        intent, conf, why = classify(raw, a.use_llm)
        d_late = late_by_supplier.get(sender, 0)
        rows.append({
            "supplier": sender, "subject": subj, "intent": intent,
            "confidence": conf, "evidence": why,
            "priority": priority_for(intent, detect_urgency(raw), d_late),
            "action": DRAFTS.get(intent, DRAFTS["other"]),
            "days_late": d_late,
        })

    rows.sort(key=lambda r: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[r["priority"]])

    actions = write_actions(rows, os.path.join(ROOT, "actions.csv"))
    report_path = os.path.join(ROOT, "ops_report.html")
    html_out = write_report(rows, late, report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"✓ {len(rows)} emails triaged: " +
          ", ".join(f"{r['intent']}({r['priority'][0]})" for r in rows))
    print(f"✓ {len(late)} overdue orders · ${sum(l['value_at_risk'] for l in late):,.0f} at risk")
    print(f"✓ wrote {os.path.relpath(actions, ROOT)}")
    print(f"✓ wrote {os.path.relpath(report_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
