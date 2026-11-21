"""Deterministic, synthetic financial-audit environment with a Gymnasium-like API."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import operator
import random
import string
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

VERSION = "0.3.0"
MAX_STEPS = 100
MAX_PAGE = 50
MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_EXPLANATION = 1000
TABLES = ("ledger", "invoices", "payments", "fx")
ERROR_TYPES = ("duplicate_invoice", "payment_allocation", "fx_rate_date", "period_cutoff")
SCENARIOS = ("clean", *ERROR_TYPES, "interacting_fx_allocation", "interacting_cutoff_duplicate")
CURRENCIES = {"KZT": 2, "USD": 2, "EUR": 2, "JPY": 0}
_KZT_PER_UNIT = {"KZT": 1.0, "USD": 500.0, "EUR": 540.0, "JPY": 3.5}

def _fx_rate(source: str, target: str) -> float:
    return _KZT_PER_UNIT[source] / _KZT_PER_UNIT[target]

def _to_minor(amount_minor: int, source: str, target: str) -> int:
    source_scale = 10 ** CURRENCIES[source]
    target_scale = 10 ** CURRENCIES[target]
    return round((amount_minor / source_scale) * _fx_rate(source, target) * target_scale)
COMPANIES = ("Asterion Labs", "Northwind Atelier", "Kite & Harbor", "Morrow Systems", "Pinecone Works", "Orbit Foundry")


def _id(rng: random.Random, prefix: str) -> str:
    return prefix + "-" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))


def _iso(day: date, offset: str = "+06:00") -> str:
    return f"{day.isoformat()}T12:00:00{offset}"


def _pair(rng, *, timestamp, account, amount, currency, entity_id, invoice_id=None, payment_id=None):
    base = {"timestamp": timestamp, "currency": currency, "entity_id": entity_id,
            "invoice_id": invoice_id, "payment_id": payment_id}
    a = {**base, "row_id": _id(rng, "L"), "account": account, "debit_minor": amount, "credit_minor": 0}
    b = {**base, "row_id": _id(rng, "L"), "account": "Accounts payable / clearing", "debit_minor": 0, "credit_minor": amount}
    return [a, b]


def generate_episode(seed: int, config: dict | None = None) -> tuple[dict, dict, dict]:
    """Return (open episode, private truth, provenance). Never serialize private truth to agent observations."""
    config = dict(config or {})
    rng = random.Random(int(seed))
    scenario = config.get("scenario") or SCENARIOS[int(seed) % len(SCENARIOS)]
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported scenario: {scenario}")
    base = config.get("base_currency", "KZT")
    if base not in CURRENCIES:
        raise ValueError("unsupported base currency")
    start = date.fromisoformat(config.get("period_start", "2026-10-01"))
    end = date.fromisoformat(config.get("period_end", "2026-10-31"))
    if end < start:
        raise ValueError("period_end must not precede period_start")
    offset = rng.choice(("+06:00", "+05:00", "+00:00"))
    episode_hash = hashlib.sha256(f"{VERSION}:{seed}:{json.dumps(config,sort_keys=True)}".encode()).hexdigest()[:10].upper()
    episode_id = f"EP-{episode_hash}"
    entity_id = _id(rng, "ENT")
    company = rng.choice(COMPANIES)
    foreign = rng.choice([c for c in CURRENCIES if c != base])

    invoices, payments, ledger, fx = [], [], [], []
    truth, provenance_changes = [], []
    invoice_by_id = {}
    for i in range(75):
        inv_id = _id(rng, "INV")
        doc_id = _id(rng, "DOC")
        inv_day = start + timedelta(days=rng.randrange(max(1, (end-start).days)))
        currency = foreign if i in (2, 5) else base
        minor = rng.choice((245000, 320000, 475000, 475000, 680000, 910000))
        if CURRENCIES[currency] == 0:
            minor = int(minor / 100)
        row_day = inv_day
        line = _pair(rng, timestamp=_iso(row_day, offset), account="Operating expense",
                     amount=_to_minor(minor, currency, base), currency=base, entity_id=entity_id, invoice_id=inv_id)
        ledger.extend(line)
        invoice = {"doc_id": doc_id, "invoice_id": inv_id, "entity_id": entity_id,
                   "issued_at": _iso(inv_day, offset), "due_at": _iso(inv_day + timedelta(days=30), offset),
                   "currency": currency, "total_minor": minor,
                   "line_items": [{"description": rng.choice(("Cloud hosting", "Studio rental", "Research services", "Office supplies", "Equipment lease")),
                                   "quantity": 1, "unit_minor": minor}]}
        invoices.append(invoice); invoice_by_id[inv_id] = invoice
        if currency != base:
            rate = _fx_rate(currency, base)
            fx.append({"date": inv_day.isoformat(), "source_currency": currency, "target_currency": base, "rate": f"{rate:.8f}"})

    # A clean, plausible split payment distractor and a lawful credit note keep heuristics honest.
    split_invoice = invoices[0]
    split_total = split_invoice["total_minor"]
    split_day = date.fromisoformat(split_invoice["issued_at"][:10]) + timedelta(days=5)
    for part in (split_total // 3, split_total - split_total // 3):
        pay_id, doc_id = _id(rng, "PAY"), _id(rng, "DOC")
        payments.append({"doc_id": doc_id, "payment_id": pay_id, "timestamp": _iso(split_day, offset),
                         "currency": split_invoice["currency"], "amount_minor": part,
                         "invoice_refs": [{"invoice_id": split_invoice["invoice_id"], "amount_minor": part}]})
        ledger.extend(_pair(rng, timestamp=_iso(split_day, offset), account="Cash / bank",
                            amount=_to_minor(part, split_invoice["currency"], base), currency=base, entity_id=entity_id,
                            invoice_id=split_invoice["invoice_id"], payment_id=pay_id))

    # Four normal payments; identical values occur intentionally as a harmless distractor.
    for invoice in invoices[1:5]:
        day = date.fromisoformat(invoice["issued_at"][:10]) + timedelta(days=4)
        pay_id, doc_id = _id(rng, "PAY"), _id(rng, "DOC")
        payments.append({"doc_id": doc_id, "payment_id": pay_id, "timestamp": _iso(day, offset),
                         "currency": invoice["currency"], "amount_minor": invoice["total_minor"],
                         "invoice_refs": [{"invoice_id": invoice["invoice_id"], "amount_minor": invoice["total_minor"]}]})
        ledger.extend(_pair(rng, timestamp=_iso(day, offset), account="Cash / bank",
                            amount=_to_minor(invoice["total_minor"], invoice["currency"], base), currency=base, entity_id=entity_id,
                            invoice_id=invoice["invoice_id"], payment_id=pay_id))

    # Balanced non-invoice journal activity gives episodes realistic inspection depth.
    for _ in range(20):
        misc_day = start + timedelta(days=rng.randrange(max(1, (end-start).days)))
        misc_amount = rng.choice((12500, 27500, 48000, 48000, 125000))
        ledger.extend(_pair(rng, timestamp=_iso(misc_day, offset), account="Operating adjustment",
                            amount=misc_amount, currency=base, entity_id=entity_id))

    def add_issue(kind, affected, amount, currency, evidence, explanation):
        truth.append({"issue_id": _id(rng, "ISS"), "type": kind, "affected_ids": sorted(set(affected)),
                      "expected_amount_minor": amount, "currency": currency,
                      "allowed_evidence_sets": [sorted(set(evidence))], "tolerance_minor": 1})
        provenance_changes.append({"type": kind, "changed_ids": sorted(set(affected)), "cause": explanation})

    def inject_duplicate():
        original = invoices[6]
        duplicate = copy.deepcopy(original)
        duplicate["doc_id"] = _id(rng, "DOC")
        duplicate["issued_at"] = _iso(date.fromisoformat(original["issued_at"][:10]) + timedelta(days=2), offset)
        invoices.append(duplicate)
        add_issue("duplicate_invoice", [original["invoice_id"]], original["total_minor"], original["currency"],
                  [original["doc_id"], duplicate["doc_id"]], "A second supplier document reuses the same invoice identity and amount.")

    def inject_allocation():
        invoice = invoices[5]
        paid = max(2, invoice["total_minor"] // 2)
        missing = invoice["total_minor"] - paid
        pay_id, doc_id = _id(rng, "PAY"), _id(rng, "DOC")
        payments.append({"doc_id": doc_id, "payment_id": pay_id,
                         "timestamp": _iso(date.fromisoformat(invoice["issued_at"][:10]) + timedelta(days=9), offset),
                         "currency": invoice["currency"], "amount_minor": invoice["total_minor"],
                         "invoice_refs": [{"invoice_id": invoice["invoice_id"], "amount_minor": paid}]})
        ledger.extend(_pair(rng, timestamp=payments[-1]["timestamp"], account="Cash / bank",
                            amount=_to_minor(invoice["total_minor"], invoice["currency"], base), currency=base, entity_id=entity_id,
                            invoice_id=invoice["invoice_id"], payment_id=pay_id))
        add_issue("payment_allocation", [pay_id, invoice["invoice_id"]], missing, invoice["currency"],
                  [doc_id, invoice["doc_id"]], "The payment document leaves part of the invoice unallocated.")

    def inject_fx():
        invoice = invoices[2]
        inv_day = date.fromisoformat(invoice["issued_at"][:10])
        payment_day = inv_day + timedelta(days=7)
        currency = invoice["currency"]
        correct = _fx_rate(currency, base)
        wrong = correct * 1.08
        fx.extend([{"date": inv_day.isoformat(), "source_currency": currency, "target_currency": base, "rate": f"{correct:.8f}"},
                   {"date": payment_day.isoformat(), "source_currency": currency, "target_currency": base, "rate": f"{wrong:.8f}"}])
        row = next(r for r in ledger if r["invoice_id"] == invoice["invoice_id"])
        wrong_base_minor = round(invoice["total_minor"] * wrong * (10 ** CURRENCIES[base]) / (10 ** CURRENCIES[currency]))
        for posting in ledger:
            if posting["invoice_id"] == invoice["invoice_id"]:
                posting["debit_minor"] = wrong_base_minor if posting["debit_minor"] else 0
                posting["credit_minor"] = wrong_base_minor if posting["credit_minor"] else 0
        delta = max(1, round(invoice["total_minor"] * abs(wrong-correct) * (10 ** CURRENCIES[base]) / (10 ** CURRENCIES[currency])))
        add_issue("fx_rate_date", [invoice["invoice_id"], row["row_id"]], delta, base,
                  [invoice["doc_id"], row["row_id"]], "The policy requires the invoice-date FX rate; the posting evidence uses a later rate.")

    def inject_cutoff():
        invoice = invoices[7]
        row = next(r for r in ledger if r["invoice_id"] == invoice["invoice_id"])
        late = end + timedelta(days=1)
        row["timestamp"] = _iso(late, offset)
        # Keep debit/credit pair on the same timestamp to preserve the journal balance.
        for peer in ledger:
            if peer["invoice_id"] == invoice["invoice_id"]:
                peer["timestamp"] = _iso(late, offset)
        add_issue("period_cutoff", [invoice["invoice_id"], row["row_id"]], invoice["total_minor"], base,
                  [invoice["doc_id"], row["row_id"]], "The invoice belongs to the period but its journal entry is dated after period end.")

    if scenario == "duplicate_invoice": inject_duplicate()
    elif scenario == "payment_allocation": inject_allocation()
    elif scenario == "fx_rate_date": inject_fx()
    elif scenario == "period_cutoff": inject_cutoff()
    elif scenario == "interacting_fx_allocation": inject_fx(); inject_allocation()
    elif scenario == "interacting_cutoff_duplicate": inject_cutoff(); inject_duplicate()

    # Stable ordering is seeded; generated IDs and company labels vary between seeds.
    rng.shuffle(ledger); rng.shuffle(invoices); rng.shuffle(payments); rng.shuffle(fx)
    doc_count = len(invoices) + len(payments)
    manifest = {"schema_version": "1.0", "episode_id": episode_id, "company": company,
                "entity_id": entity_id, "base_currency": base, "minor_units": CURRENCIES,
                "period_start": start.isoformat(), "period_end": end.isoformat(), "timezone": offset,
                "ledger_rows": len(ledger), "document_count": doc_count,
                "rounding": "round half away from zero; tolerance 1 minor unit"}
    policy = {"fx_rate_date": "invoice_date", "period_boundary": "inclusive start, exclusive end+1 day",
              "rounding": "half away from zero", "base_currency": base}
    episode = {"manifest": manifest, "policy": policy, "tables": {"ledger": ledger, "invoices": invoices,
                "payments": payments, "fx": fx}}
    provenance = {"episode_id": episode_id, "seed_digest": hashlib.sha256(str(seed).encode()).hexdigest(),
                  "scenario": scenario, "changes": provenance_changes}
    return episode, {"episode_id": episode_id, "issues": truth}, provenance


def _canonical(ids):
    return sorted(set(str(item) for item in ids))


def _matchable(finding, truth):
    try:
        amount = int(finding["claimed_amount_minor"])
        expected = int(truth["expected_amount_minor"])
        if finding.get("type") != truth["type"]: return False
        if _canonical(finding.get("affected_ids", [])) != truth["affected_ids"]: return False
        if finding.get("currency") != truth["currency"]: return False
        if abs(amount - expected) > int(truth.get("tolerance_minor", 0)): return False
        evidence = set(_canonical(finding.get("evidence_ids", [])))
        return any(set(allowed).issubset(evidence) for allowed in truth["allowed_evidence_sets"])
    except (KeyError, TypeError, ValueError):
        return False


def score_findings(findings: list[dict], truth: dict, inspection_steps: int) -> dict:
    """Maximum one-to-one evidence-aware matching; duplicates remain false positives."""
    gold = truth.get("issues", [])
    adjacency = [[i for i, target in enumerate(gold) if _matchable(candidate, target)] for candidate in findings]
    matched_gold: dict[int, int] = {}
    def augment(pred_index, seen):
        for gold_index in adjacency[pred_index]:
            if gold_index in seen: continue
            seen.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], seen):
                matched_gold[gold_index] = pred_index
                return True
        return False
    for p in range(len(findings)): augment(p, set())
    tp = len(matched_gold); fp = len(findings) - tp; fn = len(gold) - tp
    f1 = (2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn) else 1.0
    reward = max(-1.0, min(1.0, 2*f1 - 1 - 0.002*inspection_steps))
    return {"tp":tp,"fp":fp,"fn":fn,"precision":tp/(tp+fp) if tp+fp else 0.0,
            "recall":tp/(tp+fn) if tp+fn else 0.0,"f1":f1,"inspection_steps":inspection_steps,
            "reward":reward,"return":reward}


_SAFE_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_SAFE_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
def safe_calculate(expression: str):
    if len(expression) > 160: raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")
    def walk(node):
        if isinstance(node, ast.Expression): return walk(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
            if isinstance(node.op, ast.Pow) and abs(walk(node.right)) > 8: raise ValueError("power too large")
            return _SAFE_BINOPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY: return _SAFE_UNARY[type(node.op)](walk(node.operand))
        raise ValueError("only decimal arithmetic is allowed")
    result = walk(tree)
    if abs(result) > 1e18: raise ValueError("result out of range")
    return result


class AuditEnv:
    def __init__(self): self.reset(0)
    def reset(self, seed=0, config=None):
        self.episode, self._truth, self._provenance = generate_episode(seed, config)
        self.steps = 0; self.inspection_steps = 0; self.done = False
        self.mode = (config or {}).get("mode", "open")
        self.mode = "closed" if self.mode == "closed" else "open"
        self.submission = []; self._last_metrics = None
        return self._initial_observation(), {"schema_version": VERSION}
    def _initial_observation(self):
        return {"task":"Find and report supported financial anomalies. Cite source record/document IDs.",
                "manifest":copy.deepcopy(self.episode["manifest"]),"policy":copy.deepcopy(self.episode["policy"]),
                "budget":{"steps_remaining":MAX_STEPS,"max_steps":MAX_STEPS,"max_page":MAX_PAGE},
                "available_actions":["list_records","get_record","get_document","calculator","submit"]}
    def _lookup(self, record_id):
        for table, rows in self.episode["tables"].items():
            for row in rows:
                if row.get("row_id") == record_id or row.get("doc_id") == record_id or row.get("payment_id") == record_id or row.get("invoice_id") == record_id:
                    return table, row
        return None, None
    def step(self, action):
        if self.done: return {"error":"episode already finished","budget":{"steps_remaining":0}}, 0, True, False, {}
        self.steps += 1
        if not isinstance(action, dict): action = {}
        name = action.get("action")
        obs = {}
        reward = 0.0; terminated = False; truncated = False; info = {}
        if name == "list_records":
            table = action.get("table")
            if table not in TABLES:
                obs = {"error":"unknown table"}
            else:
                try: offset = max(0, int(action.get("cursor", 0)))
                except (ValueError, TypeError): offset = 0
                try: limit = min(MAX_PAGE, max(1, int(action.get("limit", 20))))
                except (ValueError, TypeError): limit = 20
                records = self.episode["tables"][table]
                obs = {"table":table,"rows":copy.deepcopy(records[offset:offset+limit]),"cursor_next":offset+limit if offset+limit<len(records) else None,"count":len(records)}
            self.inspection_steps += 1
        elif name in ("get_record", "get_document"):
            record_id = str(action.get("id", ""))
            table, row = self._lookup(record_id)
            if row is None: obs = {"error":"record not found"}
            elif name == "get_document" and table not in ("invoices", "payments"): obs = {"error":"document not found"}
            else: obs = {"table":table,"record":copy.deepcopy(row)}
            self.inspection_steps += 1
        elif name == "calculator":
            try: obs = {"result":safe_calculate(str(action.get("expression", "")))}
            except Exception: obs = {"error":"invalid arithmetic expression"}
            self.inspection_steps += 1
        elif name == "submit":
            findings = action.get("findings")
            if not isinstance(findings, list) or len(findings) > MAX_FINDINGS:
                findings = []; reward = -1.0 if self.mode == "open" else 0.0; obs = {"error":"invalid submission structure"}
                if self.mode == "closed": self._last_metrics = {"invalid_submission":True,"reward":-1.0}
            else:
                for finding in findings:
                    evidence = finding.get("evidence_ids") if isinstance(finding, dict) else None
                    explanation = finding.get("explanation", "") if isinstance(finding, dict) else None
                    if (not isinstance(finding, dict) or not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE
                        or not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION
                        or not isinstance(finding.get("affected_ids"), list)
                        or type(finding.get("claimed_amount_minor")) is not int
                        or not isinstance(finding.get("currency"), str)):
                        reward=-1.0 if self.mode == "open" else 0.0; obs={"error":"finding violates schema limits"}
                        if self.mode == "closed": self._last_metrics = {"invalid_submission":True,"reward":-1.0}
                        break
                else:
                    self.submission = copy.deepcopy(findings)
                    if self.mode == "open":
                        score=score_findings(findings,self._truth,self.inspection_steps)
                        reward=score["reward"]; obs={"submitted":True,"metrics":score}
                    else:
                        reward=0.0; obs={"submitted":True,"message":"Episode submitted."}
                        self._last_metrics = score_findings(findings,self._truth,self.inspection_steps)
            terminated=True; self.done=True
        else:
            obs={"error":"unknown action"}; self.inspection_steps += 1
        if not terminated and self.steps >= MAX_STEPS:
            truncated=True; self.done=True; obs={"message":"Step budget exhausted.","findings":[]}
        remaining=max(0,MAX_STEPS-self.steps)
        obs["budget"]={"steps_remaining":remaining,"steps_used":self.steps,"inspection_steps":self.inspection_steps}
        return obs,reward,terminated,truncated,info
    def evaluator_result(self):
        if self.mode != "closed" or not self.done: raise RuntimeError("closed episode must be submitted before evaluator access")
        return {"truth":copy.deepcopy(self._truth),"provenance":copy.deepcopy(self._provenance),
                "metrics":copy.deepcopy(self._last_metrics)}


def oracle_findings(truth):
    return [{"type":x["type"],"affected_ids":x["affected_ids"],"claimed_amount_minor":x["expected_amount_minor"],
             "currency":x["currency"],"evidence_ids":x["allowed_evidence_sets"][0],"explanation":"Supported by cited synthetic source records."} for x in truth["issues"]]


def run_policy(policy="oracle", seed=42, scenario=None, mode="open"):
    config={"mode":mode}
    if scenario: config["scenario"]=scenario
    env=AuditEnv(); observation,_=env.reset(seed,config); trace=[]; findings=[]
    episode_id=observation["manifest"]["episode_id"]
    def act(action):
        obs,reward,terminated,truncated,info=env.step(action)
        trace.append({"episode_id":episode_id,"step":len(trace)+1,"action":action,"observation":obs,
                      "reward":reward,"terminated":terminated,"truncated":truncated})
        return obs
    def list_all(table):
        rows=[]; cursor=0
        while True:
            page=act({"action":"list_records","table":table,"cursor":cursor,"limit":MAX_PAGE})
            rows.extend(page.get("rows",[]))
            next_cursor=page.get("cursor_next")
            if next_cursor is None: break
            cursor=next_cursor
        return rows
    if policy == "oracle":
        # Evaluator-only ceiling: this adapter is intentionally outside the agent tool interface.
        findings=oracle_findings(env._truth)
        act({"action":"submit","findings":findings})
        metrics=score_findings(findings,env._truth,0)
    elif policy == "flag_all":
        rows=list_all("ledger")
        for row in rows[:MAX_FINDINGS]:
            findings.append({"type":"period_cutoff","affected_ids":[row["row_id"]],
                "claimed_amount_minor":row["debit_minor"] or row["credit_minor"],"currency":row["currency"],
                "evidence_ids":[row["row_id"]],"explanation":"Flag-all attack baseline."})
        act({"action":"submit","findings":findings})
        metrics=score_findings(findings,env._truth,env.inspection_steps)
    elif policy == "reconcile":
        ledger=list_all("ledger"); invoices=list_all("invoices"); payments=list_all("payments"); fx=list_all("fx")
        inv_groups={}
        for inv in invoices: inv_groups.setdefault(inv["invoice_id"],[]).append(inv)
        for inv_id, docs in inv_groups.items():
            if len(docs)>1:
                findings.append({"type":"duplicate_invoice","affected_ids":[inv_id],
                    "claimed_amount_minor":docs[0]["total_minor"],"currency":docs[0]["currency"],
                    "evidence_ids":[d["doc_id"] for d in docs],"explanation":"Repeated invoice identity across source documents."})
        inv_by={i["invoice_id"]:i for i in invoices}
        for payment in payments:
            refs=payment.get("invoice_refs",[])
            allocated=sum(int(ref.get("amount_minor",0)) for ref in refs)
            unallocated=int(payment.get("amount_minor",0))-allocated
            if unallocated>0 and len(refs)==1:
                inv=inv_by.get(refs[0].get("invoice_id"))
                if inv:
                    findings.append({"type":"payment_allocation","affected_ids":[payment["payment_id"],inv["invoice_id"]],
                        "claimed_amount_minor":unallocated,"currency":payment["currency"],
                        "evidence_ids":[payment["doc_id"],inv["doc_id"]],"explanation":"Payment total exceeds the amount allocated to its invoice."})
        manifest=env.episode["manifest"]; policy_doc=env.episode["policy"]
        fx_by_date={(x["date"],x["source_currency"],x["target_currency"]):float(x["rate"]) for x in fx}
        for inv in invoices:
            issue_date=inv["issued_at"][:10]
            postings=[r for r in ledger if r.get("invoice_id")==inv["invoice_id"] and r.get("account")=="Operating expense"]
            for row in postings:
                if issue_date>=manifest["period_start"] and issue_date<=manifest["period_end"] and row["timestamp"][:10]>manifest["period_end"]:
                    findings.append({"type":"period_cutoff","affected_ids":[inv["invoice_id"],row["row_id"]],
                        "claimed_amount_minor":row["debit_minor"],"currency":manifest["base_currency"],
                        "evidence_ids":[inv["doc_id"],row["row_id"]],"explanation":"In-period invoice is posted after the reporting cutoff."})
                if inv["currency"]!=manifest["base_currency"]:
                    rate=fx_by_date.get((issue_date,inv["currency"],manifest["base_currency"]))
                    if rate is None: continue
                    expected=round((inv["total_minor"]/(10**CURRENCIES[inv["currency"]]))*rate*(10**CURRENCIES[manifest["base_currency"]]))
                    delta=abs(int(row["debit_minor"])-expected)
                    if delta>1:
                        findings.append({"type":"fx_rate_date","affected_ids":[inv["invoice_id"],row["row_id"]],
                            "claimed_amount_minor":delta,"currency":manifest["base_currency"],
                            "evidence_ids":[inv["doc_id"],row["row_id"]],"explanation":"Posted base amount differs from the policy invoice-date conversion."})
        act({"action":"submit","findings":findings})
        metrics=score_findings(findings,env._truth,env.inspection_steps)
    else:
        # Fixed-seed random explorer is deterministic and abstains at submit.
        for table in ("ledger","invoices","payments"):
            act({"action":"list_records","table":table,"limit":8})
        act({"action":"submit","findings":[]})
        metrics=score_findings([],env._truth,env.inspection_steps)
    return {"metrics":metrics,"trace":trace,"episode":env.episode,"truth":env._truth,
            "provenance":env._provenance,"findings":findings}
