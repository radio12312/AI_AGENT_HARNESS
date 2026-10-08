"""Deterministic mock tools backed by fixtures/db.json, with per-case fault injection.

The tools are deliberately permissive about *policy* (they will issue a second
refund, or a refund older than 30 days): enforcing policy is the agent's job,
and that is what the eval measures. They only reject requests that a real
backend could not execute at all (unknown order, non-positive or over-total
amounts).
"""
from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

FAULT_TYPES = ("error", "timeout", "malformed")

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Look up a customer account by email address.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string", "description": "Customer email address"}},
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Get full details of one order: items, total, status, dates, tracking and past refunds.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-1001"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "List all orders (summary) for a customer ID.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string", "description": "Customer ID, e.g. C001"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the help-center knowledge base. Returns the most relevant articles.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue a refund to the original payment method for an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number", "description": "Refund amount in USD"},
                    "reason": {"type": "string"},
                },
                "required": ["order_id", "amount", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Hand the conversation to a human support specialist.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "Short summary of the issue"}},
                "required": ["summary"],
            },
        },
    },
]
TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]

_STOPWORDS = {
    "the", "and", "for", "you", "your", "what", "how", "does", "can", "with", "are", "this", "that",
    "have", "has", "from", "about", "will", "would", "when", "long", "take", "get", "my", "our",
}


@dataclass
class Fault:
    tool: str
    type: str
    on_call: object = 1  # 1-based call number, list of numbers, or "all"

    def applies(self, call_number: int) -> bool:
        if self.on_call == "all":
            return True
        if isinstance(self.on_call, list):
            return call_number in self.on_call
        return call_number == int(self.on_call)

    @classmethod
    def parse(cls, spec: dict) -> "Fault":
        if spec.get("type") not in FAULT_TYPES:
            raise ValueError(f"unknown fault type {spec.get('type')!r}; expected one of {FAULT_TYPES}")
        if spec.get("tool") not in TOOL_NAMES:
            raise ValueError(f"unknown tool in fault_injection: {spec.get('tool')!r}")
        return cls(tool=spec["tool"], type=spec["type"], on_call=spec.get("on_call", 1))


@dataclass
class ToolResult:
    content: str               # exactly what the model sees
    ok: bool                   # tool executed and returned a normal result
    fault: str | None = None   # injected fault type, if any
    error: str | None = None   # error code returned by the tool, if any


@dataclass
class ToolEnv:
    """One run's private copy of the fixture. Build a fresh one per run (= reset)."""

    db: dict
    faults: list[Fault] = field(default_factory=list)
    fault_timeout_s: float = 0.0
    call_counts: Counter = field(default_factory=Counter)
    new_refunds: list[dict] = field(default_factory=list)
    escalations: list[dict] = field(default_factory=list)

    @classmethod
    def from_fixture(cls, path: str | Path, faults: list[dict] | None = None, fault_timeout_s: float = 0.0):
        with open(path, encoding="utf-8") as f:
            db = json.load(f)
        return cls(db=db, faults=[Fault.parse(s) for s in faults or []], fault_timeout_s=fault_timeout_s)

    @classmethod
    def from_db(cls, db: dict, faults: list[dict] | None = None, fault_timeout_s: float = 0.0):
        return cls(db=copy.deepcopy(db), faults=[Fault.parse(s) for s in faults or []],
                   fault_timeout_s=fault_timeout_s)

    # ---- dispatch -----------------------------------------------------------
    def call(self, name: str, args: dict) -> ToolResult:
        if name not in TOOL_NAMES:
            return _err("unknown_tool", f"No tool named {name!r}. Available: {', '.join(TOOL_NAMES)}")
        self.call_counts[name] += 1
        n = self.call_counts[name]
        for fault in self.faults:
            if fault.tool == name and fault.applies(n):
                return self._inject(fault, name)
        try:
            payload = getattr(self, f"_t_{name}")(**args)
        except TypeError as e:
            return _err("invalid_arguments", f"{name}: {e}")
        if "error" in payload:
            return ToolResult(content=_dump(payload), ok=False, error=payload["error"])
        return ToolResult(content=_dump(payload), ok=True)

    def _inject(self, fault: Fault, name: str) -> ToolResult:
        # Faults short-circuit before the tool runs, so they never have side effects.
        if fault.type == "error":
            content = _dump({"error": "service_unavailable",
                             "message": f"503 Service Unavailable: the backend for {name} is temporarily unavailable."})
        elif fault.type == "timeout":
            if self.fault_timeout_s:
                time.sleep(self.fault_timeout_s)
            content = _dump({"error": "timeout", "message": f"{name} timed out after 30 seconds with no response."})
        else:  # malformed: truncated, invalid JSON that leaks no facts
            content = '{"status": "ok", "data": {"' + name + '": [{"id": "\u0000�'
        return ToolResult(content=content, ok=False, fault=fault.type, error=fault.type)

    # ---- state for expected_state checks -----------------------------------
    def state(self) -> dict:
        return {
            "refunds": [{"order_id": r["order_id"], "amount": r["amount"]} for r in self.new_refunds],
            "escalated": bool(self.escalations),
            "escalations": len(self.escalations),
        }

    # ---- tools --------------------------------------------------------------
    def _order(self, order_id) -> dict | None:
        oid = str(order_id).strip().upper()
        return next((o for o in self.db["orders"] if o["order_id"] == oid), None)

    def _t_get_customer(self, email):
        e = str(email).strip().lower()
        c = next((c for c in self.db["customers"] if c["email"].lower() == e), None)
        if c is None:
            return {"error": "not_found", "message": f"No customer with email {email}"}
        return dict(c)

    def _t_get_order(self, order_id):
        o = self._order(order_id)
        if o is None:
            return {"error": "not_found", "message": f"No order with id {order_id}"}
        return copy.deepcopy(o)

    def _t_list_orders(self, customer_id):
        cid = str(customer_id).strip().upper()
        if not any(c["customer_id"] == cid for c in self.db["customers"]):
            return {"error": "not_found", "message": f"No customer with id {customer_id}"}
        orders = [
            {"order_id": o["order_id"], "order_date": o["order_date"], "status": o["status"], "total": o["total"]}
            for o in self.db["orders"] if o["customer_id"] == cid
        ]
        return {"customer_id": cid, "orders": orders, "count": len(orders)}

    def _t_search_kb(self, query):
        terms = _terms(query)
        scored = []
        for a in self.db["kb_articles"]:
            title, tags, body = _terms(a["title"]), _terms(" ".join(a["tags"])), _terms(a["body"])
            score = sum(2 * (t in title) + 2 * (t in tags) + (t in body) for t in terms)
            if score:
                scored.append((-score, a["id"], a))
        scored.sort(key=lambda x: (x[0], x[1]))
        results = [{"id": a["id"], "title": a["title"], "body": a["body"]} for _, _, a in scored[:2]]
        return {"query": query, "results": results}

    def _t_issue_refund(self, order_id, amount, reason=""):
        o = self._order(order_id)
        if o is None:
            return {"error": "not_found", "message": f"No order with id {order_id}"}
        try:
            amt = round(float(str(amount).replace("$", "").replace(",", "")), 2)
        except ValueError:
            return {"error": "invalid_amount", "message": f"Amount {amount!r} is not a number"}
        if amt <= 0 or amt > o["total"]:
            return {"error": "invalid_amount", "message": f"Amount must be between 0 and the order total {o['total']:.2f}"}
        refund_id = f"RF-{1001 + sum(len(x['refunds']) for x in self.db['orders'])}"
        record = {"refund_id": refund_id, "amount": amt, "reason": str(reason), "date": self.db["meta"]["today"]}
        o["refunds"].append(record)
        if sum(r["amount"] for r in o["refunds"]) >= o["total"] - 0.005:
            o["status"] = "refunded"
        self.new_refunds.append({"order_id": o["order_id"], **record})
        return {"refund_id": refund_id, "order_id": o["order_id"], "amount": amt, "status": "processed",
                "message": f"Refund of ${amt:.2f} issued to the original payment method (5-7 business days)."}

    def _t_escalate_to_human(self, summary):
        ticket = f"ESC-{1001 + len(self.escalations)}"
        self.escalations.append({"ticket_id": ticket, "summary": str(summary)})
        return {"ticket_id": ticket, "status": "queued",
                "message": "A human specialist will follow up with the customer within 24 hours."}


def _terms(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(content=_dump({"error": code, "message": message}), ok=False, error=code)
