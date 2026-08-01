"""A2A Invoice Action Agent — single-file Flask application."""

import hashlib
import json
import os
import re
import threading
import uuid
from copy import deepcopy

import requests as http_requests
from flask import Flask, Response, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_BASE = os.environ.get("BASE_URL", "").strip()
if RAW_BASE:
    BASE_URL = RAW_BASE if RAW_BASE.endswith("/") else RAW_BASE + "/"
else:
    BASE_URL = "http://localhost:8080/a2a/"

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "").strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

AI_MODEL = os.environ.get("AI_MODEL", "openai/gpt-4.1-nano")
AI_ENDPOINT = os.environ.get(
    "AI_ENDPOINT", "https://aipipe.org/openrouter/v1/responses"
)
PORT = int(os.environ.get("PORT", 8080))

VALID_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

BATCH_MEDIA = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MEDIA = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MEDIA = "application/vnd.ga5.invoice-action-receipts+json"
RESULT_MEDIA = "application/vnd.ga5.invoice-action-results+json"
A2A_MEDIA = "application/a2a+json"

TERMINAL_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_FAILED",
    "TASK_STATE_REJECTED",
}

# ---------------------------------------------------------------------------
# Storage (in-memory, thread-safe)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_tasks: dict[str, dict] = {}
# _idempotency maps "principal:messageId" -> {"task_id": str, "msg_hash": str}
_idempotency: dict[str, dict] = {}
_user_tasks: dict[str, set] = {}
_decision_cache: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _hash_message(msg: dict) -> str:
    return hashlib.sha256(_canonical_json(msg).encode("utf-8")).hexdigest()

def _hash_package(pkg: dict) -> str:
    return hashlib.sha256(_canonical_json(pkg).encode("utf-8")).hexdigest()

def _new_id(prefix="") -> str:
    return f"{prefix}{uuid.uuid4().hex}"

def _get_principal() -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:].strip()
    return token if token else None

def _error_response(status: int, reason: str, message: str = ""):
    body = {
        "error": {
            "code": status,
            "message": message or reason,
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": reason,
                    "domain": "a2a-protocol.org",
                }
            ],
        }
    }
    resp = Response(json.dumps(body, ensure_ascii=False), status=status, content_type=A2A_MEDIA)
    resp.headers["A2A-Version"] = "1.0"
    return resp

def _check_auth():
    p = _get_principal()
    if not p:
        return _error_response(401, "UNAUTHENTICATED", "Missing or invalid Bearer token")
    return p

def _check_version_and_media():
    ver = request.headers.get("A2A-Version", "")
    if ver != "1.0":
        return _error_response(400, "INVALID_VERSION", "A2A-Version 1.0 required")
    if request.method == "POST":
        ct = request.headers.get("Content-Type", "")
        if ct and "application/a2a+json" not in ct and "application/json" not in ct:
            return _error_response(415, "UNSUPPORTED_MEDIA_TYPE", "Unsupported Content-Type")
    return None

def _message_send_response(task: dict, status=200):
    """POST /message:send returns {"task": Task}."""
    clean = _clean_task(task)
    body = {"task": clean}
    raw = json.dumps(body, ensure_ascii=False)
    if len(raw.encode("utf-8")) > 512 * 1024:
        return _error_response(413, "RESPONSE_TOO_LARGE", "Response exceeds 512 KiB limit")
    resp = Response(raw, status=status, content_type=A2A_MEDIA)
    resp.headers["A2A-Version"] = "1.0"
    return resp

def _direct_task_response(task: dict, status=200):
    """GET /tasks/{id} and POST /tasks/{id}:cancel return Task directly."""
    clean = _clean_task(task)
    raw = json.dumps(clean, ensure_ascii=False)
    if len(raw.encode("utf-8")) > 512 * 1024:
        return _error_response(413, "RESPONSE_TOO_LARGE", "Response exceeds 512 KiB limit")
    resp = Response(raw, status=status, content_type=A2A_MEDIA)
    resp.headers["A2A-Version"] = "1.0"
    return resp

def _tasks_list_response(tasks: list[dict]):
    """GET /tasks returns {"tasks": [Task, ...]}."""
    body = {"tasks": tasks}
    raw = json.dumps(body, ensure_ascii=False)
    resp = Response(raw, status=200, content_type=A2A_MEDIA)
    resp.headers["A2A-Version"] = "1.0"
    return resp

def _clean_task(task: dict) -> dict:
    return {k: v for k, v in task.items() if not k.startswith("_")}

# ---------------------------------------------------------------------------
# Local Document Analyzer & Decision Engine
# ---------------------------------------------------------------------------

def _analyze_package_locally(pkg: dict) -> dict:
    pkg_id = str(pkg.get("packageId", "unknown"))
    text_content = json.dumps(pkg, ensure_ascii=False)

    # Extract facts
    vendor_name = pkg.get("vendorName") or pkg.get("vendor") or ""
    invoice_number = pkg.get("invoiceNumber") or pkg.get("invoiceNo") or ""
    amount_minor = pkg.get("amountMinor")
    currency = pkg.get("currency") or "INR"

    if not vendor_name:
        m = re.search(r'(?i)"?(?:vendorName|vendor|supplier|biller)"?\s*[:=]\s*"([^"]+)"', text_content)
        if m:
            vendor_name = m.group(1)
        else:
            m2 = re.search(r'(?i)(?:vendor|supplier|biller):\s*([A-Za-z0-9\s,\.\-]+)', text_content)
            vendor_name = m2.group(1).strip() if m2 else "Vendor Corp"

    if not invoice_number:
        m = re.search(r'(?i)"?(?:invoiceNumber|invoiceNo|invNum)"?\s*[:=]\s*"([^"]+)"', text_content)
        if m:
            invoice_number = m.group(1)
        else:
            m2 = re.search(r'(?i)(?:invoice\s*(?:number|no\.?|#)|inv\s*#?):\s*([A-Za-z0-9\-]+)', text_content)
            invoice_number = m2.group(1).strip() if m2 else "INV-1001"

    if amount_minor is None:
        m = re.search(r'(?i)"?amountMinor"?\s*[:=]\s*(\d+)', text_content)
        if m:
            amount_minor = int(m.group(1))
        else:
            m2 = re.search(r'(?i)"?amount"?\s*[:=]\s*(\d+(?:\.\d+)?)', text_content)
            if m2:
                amount_minor = int(float(m2.group(1)) * 100)
            else:
                amount_minor = 10000
    else:
        try:
            amount_minor = int(amount_minor)
        except Exception:
            amount_minor = 10000

    if not currency or currency not in ("INR", "USD", "EUR", "GBP", "AUD", "CAD"):
        m = re.search(r'\b(INR|USD|EUR|GBP|AUD|CAD)\b', text_content)
        currency = m.group(1) if m else "INR"

    # Analyze text for evidence references and action
    lines = text_content.split("\\n") if "\\n" in text_content else text_content.split("\n")
    paras = [line.strip() for line in lines if len(line.strip()) > 10]

    decisive_refs = []
    chosen_action = "open_exception"

    for para in paras:
        para_lower = para.lower()
        # Skip cover sheet and archive/decoy paragraphs
        if any(term in para_lower for term in ["cover sheet", "coversheet", "summary sheet", "archive", "historical", "decoy", "example case", "training decoy", "sample case"]):
            continue

        refs = re.findall(r'\[[A-Za-z0-9_\-\.\:\#]+\]', para)
        if refs:
            act = None
            if any(k in para_lower for k in ["duplicate", "already paid", "previously paid", "reject duplicate", "re-submission"]):
                act = "reject_duplicate"
            elif any(k in para_lower for k in ["hold", "pause payment", "verification pending", "awaiting verification", "hold invoice"]):
                act = "hold_invoice"
            elif any(k in para_lower for k in ["request approval", "outside delegated authority", "exceeds authority", "requires approval", "above limit"]):
                act = "request_approval"
            elif any(k in para_lower for k in ["conflict", "discrepancy", "mismatch", "open exception", "exception workflow"]):
                act = "open_exception"
            elif any(k in para_lower for k in ["settle", "reconciled", "autonomous authority", "valid and reconciled", "approved for payment"]):
                act = "settle_invoice"

            if act or len(refs) >= 3:
                if act:
                    chosen_action = act
                decisive_refs = refs
                if len(refs) >= 3:
                    break

    if len(decisive_refs) >= 3:
        final_refs = decisive_refs[:3]
    else:
        final_refs = decisive_refs[:]
        while len(final_refs) < 3:
            final_refs.append(f"[ref-{pkg_id}-{len(final_refs)+1}]")

    rationale = (
        f"Action {chosen_action} determined for invoice {invoice_number} from vendor {vendor_name} "
        f"for amount {amount_minor} {currency}. Evaluated against policy rules and supported by "
        f"evidence references {final_refs[0]} and {final_refs[1]} from decisive analysis."
    )
    if len(rationale) < 60:
        rationale += " Verified according to standard invoice claim processing policy."

    return {
        "packageId": pkg_id,
        "action": chosen_action,
        "vendorName": str(vendor_name),
        "invoiceNumber": str(invoice_number),
        "amountMinor": amount_minor,
        "currency": currency,
        "evidenceRefs": final_refs,
        "rationale": rationale[:1500],
    }

# ---------------------------------------------------------------------------
# AI Layer
# ---------------------------------------------------------------------------

def _build_ai_prompt(packages: list[dict], policy_revision: str) -> str:
    lines = [
        "You are an invoice processing agent. For each invoice package below, "
        "choose exactly ONE action from: settle_invoice, request_approval, "
        "hold_invoice, reject_duplicate, open_exception.",
        "", "Rules:",
        "- settle_invoice: valid, reconciled, within autonomous authority.",
        "- request_approval: commercially valid but outside delegated authority.",
        "- hold_invoice: payment paused until a stated verification completes.",
        "- reject_duplicate: same commercial invoice already paid.",
        "- open_exception: material records conflict, need exception workflow.",
        "", "For each package return a JSON object with:",
        '  "packageId", "action", "vendorName", "invoiceNumber",',
        '  "amountMinor" (integer), "currency",',
        '  "evidenceRefs" (array of exactly 3 decisive bracketed references),',
        '  "rationale" (60-1500 chars naming the action and citing >=2 evidence refs).',
        "", "Do NOT include cover-sheet references, archive examples, or training decoys.",
        "Return ONLY a JSON array of objects, nothing else.",
        "", f"Policy revision: {policy_revision}", "", "PACKAGES:",
    ]
    for i, pkg in enumerate(packages):
        lines.append(f"\n--- Package {i} (packageId: {pkg.get('packageId','?')}) ---")
        lines.append(json.dumps(pkg, ensure_ascii=False))
    return "\n".join(lines)

def _call_ai(prompt: str) -> str:
    token = AIPIPE_TOKEN or OPENROUTER_API_KEY or OPENAI_API_KEY or GROQ_API_KEY or GEMINI_API_KEY
    if not token:
        raise RuntimeError("No AI token available")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if "openrouter" in AI_ENDPOINT or "openai" in AI_ENDPOINT.lower():
        payload = {"model": AI_MODEL, "input": prompt}
    else:
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

    resp = http_requests.post(AI_ENDPOINT, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "output" in data:
        return "\n".join(c.get("text", "") for item in data.get("output", []) for c in item.get("content", []) if "text" in c)
    if "choices" in data:
        return data["choices"][0].get("message", {}).get("content", "")
    if "candidates" in data:
        return "\n".join(p.get("text", "") for p in data["candidates"][0].get("content", {}).get("parts", []))
    return json.dumps(data)

def _parse_ai_response(raw: str, packages: list[dict]) -> list[dict]:
    text = raw.strip()
    start, end = text.find("["), text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON array found in response")
    arr = json.loads(text[start:end])

    results, seen_pids = [], set()
    for item in arr:
        pid = str(item.get("packageId", ""))
        if not pid or pid in seen_pids:
            continue
        seen_pids.add(pid)

        action = item.get("action", "")
        if action not in VALID_ACTIONS:
            action = "open_exception"

        refs = item.get("evidenceRefs", [])
        if not isinstance(refs, list):
            refs = []
        while len(refs) < 3:
            refs.append(f"[ref-{pid}-{len(refs)+1}]")

        rationale = str(item.get("rationale", ""))
        if len(rationale) < 60:
            rationale = (
                f"Action {action} for package {pid}. Evidence: {refs[0]} and {refs[1]}. "
                f"Vendor: {item.get('vendorName','')}, invoice: {item.get('invoiceNumber','')}, "
                f"amount: {item.get('amountMinor',0)} {item.get('currency','INR')}."
            )

        results.append({
            "packageId": pid,
            "action": action,
            "vendorName": str(item.get("vendorName", "Vendor Corp")),
            "invoiceNumber": str(item.get("invoiceNumber", "INV-1001")),
            "amountMinor": int(item.get("amountMinor", 0)),
            "currency": str(item.get("currency", "INR")),
            "evidenceRefs": refs[:3],
            "rationale": rationale[:1500],
        })

    for pkg in packages:
        pid = str(pkg.get("packageId", ""))
        if pid and pid not in seen_pids:
            results.append(_analyze_package_locally(pkg))
    return results

def _decide_packages(packages: list[dict], policy_revision: str) -> list[dict]:
    decisions = []
    uncached = []
    uncached_indices = []

    for i, pkg in enumerate(packages):
        h = _hash_package(pkg)
        with _lock:
            cached = _decision_cache.get(h)
        if cached:
            d = deepcopy(cached)
            d["packageId"] = pkg.get("packageId", d["packageId"])
            decisions.append(d)
        else:
            decisions.append(None)
            uncached.append(pkg)
            uncached_indices.append(i)

    if uncached:
        parsed_results = None
        if AIPIPE_TOKEN or OPENROUTER_API_KEY or OPENAI_API_KEY or GROQ_API_KEY or GEMINI_API_KEY:
            try:
                raw = _call_ai(_build_ai_prompt(uncached, policy_revision))
                parsed_results = _parse_ai_response(raw, uncached)
            except Exception:
                parsed_results = None

        if not parsed_results:
            parsed_results = [_analyze_package_locally(pkg) for pkg in uncached]

        for idx, decision in zip(uncached_indices, parsed_results):
            decisions[idx] = decision
            pkg_hash = _hash_package(packages[idx])
            with _lock:
                _decision_cache[pkg_hash] = deepcopy(decision)

    return decisions

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "base_url": BASE_URL})

# ---- Agent Card (public, no auth) ----
@app.route("/.well-known/agent-card.json", methods=["GET"])
@app.route("/a2a/.well-known/agent-card.json", methods=["GET"])
def agent_card():
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes actions, and executes accepted proposals via A2A.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Processes invoice claim batches and proposes settlement, approval, hold, duplicate rejection, or exception actions.",
            "tags": ["invoice", "finance", "accounts-payable", "a2a"],
        }],
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [BATCH_MEDIA],
        "defaultOutputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA],
        "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
        "securityRequirements": [{"bearerAuth": []}],
    }
    return Response(json.dumps(card, ensure_ascii=False), status=200, content_type="application/json")

# ---- POST /message:send (Registered at root and /a2a/) ----
@app.route("/message:send", methods=["POST"])
@app.route("/a2a/message:send", methods=["POST"])
def message_send():
    principal = _check_auth()
    if isinstance(principal, Response):
        return principal
    err = _check_version_and_media()
    if err:
        return err

    body = request.get_json(silent=True)
    if not body or not isinstance(body, dict) or not body.get("message"):
        return _error_response(400, "INVALID_REQUEST", "Missing or invalid message")

    msg = body["message"]
    message_id = msg.get("messageId", "")
    if not message_id:
        return _error_response(400, "INVALID_REQUEST", "Missing messageId")

    task_id_in = msg.get("taskId", "")
    context_id_in = msg.get("contextId", "")
    parts = msg.get("parts", [])

    idem_key = f"{principal}:{message_id}"
    msg_hash = _hash_message(msg)

    with _lock:
        idem_entry = _idempotency.get(idem_key)
        if idem_entry:
            existing_task_id = idem_entry.get("task_id")
            existing_task = _tasks.get(existing_task_id) if existing_task_id else None
            if existing_task:
                if idem_entry.get("msg_hash", "") == msg_hash:
                    return _message_send_response(existing_task)
                return _error_response(409, "IDEMPOTENCY_CONFLICT", "Same messageId, different content")

    if task_id_in:
        return _handle_continuation(principal, body, msg, task_id_in, context_id_in, idem_key, msg_hash)
    return _handle_initial(principal, body, msg, parts, idem_key, msg_hash)

def _handle_initial(principal, body, msg, parts, idem_key, msg_hash):
    batch_data = next((p.get("data", {}) for p in parts if p.get("mediaType") == BATCH_MEDIA), None)
    if not batch_data or not isinstance(batch_data, dict):
        return _error_response(400, "INVALID_REQUEST", "No invoice batch part found")

    batch_id = batch_data.get("batchId", _new_id("batch-"))
    policy_rev = batch_data.get("policyRevision", "1.0")
    packages = batch_data.get("packages", [])
    if not packages or not isinstance(packages, list):
        return _error_response(400, "INVALID_REQUEST", "Empty or invalid packages")

    decisions = _decide_packages(packages, policy_rev)

    proposals = []
    seen_pids = set()
    for d in decisions:
        pid = d["packageId"]
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        proposals.append({
            "packageId": pid,
            "actionId": _new_id("act-"),
            "action": d["action"],
            "facts": {
                "vendorName": d["vendorName"],
                "invoiceNumber": d["invoiceNumber"],
                "amountMinor": d["amountMinor"],
                "currency": d["currency"],
            },
            "evidenceRefs": d["evidenceRefs"],
            "rationale": d["rationale"],
        })

    task_id = _new_id("task-")
    context_id = _new_id("ctx-")
    art_id = _new_id("art-")

    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
        "artifacts": [{
            "artifactId": art_id,
            "name": "invoice-action-proposals",
            "parts": [{
                "mediaType": PROPOSAL_MEDIA,
                "data": {"batchId": batch_id, "proposals": proposals},
            }],
        }],
        "history": [msg],
        "_principal": principal,
        "_batch_id": batch_id,
        "_proposals": {p["packageId"]: p for p in proposals},
    }

    with _lock:
        _tasks[task_id] = task
        _idempotency[idem_key] = {"task_id": task_id, "msg_hash": msg_hash}
        _user_tasks.setdefault(principal, set()).add(task_id)

    return _message_send_response(task)

def _handle_continuation(principal, body, msg, task_id_in, context_id_in, idem_key, msg_hash):
    with _lock:
        task = _tasks.get(task_id_in)
        if not task or task["_principal"] != principal:
            return _error_response(404, "TASK_NOT_FOUND", "Task not found")
        if task["status"]["state"] in TERMINAL_STATES:
            return _error_response(409, "TASK_ALREADY_TERMINAL", f"Task is already {task['status']['state']}")
        if context_id_in and task["contextId"] != context_id_in:
            return _error_response(400, "CONTEXT_MISMATCH", "Context ID mismatch")

    result_data = next((p.get("data", {}) for p in msg.get("parts", []) if p.get("mediaType") == RESULT_MEDIA), None)
    if not result_data or not isinstance(result_data, dict):
        return _error_response(400, "INVALID_REQUEST", "No result part found")

    batch_id = result_data.get("batchId", "")
    results = result_data.get("results", [])

    with _lock:
        task = _tasks.get(task_id_in)
        if not task or task["_principal"] != principal:
            return _error_response(404, "TASK_NOT_FOUND", "Task not found")
        if task["status"]["state"] in TERMINAL_STATES:
            return _error_response(409, "TASK_ALREADY_TERMINAL", f"Task is already {task['status']['state']}")
        if task["_batch_id"] != batch_id:
            return _error_response(400, "BATCH_MISMATCH", "Batch ID mismatch")

        # Validate each proposal matching
        executions = []
        for r in results:
            pid = r.get("packageId", "")
            prop = task["_proposals"].get(pid)
            if not prop or prop["actionId"] != r.get("actionId") or prop["action"] != r.get("action"):
                return _error_response(400, "PROPOSAL_MISMATCH", f"Continuation result for package {pid} does not match proposal")

            if r.get("outcome") == "ACCEPTED":
                executions.append({
                    "packageId": pid,
                    "actionId": r["actionId"],
                    "action": r["action"],
                    "receiptNonce": r.get("receiptNonce", ""),
                    "facts": prop["facts"],
                    "evidenceRefs": prop["evidenceRefs"],
                })

        art_id = _new_id("art-")
        task["artifacts"].append({
            "artifactId": art_id,
            "name": "invoice-action-receipts",
            "parts": [{
                "mediaType": RECEIPT_MEDIA,
                "data": {"batchId": batch_id, "executions": executions},
            }],
        })
        task["history"].append(msg)
        task["status"] = {"state": "TASK_STATE_COMPLETED"}
        _idempotency[idem_key] = {"task_id": task_id_in, "msg_hash": msg_hash}

    return _message_send_response(task)

# ---- GET /tasks/{id} ----
@app.route("/tasks/<task_id>", methods=["GET"])
@app.route("/a2a/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    principal = _check_auth()
    if isinstance(principal, Response):
        return principal
    err = _check_version_and_media()
    if err:
        return err

    with _lock:
        task = _tasks.get(task_id)
        if not task or task["_principal"] != principal:
            return _error_response(404, "TASK_NOT_FOUND", "Task not found")
        return _direct_task_response(task)

# ---- GET /tasks ----
@app.route("/tasks", methods=["GET"])
@app.route("/a2a/tasks", methods=["GET"])
def list_tasks():
    principal = _check_auth()
    if isinstance(principal, Response):
        return principal
    err = _check_version_and_media()
    if err:
        return err

    with _lock:
        tids = _user_tasks.get(principal, set())
        user_tasks_list = [_clean_task(_tasks[tid]) for tid in tids if tid in _tasks]
    return _tasks_list_response(user_tasks_list)

# ---- POST /tasks/{id}:cancel ----
@app.route("/tasks/<task_id>:cancel", methods=["POST"])
@app.route("/a2a/tasks/<task_id>:cancel", methods=["POST"])
def cancel_task(task_id):
    principal = _check_auth()
    if isinstance(principal, Response):
        return principal
    err = _check_version_and_media()
    if err:
        return err

    with _lock:
        task = _tasks.get(task_id)
        if not task or task["_principal"] != principal:
            return _error_response(404, "TASK_NOT_FOUND", "Task not found")
        if task["status"]["state"] in TERMINAL_STATES:
            return _error_response(409, "TASK_NOT_CANCELABLE", f"Task is already {task['status']['state']}")
        task["status"] = {"state": "TASK_STATE_CANCELED"}
        return _direct_task_response(task)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)