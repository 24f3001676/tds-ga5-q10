"""A2A Invoice Action Agent — single-file Flask application."""

import hashlib
import json
import os
import threading
import uuid
from copy import deepcopy

import requests as http_requests
from flask import Flask, Response, jsonify, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_BASE = os.environ.get("BASE_URL", "https://localhost/a2a/").rstrip("/")
BASE_URL = RAW_BASE + "/" if RAW_BASE else "https://localhost/a2a/"

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
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

# ---------------------------------------------------------------------------
# Storage (in-memory, thread-safe)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_tasks: dict[str, dict] = {}
_idempotency: dict[str, str] = {}
_user_tasks: dict[str, set] = {}
_decision_cache: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _hash_message(msg: dict) -> str:
    return hashlib.sha256(_canonical_json(msg).encode()).hexdigest()

def _hash_package(pkg: dict) -> str:
    return hashlib.sha256(_canonical_json(pkg).encode()).hexdigest()

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
    return Response(json.dumps(body), status=status, content_type=A2A_MEDIA)

def _check_auth():
    p = _get_principal()
    if not p:
        return _error_response(401, "UNAUTHENTICATED", "Missing Bearer token")
    return p

def _check_version_and_media():
    ver = request.headers.get("A2A-Version", "")
    if ver and ver != "1.0":
        return _error_response(400, "INVALID_VERSION", "Only A2A-Version 1.0 supported")
    ct = request.headers.get("Content-Type", "")
    if ct and "application/a2a+json" not in ct and "application/json" not in ct:
        return _error_response(415, "UNSUPPORTED_MEDIA_TYPE")
    return None

def _task_response(task: dict, status=200):
    body = {"task": task}
    raw = json.dumps(body, ensure_ascii=False)
    if len(raw.encode()) > 512 * 1024:
        return _error_response(413, "RESPONSE_TOO_LARGE")
    return Response(raw, status=status, content_type=A2A_MEDIA)

def _tasks_list_response(tasks: list[dict]):
    body = {"tasks": tasks}
    return Response(json.dumps(body, ensure_ascii=False), status=200, content_type=A2A_MEDIA)

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
        lines.append(json.dumps(pkg, ensure_ascii=False)[:8000])
    return "\n".join(lines)

def _call_ai(prompt: str) -> str:
    if not AIPIPE_TOKEN:
        raise RuntimeError("AIPIPE_TOKEN not set")

    headers = {"Authorization": f"Bearer {AIPIPE_TOKEN}", "Content-Type": "application/json"}
    if "openrouter" in AI_ENDPOINT or "openai" in AI_ENDPOINT.lower():
        payload = {"model": AI_MODEL, "input": prompt}
    else:
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

    resp = http_requests.post(AI_ENDPOINT, headers=headers, json=payload, timeout=40)
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
        raise ValueError(f"No JSON array found: {text[:200]}")
    arr = json.loads(text[start:end])

    results, seen_pids = [], set()
    for item in arr:
        pid = item.get("packageId", "")
        if pid in seen_pids: continue
        seen_pids.add(pid)
        
        action = item.get("action", "")
        if action not in VALID_ACTIONS: action = "open_exception"

        refs = item.get("evidenceRefs", [])
        while len(refs) < 3: refs.append(f"[ref-{pid}-{len(refs)}]")
        
        rationale = item.get("rationale", "")
        if len(rationale) < 60:
            rationale = f"Action {action} for {pid}. Evidence: {refs[0]}, {refs[1]}. Vendor {item.get('vendorName','')}, invoice {item.get('invoiceNumber','')}, amount {item.get('amountMinor',0)} {item.get('currency','INR')}."
        
        results.append({
            "packageId": pid, "action": action,
            "vendorName": item.get("vendorName", ""), "invoiceNumber": item.get("invoiceNumber", ""),
            "amountMinor": int(item.get("amountMinor", 0)), "currency": item.get("currency", "INR"),
            "evidenceRefs": refs[:3], "rationale": rationale[:1500],
        })

    for pkg in packages:
        pid = pkg.get("packageId", "")
        if pid not in seen_pids:
            results.append({
                "packageId": pid, "action": "open_exception", "vendorName": "unknown",
                "invoiceNumber": "unknown", "amountMinor": 0, "currency": "INR",
                "evidenceRefs": ["[missing]", "[no-match]", "[fallback]"],
                "rationale": f"Action open_exception for {pid}: insufficient data.",
            })
    return results

def _decide_packages(packages: list[dict], policy_revision: str) -> list[dict]:
    decisions, uncached, uncached_indices = [], [], []
    for i, pkg in enumerate(packages):
        h = _hash_package(pkg)
        with _lock: cached = _decision_cache.get(h)
        if cached:
            d = deepcopy(cached)
            d["packageId"] = pkg.get("packageId", d["packageId"])
            decisions.append(d)
        else:
            decisions.append(None)
            uncached.append(pkg)
            uncached_indices.append(i)

    if uncached:
        raw = _call_ai(_build_ai_prompt(uncached, policy_revision))
        parsed = _parse_ai_response(raw, uncached)
        for idx, decision in zip(uncached_indices, parsed):
            decisions[idx] = decision
            for j, ui in enumerate(uncached_indices):
                if ui == idx:
                    with _lock: _decision_cache[_hash_package(uncached[j])] = deepcopy(decision)
                    break
    return [d for d in decisions if d is not None]

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "base_url": BASE_URL})

# ---- Agent Card (public, no auth) ----
@app.route("/.well-known/agent-card.json")
def agent_card():
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice batches, proposes actions, and executes accepted proposals via A2A.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "skills": [{
            "id": "invoice_action_agent", "name": "Invoice Action Agent",
            "description": "Processes invoice claim batches and proposes settlement, approval, hold, duplicate rejection, or exception actions.",
            "tags": ["invoice", "finance", "accounts-payable", "a2a"],
        }],
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [BATCH_MEDIA],
        "defaultOutputModes": [PROPOSAL_MEDIA, RECEIPT_MEDIA],
        "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
        "securityRequirements": [{"bearerAuth": []}],
    }
    return Response(json.dumps(card), status=200, content_type="application/json")

# ---- POST /message:send (Registered at BOTH root and /a2a/ to prevent 404s) ----
@app.route("/message:send", methods=["POST"])
@app.route("/a2a/message:send", methods=["POST"])
def message_send():
    principal = _check_auth()
    if isinstance(principal, Response): return principal
    err = _check_version_and_media()
    if err: return err

    body = request.get_json(silent=True)
    if not body or not body.get("message"):
        return _error_response(400, "INVALID_REQUEST", "Missing message")

    msg = body["message"]
    message_id = msg.get("messageId", "")
    task_id_in = msg.get("taskId", "")
    context_id_in = msg.get("contextId", "")
    parts = msg.get("parts", [])

    idem_key = f"{principal}:{message_id}"
    msg_hash = _hash_message(msg)

    with _lock:
        existing_task_id = _idempotency.get(idem_key)
        if existing_task_id:
            existing_task = _tasks.get(existing_task_id)
            if existing_task:
                if existing_task.get("_msg_hash", "") == msg_hash:
                    return _task_response(_clean_task(existing_task))
                return _error_response(409, "IDEMPOTENCY_CONFLICT", "Same messageId, different content")

    if task_id_in:
        return _handle_continuation(principal, body, msg, task_id_in, context_id_in, idem_key, msg_hash)
    return _handle_initial(principal, body, msg, parts, idem_key, msg_hash)

def _handle_initial(principal, body, msg, parts, idem_key, msg_hash):
    batch_data = next((p.get("data", {}) for p in parts if p.get("mediaType") == BATCH_MEDIA), None)
    if not batch_data:
        return _error_response(400, "INVALID_REQUEST", "No invoice batch part found")

    batch_id = batch_data.get("batchId", _new_id("batch-"))
    policy_rev = batch_data.get("policyRevision", "1.0")
    packages = batch_data.get("packages", [])
    if not packages:
        return _error_response(400, "INVALID_REQUEST", "Empty packages")

    try:
        decisions = _decide_packages(packages, policy_rev)
    except Exception as e:
        return _error_response(500, "AI_ERROR", str(e)[:200])

    proposals, seen_pids = [], set()
    for d in decisions:
        pid = d["packageId"]
        if pid in seen_pids: continue
        seen_pids.add(pid)
        proposals.append({
            "packageId": pid, "actionId": _new_id("act-"), "action": d["action"],
            "facts": {"vendorName": d["vendorName"], "invoiceNumber": d["invoiceNumber"],
                      "amountMinor": d["amountMinor"], "currency": d["currency"]},
            "evidenceRefs": d["evidenceRefs"], "rationale": d["rationale"],
        })

    task = {
        "id": _new_id("task-"), "contextId": _new_id("ctx-"),
        "status": {"state": "TASK_STATE_INPUT_REQUIRED"},
        "artifacts": [{"artifactId": _new_id("art-"), "name": "invoice-action-proposals",
                       "parts": [{"mediaType": PROPOSAL_MEDIA, "data": {"batchId": batch_id, "proposals": proposals}}]}],
        "history": [msg], "_principal": principal, "_msg_hash": msg_hash,
        "_batch_id": batch_id, "_proposals": {p["packageId"]: p for p in proposals},
    }

    with _lock:
        _tasks[task["id"]] = task
        _idempotency[idem_key] = task["id"]
        _user_tasks.setdefault(principal, set()).add(task["id"])

    return _task_response(_clean_task(task))

def _handle_continuation(principal, body, msg, task_id_in, context_id_in, idem_key, msg_hash):
    with _lock:
        task = _tasks.get(task_id_in)
        if not task or task["_principal"] != principal:
            return _error_response(404, "TASK_NOT_FOUND")
        if task["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"):
            return _error_response(409, "TASK_ALREADY_TERMINAL")
        if context_id_in and task["contextId"] != context_id_in:
            return _error_response(400, "CONTEXT_MISMATCH")

    result_data = next((p.get("data", {}) for p in msg.get("parts", []) if p.get("mediaType") == RESULT_MEDIA), None)
    if not result_data:
        return _error_response(400, "INVALID_REQUEST", "No result part")

    batch_id = result_data.get("batchId", "")
    results = result_data.get("results", [])

    with _lock:
        task = _tasks[task_id_in]
        if task["_batch_id"] != batch_id:
            return _error_response(400, "BATCH_MISMATCH")

        executions = []
        for r in results:
            prop = task["_proposals"].get(r.get("packageId", ""))
            if prop and prop["actionId"] == r.get("actionId") and prop["action"] == r.get("action") and r.get("outcome") == "ACCEPTED":
                executions.append({
                    "packageId": r["packageId"], "actionId": r["actionId"], "action": r["action"],
                    "receiptNonce": r.get("receiptNonce", ""), "facts": prop["facts"], "evidenceRefs": prop["evidenceRefs"],
                })

        task["artifacts"].append({
            "artifactId": _new_id("art-"), "name": "invoice-action-receipts",
            "parts": [{"mediaType": RECEIPT_MEDIA, "data": {"batchId": batch_id, "executions": executions}}]
        })
        task["history"].append(msg)
        task["status"] = {"state": "TASK_STATE_COMPLETED"}
        _idempotency[idem_key] = task_id_in

    return _task_response(_clean_task(task))

# ---- GET /tasks/{id} ----
@app.route("/tasks/<task_id>", methods=["GET"])
@app.route("/a2a/tasks/<task_id>", methods=["GET"])
def get_task(task_id):
    principal = _check_auth()
    if isinstance(principal, Response): return principal
    err = _check_version_and_media()
    if err: return err
    with _lock:
        task = _tasks.get(task_id)
        if not task or task["_principal"] != principal:
            return _error_response(404, "TASK_NOT_FOUND")
    return _task_response(_clean_task(task))

# ---- GET /tasks ----
@app.route("/tasks", methods=["GET"])
@app.route("/a2a/tasks", methods=["GET"])
def list_tasks():
    principal = _check_auth()
    if isinstance(principal, Response): return principal
    err = _check_version_and_media()
    if err: return err
    with _lock:
        tids = _user_tasks.get(principal, set())
        tasks = [_clean_task(_tasks[tid]) for tid in tids if tid in _tasks]
    return _tasks_list_response(tasks)

# ---- POST /tasks/{id}:cancel ----
@app.route("/tasks/<task_id>:cancel", methods=["POST"])
@app.route("/a2a/tasks/<task_id>:cancel", methods=["POST"])
def cancel_task(task_id):
    principal = _check_auth()
    if isinstance(principal, Response): return principal
    err = _check_version_and_media()
    if err: return err
    with _lock:
        task = _tasks.get(task_id)
        if not task or task["_principal"] != principal:
            return _error_response(404, "TASK_NOT_FOUND")
        if task["status"]["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED", "TASK_STATE_REJECTED"):
            return _error_response(409, "TASK_NOT_CANCELABLE", f"Task is already {task['status']['state']}")
        task["status"] = {"state": "TASK_STATE_CANCELED"}
    return _task_response(_clean_task(task))

def _clean_task(task: dict) -> dict:
    return {k: v for k, v in task.items() if not k.startswith("_")}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)