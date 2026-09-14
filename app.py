"""
MIRA Flask app — API + animated virtual assistant UI.
Run: python app.py  →  http://127.0.0.1:5000/assistant
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, session
from flask_cors import CORS

from mira_engine import MiraEngine

ROOT = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)
app.secret_key = os.environ.get("MIRA_SECRET_KEY", "mira-dev-change-in-production")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600
CORS(app, supports_credentials=True)

_engine = MiraEngine(csv_path=ROOT / "med.csv")


def _session_to_dict(s) -> dict:
    return {
        "step": s.step,
        "candidate_ids": s.candidate_ids,
        "last_symptom": s.last_symptom,
        "user_days": s.user_days,
        "trend": s.trend,
        "severity": s.severity,
        "impact": s.impact,
        "age_group": s.age_group,
        "fever_level": s.fever_level,
        "medication": s.medication,
        "needs_red_flag": s.needs_red_flag,
        "match_scores": {str(k): v for k, v in s.match_scores.items()},
        "symptom_queries": s.symptom_queries,
    }


def _session_from_dict(data: dict):
    s = _engine.new_session()
    s.step = data.get("step", "welcome")
    s.candidate_ids = list(data.get("candidate_ids", []))
    s.last_symptom = data.get("last_symptom", "")
    s.user_days = data.get("user_days")
    s.trend = data.get("trend", "")
    s.severity = data.get("severity", "")
    s.impact = data.get("impact", "")
    s.age_group = data.get("age_group", "")
    s.fever_level = data.get("fever_level", "")
    s.medication = data.get("medication", "")
    s.needs_red_flag = data.get("needs_red_flag", False)
    s.match_scores = {int(k): float(v) for k, v in data.get("match_scores", {}).items()}
    s.symptom_queries = list(data.get("symptom_queries", []))
    return s


def _get_session():
    if "mira" not in session:
        s = _engine.new_session()
        session["mira"] = _session_to_dict(s)
    return _session_from_dict(session["mira"])


def _save_session(s):
    session["mira"] = _session_to_dict(s)


def _response_payload(result, step: str):
    msgs = result.all_messages()
    return {
        "response": msgs[-1] if msgs else result.message,
        "messages": msgs,
        "suggestions": result.suggestions,
        "input_hint": result.input_hint,
        "done": result.done,
        "step": step,
        "condition": result.condition,
        "suggested_action": result.suggested_action,
        "matches": result.matches,
        "emergency": result.emergency,
        "source": result.source,
        "category": result.category,
    }


# ——— Static site assets (HTML pages in project root) ———

@app.route("/styles.css")
def site_styles():
    return send_from_directory(ROOT, "styles.css", max_age=3600)


@app.route("/site.js")
def site_js():
    return send_from_directory(ROOT, "site.js", max_age=3600)


@app.route("/<page>.html")
def site_page(page: str):
    allowed = {"index", "login", "signup", "faq", "about_page", "contact_us"}
    if page not in allowed:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(ROOT, f"{page}.html")


# ——— Virtual assistant UI ———

@app.route("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.route("/assistant")
def assistant():
    """Animated VA consultation interface."""
    return render_template("assistant.html")


# ——— API ———

@app.route("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "conditions_loaded": len(_engine.df),
            "service": "MIRA consultation API",
        }
    )


@app.route("/start", methods=["GET", "POST"])
@app.route("/api/start", methods=["GET", "POST"])
def start_chat():
    session.pop("mira", None)
    s = _engine.new_session()
    session["mira"] = _session_to_dict(s)
    welcome = _engine.get_welcome()
    return jsonify(_response_payload(welcome, "welcome"))


@app.route("/consult", methods=["POST"])
@app.route("/api/consult", methods=["POST"])
def consult():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or payload.get("symptom") or "").strip()
    if not message:
        return jsonify(
            {
                "response": "Please provide a message.",
                "done": False,
                "step": session.get("mira", {}).get("step", "welcome"),
            }
        ), 400

    s = _get_session()
    result = _engine.process(s, message)
    step = s.step if not result.done else "done"

    if result.done:
        session.pop("mira", None)
    else:
        _save_session(s)

    return jsonify(_response_payload(result, step))


@app.route("/duration", methods=["POST"])
@app.route("/severity", methods=["POST"])
@app.route("/risk", methods=["POST"])
def legacy_step():
    payload = request.get_json(silent=True) or {}
    text = (
        payload.get("duration")
        or payload.get("severity")
        or payload.get("risk_factor")
        or payload.get("message")
        or ""
    )
    s = _get_session()
    result = _engine.process(s, str(text))
    step = s.step if not result.done else "done"
    if result.done:
        session.pop("mira", None)
    else:
        _save_session(s)
    return jsonify(_response_payload(result, step))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
