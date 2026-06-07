"""
app.py — DisasterGuard Agent: Web Service Entry Point (for Render deployment)
Exposes a minimal chat UI + JSON API around the DisasterGuardAgent.
Run locally:  python app.py
Run in prod:  gunicorn app:app --bind 0.0.0.0:$PORT
"""
import logging
import os
import uuid

from flask import Flask, jsonify, render_template_string, request

from agent import DisasterGuardAgent

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
agent = DisasterGuardAgent()

PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DisasterGuard Agent</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; max-width: 760px; margin: 0 auto; padding: 16px; background: #0f172a; color: #e2e8f0; }
    h1 { font-size: 1.4rem; }
    #log { border: 1px solid #334155; border-radius: 8px; padding: 12px; height: 60vh; overflow-y: auto; background: #1e293b; white-space: pre-wrap; line-height: 1.4; }
    .msg { margin-bottom: 14px; }
    .you { color: #38bdf8; font-weight: 600; }
    .agent { color: #4ade80; font-weight: 600; }
    form { display: flex; gap: 8px; margin-top: 12px; }
    input[type=text] { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #e2e8f0; }
    button { padding: 10px 18px; border-radius: 6px; border: none; background: #38bdf8; color: #0f172a; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: default; }
  </style>
</head>
<body>
  <h1>🌍 DisasterGuard Agent</h1>
  <p>AI agent for real-time natural-disaster monitoring and crisis management (GDACS, USGS, NASA EONET, ML risk analysis, historical RAG search).</p>
  <div id="log"></div>
  <form id="chat-form">
    <input id="msg" type="text" autocomplete="off" placeholder="Ask about earthquakes, wildfires, tsunami risk..." />
    <button id="send" type="submit">Send</button>
  </form>
  <script>
    const sessionId = "web_" + Math.random().toString(36).slice(2);
    const log = document.getElementById("log");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("msg");
    const button = document.getElementById("send");

    function append(role, text) {
      const div = document.createElement("div");
      div.className = "msg";
      const label = document.createElement("span");
      label.className = role === "you" ? "you" : "agent";
      label.textContent = role === "you" ? "You: " : "DisasterGuard: ";
      div.appendChild(label);
      div.appendChild(document.createTextNode(text));
      log.appendChild(div);
      log.scrollTop = log.scrollHeight;
    }

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      append("you", message);
      input.value = "";
      button.disabled = true;
      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message, session_id: sessionId }),
        });
        const data = await resp.json();
        append("agent", data.response || data.error || "(no response)");
      } catch (err) {
        append("agent", "Network error: " + err);
      } finally {
        button.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE_HTML)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    message = (payload.get("message") or "").strip()
    session_id = payload.get("session_id") or str(uuid.uuid4())

    if not message:
        return jsonify({"error": "message is required"}), 400

    response_text = agent.run(message, session_id=session_id)
    return jsonify({"response": response_text, "session_id": session_id})


@app.route("/api/clear", methods=["POST"])
def clear():
    payload = request.get_json(silent=True) or {}
    session_id = payload.get("session_id") or ""
    agent.clear_history(session_id)
    return jsonify({"status": "cleared", "session_id": session_id})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
