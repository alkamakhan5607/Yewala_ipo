"""Company research assistant: Flask backend + Claude API with web search."""
import json
import os
import re
import time
from collections import defaultdict

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

load_dotenv()

app = Flask(__name__, static_folder="static")
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are an equity research assistant for retail investors in India.
Research the company the user names using web search. Prefer primary sources: the
company's DRHP/RHP, annual reports, NSE/BSE filings, SEBI, and reputable news.

Rules:
- Use only facts you found. If something is unknown, say "Not found".
- Never tell the user to buy, sell, or subscribe. Never give price targets or promise returns.
- Be balanced: show the strongest points on both sides.
- Keep every list item to one or two plain sentences.

Return ONLY one JSON object (no markdown fences) with exactly these keys:
{
  "company": string,
  "is_listed_or_ipo_status": string,
  "summary": string,
  "business_model": string,
  "financial_health": [string],
  "valuation": [string],
  "positives": [string],
  "risks": [string],
  "red_flags": [string],
  "assessment": "Favourable" | "Mixed" | "Cautious" | "Insufficient data",
  "assessment_reason": string,
  "confidence": "Low" | "Medium" | "High",
  "questions_before_investing": [string],
  "sources": [{"title": string, "url": string}]
}"""

# Very small in-memory rate limit: 5 requests per IP per 10 minutes.
HITS = defaultdict(list)
LIMIT, WINDOW = 5, 600


def rate_limited(ip: str) -> bool:
    now = time.time()
    HITS[ip] = [t for t in HITS[ip] if now - t < WINDOW]
    if len(HITS[ip]) >= LIMIT:
        return True
    HITS[ip].append(now)
    return False


def parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The model did not return a report.")
    return json.loads(match.group(0))


def research(company: str) -> dict:
    messages = [{"role": "user", "content": f"Research this company: {company}"}]
    tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}]
    for _ in range(3):  # a long search turn can pause; continue it
        resp = client.messages.create(
            model=MODEL, max_tokens=4000, system=SYSTEM_PROMPT,
            tools=tools, messages=messages,
        )
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": resp.content})
    text = "".join(b.text for b in resp.content if b.type == "text")
    return parse_json(text)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/research")
def api_research():
    company = (request.get_json(silent=True) or {}).get("company", "").strip()
    if not 2 <= len(company) <= 100:
        return jsonify(error="Enter a company name (2 to 100 characters)."), 400
    if rate_limited(request.remote_addr or "unknown"):
        return jsonify(error="Too many requests. Try again in a few minutes."), 429
    try:
        return jsonify(research(company))
    except (ValueError, json.JSONDecodeError):
        return jsonify(error="Could not build a report. Try again or rename the company."), 502
    except anthropic.APIError:
        return jsonify(error="The research service is unavailable. Try again shortly."), 503


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", 5000)))
