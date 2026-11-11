"""Company research assistant: Flask backend + Gemini API (free tier) with Google Search."""
import json
import os
import re
import time
from collections import defaultdict

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import errors, types

load_dotenv()

app = Flask(__name__, static_folder="static")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
_client = None

NO_SEARCH_NOTE = (
    "Note: live web search was unavailable, so this report uses the model's "
    "general knowledge and may be out of date."
)

SYSTEM_PROMPT = """You are an equity research assistant for retail investors in India.
Research the company the user names using Google Search. Prefer primary sources: the
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
  "questions_before_investing": [string]
}"""

HITS = defaultdict(list)
LIMIT, WINDOW = 5, 600


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


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


def generate(company: str, use_search: bool):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(google_search=types.GoogleSearch())] if use_search else None,
    )
    return get_client().models.generate_content(
        model=MODEL, contents=f"Research this company: {company}", config=config
    )


def sources_from(resp) -> list:
    found, seen = [], set()
    try:
        chunks = resp.candidates[0].grounding_metadata.grounding_chunks or []
    except (AttributeError, IndexError, TypeError):
        return found
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web and web.uri and web.uri not in seen:
            seen.add(web.uri)
            found.append({"title": web.title or web.uri, "url": web.uri})
    return found


def research(company: str) -> dict:
    searched = True
    try:
        resp = generate(company, True)
    except errors.APIError as e:
        if getattr(e, "code", None) == 429:
            raise
        searched = False
        resp = generate(company, False)
    data = parse_json(resp.text or "")
    data["sources"] = sources_from(resp) if searched else []
    if not searched:
        data["summary"] = f"{NO_SEARCH_NOTE} {data.get('summary', '')}"
    return data


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
    except KeyError:
        return jsonify(error="The server is missing its GEMINI_API_KEY."), 500
    except (ValueError, json.JSONDecodeError):
        return jsonify(error="Could not build a report. Try again or rename the company."), 502
    except errors.APIError as e:
        if getattr(e, "code", None) == 429:
            return jsonify(error="The free daily limit is used up. Try again tomorrow."), 429
        return jsonify(error="The research service is unavailable. Try again shortly."), 503


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", 5000)))
  
