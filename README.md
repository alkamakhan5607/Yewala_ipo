# Company Check

A web app that researches a company or IPO using Claude with web search and shows a balanced, sourced report: business, financial health, valuation, strengths, risks, red flags, and questions to ask before investing.

## Run it

1. pip install -r requirements.txt
2. Create a .env file with ANTHROPIC_API_KEY=your-key
3. python app.py
4. Open http://localhost:5000

## Deploy

Use gunicorn app:app as the start command and set ANTHROPIC_API_KEY as an environment variable.

## Note

This is an educational research tool, not investment advice. In India, personalised investment advice may need SEBI registration.
