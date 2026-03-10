"""
Manus AI Lead Enrichment Service
==================================
This FastAPI service receives a lead from Adstra GHL via webhook, performs
AI-powered web research to enrich the lead (business ownership, business name,
industry, location, online presence, confidence level, research notes), then
sends the enriched data back to BOTH Centerfy GHL and Adstra GHL.

The enrichment methodology mirrors the Manus Enriched Leads Report format:
  - Full Name
  - Email
  - Phone
  - Business Owner? (Yes / No)
  - Business Name
  - Business Type / Industry
  - Business Location
  - Online Presence Found (URLs)
  - Confidence Level (High / Medium / Low)
  - Research Notes

Environment Variables Required:
  OPENAI_API_KEY          : OpenAI API key (for GPT-4 research + analysis)
  SERPER_API_KEY          : Serper.dev API key (for Google Search results)
  CENTERFY_GHL_WEBHOOK    : Centerfy GHL Inbound Webhook URL
  ADSTRA_GHL_WEBHOOK      : Adstra GHL Inbound Webhook URL (to update original contact)

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import json
import logging
import requests
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "")
SERPER_API_KEY       = os.getenv("SERPER_API_KEY", "")
CENTERFY_GHL_WEBHOOK = os.getenv("CENTERFY_GHL_WEBHOOK", "")
ADSTRA_GHL_WEBHOOK   = os.getenv("ADSTRA_GHL_WEBHOOK", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# OpenAI client — uses OPENAI_API_KEY from environment automatically
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="Manus Lead Enrichment Service",
    description="AI-powered lead enrichment: receives leads from Adstra GHL, researches them, and sends enriched data to Centerfy GHL and Adstra GHL.",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Step 1: Web Search via Serper.dev
# ---------------------------------------------------------------------------
def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Performs a Google search via the Serper.dev API and returns a list of results.
    Each result contains: title, link, snippet.
    """
    if not SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not set — skipping web search.")
        return []
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            results = []
            for item in data.get("organic", []):
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })
            return results
        else:
            logger.error(f"Serper search error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Serper search exception: {e}")
        return []


def gather_search_results(first_name: str, last_name: str, email: str, phone: str) -> str:
    """
    Runs multiple targeted searches for the lead and compiles all results
    into a single text block for the AI to analyze.
    """
    full_name = f"{first_name} {last_name}".strip()
    email_prefix = email.split("@")[0] if "@" in email else ""

    queries = [
        f'"{full_name}" business owner',
        f'"{full_name}" company',
        f'"{email}" business',
        f'"{full_name}" LinkedIn',
        f'"{full_name}" Instagram OR Facebook',
    ]
    if email_prefix and len(email_prefix) > 4:
        queries.append(f'"{email_prefix}" business')

    all_results = []
    seen_links = set()
    for query in queries:
        results = web_search(query, num_results=4)
        for r in results:
            if r["link"] not in seen_links:
                seen_links.add(r["link"])
                all_results.append(r)

    if not all_results:
        return "No search results found."

    compiled = []
    for r in all_results:
        compiled.append(f"Title: {r['title']}\nURL: {r['link']}\nSnippet: {r['snippet']}\n")

    return "\n".join(compiled)


# ---------------------------------------------------------------------------
# Step 2: AI Analysis via GPT-4
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert lead researcher for a credit card processing sales company.
Your job is to analyze web search results about a lead and determine whether they are a business owner,
and if so, gather key details about their business.

You must respond ONLY with a valid JSON object — no markdown, no explanation, just raw JSON.

The JSON must have exactly these fields:
{
  "is_business_owner": true or false,
  "business_name": "string or empty string",
  "business_type": "string describing industry/type or empty string",
  "business_location": "City, State or empty string",
  "online_presence": ["list", "of", "URLs"],
  "confidence_level": "High" or "Medium" or "Low",
  "research_notes": "2-4 sentence summary of what was found and why you made this determination"
}

Guidelines:
- Set is_business_owner to true only if there is clear evidence they own or operate a business.
- Employees, students, or government workers are NOT business owners.
- confidence_level should be High if multiple sources confirm ownership, Medium if one source suggests it, Low if no clear evidence.
- online_presence should list only URLs directly related to this person or their business (LinkedIn, Instagram, Facebook, website, etc.).
- research_notes should be professional, factual, and specific — cite what you found.
- If no business is found, set business_name, business_type, business_location to empty strings and online_presence to [].
"""

def analyze_lead_with_ai(first_name: str, last_name: str, email: str, phone: str, search_results: str) -> dict:
    """
    Sends the search results to GPT-4 for analysis and returns a structured enrichment dict.
    """
    full_name = f"{first_name} {last_name}".strip()

    user_message = f"""Lead Information:
- Full Name: {full_name}
- Email: {email}
- Phone: {phone}

Web Search Results:
{search_results}

Analyze the above search results and determine if this person is a business owner.
Return your findings as a JSON object exactly as specified."""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
        return result
    except json.JSONDecodeError as e:
        logger.error(f"AI returned invalid JSON: {e}")
        return _default_enrichment()
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        return _default_enrichment()


def _default_enrichment() -> dict:
    return {
        "is_business_owner": False,
        "business_name": "",
        "business_type": "",
        "business_location": "",
        "online_presence": [],
        "confidence_level": "Low",
        "research_notes": "Enrichment could not be completed due to a processing error.",
    }


# ---------------------------------------------------------------------------
# Step 3: Send Enriched Data to GHL Webhooks
# ---------------------------------------------------------------------------
def build_ghl_payload(lead_data: dict, enrichment: dict) -> dict:
    """
    Builds the payload to send to GHL webhooks.
    Maps enrichment fields to GHL-friendly field names.
    """
    return {
        # --- Original Lead Fields ---
        "first_name":   lead_data.get("first_name", ""),
        "last_name":    lead_data.get("last_name", ""),
        "full_name":    lead_data.get("full_name", ""),
        "email":        lead_data.get("email", ""),
        "phone":        lead_data.get("phone", ""),
        "lead_source":  lead_data.get("lead_source", "Meta Ad - Adstra"),

        # --- Enriched Fields (matching the Manus Enriched Leads Report format) ---
        "enriched_is_business_owner":  "Yes" if enrichment.get("is_business_owner") else "No",
        "enriched_business_name":      enrichment.get("business_name", ""),
        "enriched_business_type":      enrichment.get("business_type", ""),
        "enriched_business_location":  enrichment.get("business_location", ""),
        "enriched_online_presence":    ", ".join(enrichment.get("online_presence", [])),
        "enriched_confidence_level":   enrichment.get("confidence_level", "Low"),
        "enriched_research_notes":     enrichment.get("research_notes", ""),
        "enrichment_status":           "Enriched" if enrichment.get("confidence_level") != "Low" else "Low Confidence",
    }


def send_to_webhook(url: str, payload: dict, destination: str) -> bool:
    """Sends a JSON payload to a GHL inbound webhook URL."""
    if not url:
        logger.warning(f"{destination} webhook URL not configured — skipping.")
        return False
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code in (200, 201, 202):
            logger.info(f"Successfully sent enriched data to {destination}")
            return True
        else:
            logger.error(f"{destination} webhook error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send to {destination}: {e}")
        return False


# ---------------------------------------------------------------------------
# Step 4: Full Enrichment Pipeline (runs in background)
# ---------------------------------------------------------------------------
def run_enrichment_pipeline(lead_data: dict):
    """
    Orchestrates the full enrichment pipeline:
    1. Gather web search results for the lead
    2. Analyze with AI to produce structured enrichment
    3. Send enriched data to Centerfy GHL
    4. Send enriched data back to Adstra GHL
    """
    email      = lead_data.get("email", "")
    first_name = lead_data.get("first_name", "")
    last_name  = lead_data.get("last_name", "")
    phone      = lead_data.get("phone", "")

    logger.info(f"Starting enrichment pipeline for: {email or phone}")

    # Step 1: Web research
    search_results = gather_search_results(first_name, last_name, email, phone)
    logger.info(f"Gathered search results for {email}")

    # Step 2: AI analysis
    enrichment = analyze_lead_with_ai(first_name, last_name, email, phone, search_results)
    logger.info(f"AI enrichment complete for {email} — Business Owner: {enrichment.get('is_business_owner')}, Confidence: {enrichment.get('confidence_level')}")

    # Step 3: Build payload
    payload = build_ghl_payload(lead_data, enrichment)

    # Step 4: Send to both GHL accounts
    send_to_webhook(CENTERFY_GHL_WEBHOOK, payload, "Centerfy GHL")
    send_to_webhook(ADSTRA_GHL_WEBHOOK,   payload, "Adstra GHL")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Manus Lead Enrichment Service v2.0"}


@app.post("/webhook/lead")
async def receive_lead_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives a lead from Adstra GHL via outbound webhook.
    Immediately returns 200 OK, then processes enrichment in the background.

    Expected GHL webhook payload fields:
      first_name, last_name, full_name, email, phone
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    # Handle both flat and nested GHL payload formats
    contact = body.get("contact", {})
    lead_data = {
        "first_name":  body.get("first_name")  or contact.get("first_name", ""),
        "last_name":   body.get("last_name")   or contact.get("last_name", ""),
        "full_name":   body.get("full_name")   or contact.get("full_name", ""),
        "email":       body.get("email")       or contact.get("email", ""),
        "phone":       body.get("phone")       or contact.get("phone", ""),
        "lead_source": body.get("lead_source", "Meta Ad - Adstra"),
    }

    if not lead_data["email"] and not lead_data["phone"]:
        return JSONResponse(
            status_code=422,
            content={"status": "skipped", "reason": "No email or phone provided."},
        )

    background_tasks.add_task(run_enrichment_pipeline, lead_data)
    logger.info(f"Lead received and queued: {lead_data.get('email') or lead_data.get('phone')}")

    return JSONResponse(
        status_code=200,
        content={"status": "received", "message": "Lead queued for enrichment."},
    )
