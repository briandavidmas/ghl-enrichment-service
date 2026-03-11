"""
Manus AI Lead Enrichment Service v4.0
======================================
This FastAPI service receives a lead from Adstra GHL via webhook, performs
AI-powered web research to enrich the lead (business ownership, business name,
industry, location, online presence, confidence level, research notes), then
updates the contact directly via the GHL API in BOTH Centerfy GHL and Adstra GHL.

Using the GHL API directly (instead of inbound webhooks) ensures:
- No duplicate contacts are created
- No workflow loops are triggered
- Updates are applied precisely to the correct contact

Environment Variables Required:
  OPENAI_API_KEY          : OpenAI API key (for GPT-4 research + analysis)
  SERPER_API_KEY          : Serper.dev API key (for Google Search results)
  ADSTRA_GHL_API_KEY      : Adstra GHL Private Integration API key
  ADSTRA_GHL_LOCATION_ID  : Adstra GHL Location ID
  CENTERFY_GHL_API_KEY    : Centerfy GHL Private Integration API key
  CENTERFY_GHL_LOCATION_ID: Centerfy GHL Location ID

GHL Custom Field IDs (Adstra):
  business_owner          -> contact.business_owner
  company_name            -> contact.company_name
  do_you_have_a_business  -> contact.do_you_have_a_business
  business_type           -> contact.business_type
  business_location       -> contact.business_location
  online_precense         -> contact.online_precense  (note: GHL typo)
  confidence_level        -> contact.confidence_level
  notes                   -> contact.notes
  enrichment_status       -> contact.enrichment_status
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
OPENAI_API_KEY            = os.getenv("OPENAI_API_KEY", "")
SERPER_API_KEY            = os.getenv("SERPER_API_KEY", "")

# Adstra GHL
ADSTRA_GHL_API_KEY        = os.getenv("ADSTRA_GHL_API_KEY", "")
ADSTRA_GHL_LOCATION_ID    = os.getenv("ADSTRA_GHL_LOCATION_ID", "")

# Centerfy GHL
CENTERFY_GHL_API_KEY      = os.getenv("CENTERFY_GHL_API_KEY", "")
CENTERFY_GHL_LOCATION_ID  = os.getenv("CENTERFY_GHL_LOCATION_ID", "")

# GHL API base URL
GHL_API_BASE = "https://services.leadconnectorhq.com"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(
    title="Manus Lead Enrichment Service",
    description="AI-powered lead enrichment using GHL API v2 for direct contact updates.",
    version="4.0.0",
)


# ---------------------------------------------------------------------------
# GHL API: Find contact by email
# ---------------------------------------------------------------------------
def find_contact_by_email(email: str, api_key: str, location_id: str) -> str | None:
    """
    Searches for a contact by email in a GHL location.
    Returns the contact ID if found, or None.
    """
    if not api_key or not location_id:
        logger.warning("GHL API key or location ID not configured.")
        return None
    try:
        response = requests.get(
            f"{GHL_API_BASE}/contacts/search/duplicate",
            params={"locationId": location_id, "email": email},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Version": "2021-07-28",
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json()
            contact = data.get("contact")
            if contact:
                return contact.get("id")
        logger.warning(f"Contact not found for email {email} — status {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Error searching for contact: {e}")
        return None


# ---------------------------------------------------------------------------
# GHL API: Update contact custom fields
# ---------------------------------------------------------------------------
def update_contact_fields(contact_id: str, enrichment: dict, api_key: str, location_id: str, destination: str) -> bool:
    """
    Updates a GHL contact's custom fields using the GHL API v2.
    Uses the customFields array format for custom fields.
    """
    if not api_key or not location_id or not contact_id:
        logger.warning(f"{destination}: Missing API key, location ID, or contact ID — skipping.")
        return False

    is_owner = enrichment.get("is_business_owner", False)
    confidence = enrichment.get("confidence_level", "Low")

    # Custom fields use the key format (the part after "contact.")
    custom_fields = [
        {"key": "business_owner",        "field_value": "Yes" if is_owner else "No"},
        {"key": "company_name",          "field_value": enrichment.get("business_name", "")},
        {"key": "do_you_have_a_business","field_value": "Yes" if is_owner else "No"},
        {"key": "business_type",         "field_value": enrichment.get("business_type", "")},
        {"key": "business_location",     "field_value": enrichment.get("business_location", "")},
        {"key": "online_precense",       "field_value": ", ".join(enrichment.get("online_presence", []))},
        {"key": "confidence_level",      "field_value": confidence},
        {"key": "notes",                 "field_value": enrichment.get("research_notes", "")},
        {"key": "enrichment_status",     "field_value": "Enriched" if confidence in ("High", "Medium") else "Low Confidence"},
    ]

    payload = {"customFields": custom_fields}

    try:
        response = requests.put(
            f"{GHL_API_BASE}/contacts/{contact_id}",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Version": "2021-07-28",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        if response.status_code in (200, 201):
            logger.info(f"Successfully updated contact {contact_id} in {destination}")
            return True
        else:
            logger.error(f"{destination} API update error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to update contact in {destination}: {e}")
        return False


# ---------------------------------------------------------------------------
# GHL API: Create contact if not found
# ---------------------------------------------------------------------------
def create_contact(lead_data: dict, enrichment: dict, api_key: str, location_id: str, destination: str) -> str | None:
    """
    Creates a new contact in GHL with enrichment data pre-populated.
    Returns the new contact ID or None.
    """
    is_owner = enrichment.get("is_business_owner", False)
    confidence = enrichment.get("confidence_level", "Low")

    payload = {
        "locationId": location_id,
        "firstName": lead_data.get("first_name", ""),
        "lastName": lead_data.get("last_name", ""),
        "email": lead_data.get("email", ""),
        "phone": lead_data.get("phone", ""),
        "customFields": [
            {"key": "business_owner",        "field_value": "Yes" if is_owner else "No"},
            {"key": "company_name",          "field_value": enrichment.get("business_name", "")},
            {"key": "do_you_have_a_business","field_value": "Yes" if is_owner else "No"},
            {"key": "business_type",         "field_value": enrichment.get("business_type", "")},
            {"key": "business_location",     "field_value": enrichment.get("business_location", "")},
            {"key": "online_precense",       "field_value": ", ".join(enrichment.get("online_presence", []))},
            {"key": "confidence_level",      "field_value": confidence},
            {"key": "notes",                 "field_value": enrichment.get("research_notes", "")},
            {"key": "enrichment_status",     "field_value": "Enriched" if confidence in ("High", "Medium") else "Low Confidence"},
        ],
    }

    try:
        response = requests.post(
            f"{GHL_API_BASE}/contacts/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Version": "2021-07-28",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        if response.status_code in (200, 201):
            data = response.json()
            contact_id = data.get("contact", {}).get("id")
            logger.info(f"Created new contact {contact_id} in {destination}")
            return contact_id
        else:
            logger.error(f"{destination} create contact error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Failed to create contact in {destination}: {e}")
        return None


# ---------------------------------------------------------------------------
# Step 1: Web Search via Serper.dev
# ---------------------------------------------------------------------------
def web_search(query: str, num_results: int = 5) -> list[dict]:
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
            return [
                {"title": item.get("title", ""), "link": item.get("link", ""), "snippet": item.get("snippet", "")}
                for item in data.get("organic", [])
            ]
        else:
            logger.error(f"Serper search error {response.status_code}: {response.text}")
            return []
    except Exception as e:
        logger.error(f"Serper search exception: {e}")
        return []


def gather_search_results(first_name: str, last_name: str, email: str, phone: str) -> str:
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
        for r in web_search(query, num_results=4):
            if r["link"] not in seen_links:
                seen_links.add(r["link"])
                all_results.append(r)

    if not all_results:
        return "No search results found."

    return "\n".join(
        f"Title: {r['title']}\nURL: {r['link']}\nSnippet: {r['snippet']}\n"
        for r in all_results
    )


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
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
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
# Step 3: Update contacts via GHL API
# ---------------------------------------------------------------------------
def update_ghl_contact(email: str, lead_data: dict, enrichment: dict, api_key: str, location_id: str, destination: str):
    """
    Finds the contact by email and updates their custom fields via GHL API.
    If not found, creates the contact with enrichment data.
    """
    logger.info(f"Looking up contact in {destination} for email: {email}")
    contact_id = find_contact_by_email(email, api_key, location_id)

    if contact_id:
        logger.info(f"Found contact {contact_id} in {destination} — updating fields")
        update_contact_fields(contact_id, enrichment, api_key, location_id, destination)
    else:
        logger.info(f"Contact not found in {destination} — creating new contact")
        create_contact(lead_data, enrichment, api_key, location_id, destination)


# ---------------------------------------------------------------------------
# Step 4: Full Enrichment Pipeline (runs in background)
# ---------------------------------------------------------------------------
def run_enrichment_pipeline(lead_data: dict):
    """
    Orchestrates the full enrichment pipeline:
    1. Gather web search results for the lead
    2. Analyze with AI to produce structured enrichment
    3. Update contact in Adstra GHL via API
    4. Update contact in Centerfy GHL via API
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
    logger.info(
        f"AI enrichment complete for {email} — "
        f"Business Owner: {enrichment.get('is_business_owner')}, "
        f"Confidence: {enrichment.get('confidence_level')}"
    )
    logger.info(f"Enrichment data: {json.dumps(enrichment, indent=2)}")

    # Step 3: Update both GHL accounts via API
    update_ghl_contact(email, lead_data, enrichment, ADSTRA_GHL_API_KEY,   ADSTRA_GHL_LOCATION_ID,   "Adstra GHL")
    update_ghl_contact(email, lead_data, enrichment, CENTERFY_GHL_API_KEY, CENTERFY_GHL_LOCATION_ID, "Centerfy GHL")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Manus Lead Enrichment Service v4.0"}


@app.post("/webhook/lead")
async def receive_lead_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receives a lead from Adstra GHL via outbound webhook.
    Immediately returns 200 OK, then processes enrichment in the background.

    Expected GHL webhook payload fields:
      first_name, last_name, email, phone
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
        "email":       body.get("email")       or contact.get("email", ""),
        "phone":       body.get("phone")       or contact.get("phone", ""),
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
