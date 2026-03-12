"""
Manus AI Lead Enrichment Service v4.3
======================================
This FastAPI service receives a lead from Adstra GHL via webhook, performs
AI-powered web research to enrich the lead (business ownership, business name,
industry, location, online presence, confidence level, research notes), then
updates the contact directly via the GHL API in BOTH Centerfy GHL and Adstra GHL.

Using the GHL API directly (instead of inbound webhooks) ensures:
- No duplicate contacts are created
- No workflow loops are triggered
- Updates are applied precisely to the correct contact

v4.3 Changes:
- Phone number fallback lookup when email lookup fails
- Never overwrite companyName with empty string
- Never overwrite existing contact email/phone
- Adstra: create contact if not found (new Meta lead)
- Centerfy: only update existing contacts, never create (contacts sync from Adstra)

Environment Variables Required:
  OPENAI_API_KEY          : OpenAI API key (for GPT-4 research + analysis)
  SERPER_API_KEY          : Serper.dev API key (for Google Search results)
  ADSTRA_GHL_API_KEY      : Adstra GHL Private Integration API key
  ADSTRA_GHL_LOCATION_ID  : Adstra GHL Location ID
  CENTERFY_GHL_API_KEY    : Centerfy GHL Private Integration API key
  CENTERFY_GHL_LOCATION_ID: Centerfy GHL Location ID

GHL Custom Field IDs (Adstra):
  Business Owner          -> IHwhvNck7VKt0kWCDWNG
  Do You Have A Business  -> ddB1BCcaF35uUPDUprl9
  Business Type           -> OCjEBvvuZ7l4mKFQ0vd3
  Business Location       -> PIKRq4AWFLLXoZGEjUjG
  Online Presence         -> (key: contact.online_precense — typo in GHL)
  Confidence Level        -> RmeWNiUf3ctR5xCW9kSv
  Research Notes          -> e2Jab0wmokCZ3NMe5vxw
  Enrichment Status       -> HgLW92yemMLxNpKUWg4d

GHL Custom Field IDs (Centerfy):
  Business Owner          -> aVVjRt3h86fKSlQvkva8
  Business Type           -> NTVcL8yhL9BHtJAexNOf
  Business Location       -> 2ZMMey5prTjWBVGByWmx
  Online Presence         -> xhqfimig2Z3plQ6P4Lh0
  Confidence Level        -> p9h423CxRjv07YgNZQJ9
  Research Notes          -> 0c6xWsEzOMZvtMlDisHL
  Enrichment Status       -> 1lko4IzWhrz6qBRbGNe2
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

# ---------------------------------------------------------------------------
# GHL Custom Field ID Maps (field IDs are required for PUT /contacts/{id})
# ---------------------------------------------------------------------------
# Adstra GHL field IDs
ADSTRA_FIELD_IDS = {
    "business_owner":        "IHwhvNck7VKt0kWCDWNG",
    "do_you_have_a_business":"ddB1BCcaF35uUPDUprl9",
    "business_type":         "OCjEBvvuZ7l4mKFQ0vd3",
    "business_location":     "PIKRq4AWFLLXoZGEjUjG",
    "online_presence":       "online_precense_key",   # uses key fallback (see below)
    "confidence_level":      "RmeWNiUf3ctR5xCW9kSv",
    "research_notes":        "e2Jab0wmokCZ3NMe5vxw",
    "enrichment_status":     "HgLW92yemMLxNpKUWg4d",
}

# Centerfy GHL field IDs
CENTERFY_FIELD_IDS = {
    "business_owner":        "aVVjRt3h86fKSlQvkva8",
    "do_you_have_a_business": None,   # not present in Centerfy
    "business_type":         "NTVcL8yhL9BHtJAexNOf",
    "business_location":     "2ZMMey5prTjWBVGByWmx",
    "online_presence":       "xhqfimig2Z3plQ6P4Lh0",
    "confidence_level":      "p9h423CxRjv07YgNZQJ9",
    "research_notes":        "0c6xWsEzOMZvtMlDisHL",
    "enrichment_status":     "1lko4IzWhrz6qBRbGNe2",
}

app = FastAPI(
    title="Manus Lead Enrichment Service",
    description="AI-powered lead enrichment using GHL API v2 for direct contact updates.",
    version="4.3.0",
)


# ---------------------------------------------------------------------------
# GHL API: Find contact by email
# ---------------------------------------------------------------------------
def find_contact_by_email(email: str, api_key: str, location_id: str) -> str | None:
    """
    Searches for a contact by email in a GHL location using the duplicate-check endpoint.
    Returns the contact ID if found, or None.
    """
    if not email or not api_key or not location_id:
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
                logger.info(f"Found contact by email: {contact.get('id')}")
                return contact.get("id")
        return None
    except Exception as e:
        logger.error(f"Error searching contact by email: {e}")
        return None


# ---------------------------------------------------------------------------
# GHL API: Find contact by phone
# ---------------------------------------------------------------------------
def find_contact_by_phone(phone: str, api_key: str, location_id: str) -> str | None:
    """
    Searches for a contact by phone number in a GHL location.
    Returns the contact ID if found, or None.
    """
    if not phone or not api_key or not location_id:
        return None
    try:
        response = requests.get(
            f"{GHL_API_BASE}/contacts/search/duplicate",
            params={"locationId": location_id, "phone": phone},
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
                logger.info(f"Found contact by phone: {contact.get('id')}")
                return contact.get("id")
        return None
    except Exception as e:
        logger.error(f"Error searching contact by phone: {e}")
        return None


# ---------------------------------------------------------------------------
# GHL API: Find contact by email OR phone (with fallback)
# ---------------------------------------------------------------------------
def find_contact(email: str, phone: str, api_key: str, location_id: str) -> str | None:
    """
    Tries to find a contact by email first, then falls back to phone.
    Returns the contact ID if found, or None.
    """
    contact_id = find_contact_by_email(email, api_key, location_id)
    if contact_id:
        return contact_id
    logger.info(f"Email lookup failed — trying phone fallback for {phone}")
    contact_id = find_contact_by_phone(phone, api_key, location_id)
    return contact_id


# ---------------------------------------------------------------------------
# GHL API: Update contact custom fields
# ---------------------------------------------------------------------------
def update_contact_fields(contact_id: str, enrichment: dict, api_key: str, location_id: str, destination: str) -> bool:
    """
    Updates a GHL contact's custom fields using the GHL API v2.
    Uses the customFields array format with field IDs.
    NEVER overwrites email, phone, or companyName with empty values.
    """
    if not api_key or not location_id or not contact_id:
        logger.warning(f"{destination}: Missing API key, location ID, or contact ID — skipping.")
        return False

    is_owner = enrichment.get("is_business_owner", False)
    confidence = enrichment.get("confidence_level", "Low")
    enrichment_status = "Enriched" if confidence in ("High", "Medium") else "Low Confidence"
    online_presence_str = ", ".join(enrichment.get("online_presence", []))
    business_name = enrichment.get("business_name", "").strip()

    # Select the correct field ID map based on destination
    field_ids = ADSTRA_FIELD_IDS if destination == "Adstra GHL" else CENTERFY_FIELD_IDS

    # Build custom fields list using field IDs (required for PUT /contacts/{id})
    custom_fields = []
    field_values = {
        "business_owner":        "Yes" if is_owner else "No",
        "do_you_have_a_business":"Yes" if is_owner else "No",
        "business_type":         enrichment.get("business_type", ""),
        "business_location":     enrichment.get("business_location", ""),
        "online_presence":       online_presence_str,
        "confidence_level":      confidence,
        "research_notes":        enrichment.get("research_notes", ""),
        "enrichment_status":     enrichment_status,
    }
    for field_name, field_id in field_ids.items():
        if field_id and field_id != "online_precense_key" and field_name in field_values:
            custom_fields.append({"id": field_id, "field_value": field_values[field_name]})

    # Adstra has a typo in the online presence field key — use key fallback
    if destination == "Adstra GHL":
        custom_fields.append({"key": "contact.online_precense", "field_value": online_presence_str})

    # Build payload — only include companyName if AI found a real business name
    payload = {"customFields": custom_fields}
    if business_name:
        payload["companyName"] = business_name

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
# GHL API: Create contact in Adstra only (if not found)
# ---------------------------------------------------------------------------
def create_contact_adstra(lead_data: dict, enrichment: dict) -> str | None:
    """
    Creates a new contact in Adstra GHL with enrichment data pre-populated.
    Only used for Adstra — Centerfy contacts are synced from Adstra, not created directly.
    Returns the new contact ID or None.
    """
    is_owner = enrichment.get("is_business_owner", False)
    confidence = enrichment.get("confidence_level", "Low")
    business_name = enrichment.get("business_name", "").strip()

    payload = {
        "locationId": ADSTRA_GHL_LOCATION_ID,
        "firstName": lead_data.get("first_name", ""),
        "lastName": lead_data.get("last_name", ""),
        "email": lead_data.get("email", ""),
        "phone": lead_data.get("phone", ""),
        "customFields": [
            {"id": "IHwhvNck7VKt0kWCDWNG", "field_value": "Yes" if is_owner else "No"},
            {"id": "ddB1BCcaF35uUPDUprl9", "field_value": "Yes" if is_owner else "No"},
            {"id": "OCjEBvvuZ7l4mKFQ0vd3", "field_value": enrichment.get("business_type", "")},
            {"id": "PIKRq4AWFLLXoZGEjUjG", "field_value": enrichment.get("business_location", "")},
            {"key": "contact.online_precense", "field_value": ", ".join(enrichment.get("online_presence", []))},
            {"id": "RmeWNiUf3ctR5xCW9kSv", "field_value": confidence},
            {"id": "e2Jab0wmokCZ3NMe5vxw", "field_value": enrichment.get("research_notes", "")},
            {"id": "HgLW92yemMLxNpKUWg4d", "field_value": "Enriched" if confidence in ("High", "Medium") else "Low Confidence"},
        ],
    }
    if business_name:
        payload["companyName"] = business_name

    try:
        response = requests.post(
            f"{GHL_API_BASE}/contacts/",
            headers={
                "Authorization": f"Bearer {ADSTRA_GHL_API_KEY}",
                "Version": "2021-07-28",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        if response.status_code in (200, 201):
            data = response.json()
            contact_id = data.get("contact", {}).get("id")
            logger.info(f"Created new contact {contact_id} in Adstra GHL")
            return contact_id
        else:
            logger.error(f"Adstra create contact error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Failed to create contact in Adstra GHL: {e}")
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
def update_ghl_contact(email: str, phone: str, lead_data: dict, enrichment: dict, api_key: str, location_id: str, destination: str, create_if_missing: bool = False):
    """
    Finds the contact by email (then phone as fallback) and updates their custom fields.
    If create_if_missing is True and contact not found, creates a new contact (Adstra only).
    """
    logger.info(f"Looking up contact in {destination} — email: {email}, phone: {phone}")
    contact_id = find_contact(email, phone, api_key, location_id)

    if contact_id:
        logger.info(f"Found contact {contact_id} in {destination} — updating fields")
        update_contact_fields(contact_id, enrichment, api_key, location_id, destination)
    elif create_if_missing:
        logger.info(f"Contact not found in {destination} — creating new contact")
        create_contact_adstra(lead_data, enrichment)
    else:
        logger.warning(f"Contact not found in {destination} — skipping (no creation for this account)")


# ---------------------------------------------------------------------------
# Step 4: Full Enrichment Pipeline (runs in background)
# ---------------------------------------------------------------------------
def run_enrichment_pipeline(lead_data: dict):
    """
    Orchestrates the full enrichment pipeline:
    1. Gather web search results for the lead
    2. Analyze with AI to produce structured enrichment
    3. Update contact in Adstra GHL via API (create if not found)
    4. Update contact in Centerfy GHL via API (update only, never create)
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

    # Step 3: Update Adstra GHL (create if not found — this is the source account for Meta leads)
    update_ghl_contact(
        email, phone, lead_data, enrichment,
        ADSTRA_GHL_API_KEY, ADSTRA_GHL_LOCATION_ID, "Adstra GHL",
        create_if_missing=True
    )

    # Step 4: Update Centerfy GHL (update only — contacts sync from Adstra, never create here)
    update_ghl_contact(
        email, phone, lead_data, enrichment,
        CENTERFY_GHL_API_KEY, CENTERFY_GHL_LOCATION_ID, "Centerfy GHL",
        create_if_missing=False
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Manus Lead Enrichment Service v4.3"}


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
