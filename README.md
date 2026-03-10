# Manus Lead Enrichment Service v2 

An AI-powered lead enrichment microservice that receives leads from **Adstra GHL**, performs intelligent web research on each lead, and sends the enriched data back to **both Centerfy GHL and Adstra GHL** in real time.

This service replicates and automates the exact format of the **Manus Enriched Leads Report**, producing the following for every lead:

| Field | Description |
|---|---|
| Business Owner? | Yes or No — determined by AI web research |
| Business Name | The name of the business the lead owns or operates |
| Business Type / Industry | Industry classification (e.g., "Hair Salon", "Real Estate Agent") |
| Business Location | City, State |
| Online Presence | URLs found: LinkedIn, Instagram, Facebook, website, etc. |
| Confidence Level | High, Medium, or Low |
| Research Notes | 2–4 sentence AI-written summary of findings |

---

## How It Works

```
Meta Ad → Adstra GHL → [This Service] → Web Search (Serper/Google) → GPT-4 Analysis
                                                                          ↓
                                                               Centerfy GHL (enriched contact)
                                                               Adstra GHL   (enriched contact)
```

1. A lead submits their name, email, and phone via your Meta Ad.
2. Adstra GHL fires an outbound webhook to this service's `/webhook/lead` endpoint.
3. The service runs up to 6 targeted Google searches for the lead (name + "business owner", name + LinkedIn, email + business, etc.).
4. All search results are passed to GPT-4, which analyzes them and produces a structured enrichment profile.
5. The enriched data is sent simultaneously to **Centerfy GHL** and **Adstra GHL** via their respective inbound webhooks.
6. Both GHL accounts update the contact record with the enriched fields.

---

## Setup & Deployment

### 1. Prerequisites

- Python 3.11+
- An [OpenAI](https://platform.openai.com/) account and API key
- A [Serper.dev](https://serper.dev/) account and API key (free tier: 2,500 searches/month)
- Inbound Webhook URLs from both Centerfy GHL and Adstra GHL

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in all four values
```

### 4. Run Locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test Locally

```bash
python test_enrichment.py
```

### 6. Deploy to Production (Railway — Recommended)

1. Create a free account at [railway.app](https://railway.app).
2. Click **New Project** → **Deploy from GitHub Repo** (push this code to a GitHub repo first).
3. In the Railway project settings, add the four environment variables from your `.env` file.
4. Railway will automatically deploy and give you a public URL like `https://your-app.railway.app`.
5. Your webhook endpoint will be: `https://your-app.railway.app/webhook/lead`

---

## GHL Configuration

### Adstra GHL — Outbound Webhook (sends leads TO this service)

1. Go to **Automation → Workflows** in Adstra GHL.
2. Open the workflow triggered by your Meta Ad form.
3. Add a **Webhook** action right after the trigger.
4. Set Method: `POST`, URL: `https://your-app.railway.app/webhook/lead`
5. Save and publish.

### Centerfy GHL — Inbound Webhook (receives enriched data FROM this service)

1. Go to **Automation → Workflows → Create Workflow → Start from Scratch**.
2. Add the **Inbound Webhook** trigger. Copy the URL — this is your `CENTERFY_GHL_WEBHOOK`.
3. Add a **Create/Update Contact** action and map all fields (see Custom Fields section below).
4. Add your existing outbound call and SMS actions after the contact update.
5. Publish the workflow.

### Adstra GHL — Inbound Webhook (receives enriched data back FROM this service)

1. Create a second workflow in Adstra GHL using the **Inbound Webhook** trigger.
2. Copy the URL — this is your `ADSTRA_GHL_WEBHOOK`.
3. Add a **Update Contact** action and map the enriched fields.
4. Publish the workflow.

---

## Custom Fields to Create in Both GHL Accounts

Create these fields under **Settings → Custom Fields** (type: Text) in both Adstra and Centerfy GHL:

| Field Label | Notes |
|---|---|
| Enriched: Business Owner? | "Yes" or "No" |
| Enriched: Business Name | Name of the business |
| Enriched: Business Type | Industry / type |
| Enriched: Business Location | City, State |
| Enriched: Online Presence | Comma-separated URLs |
| Enriched: Confidence Level | High / Medium / Low |
| Enriched: Research Notes | AI-written summary |
| Enrichment Status | "Enriched" or "Low Confidence" |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/webhook/lead` | Receives lead from Adstra GHL |

---

## Cost Estimate

| Service | Cost |
|---|---|
| OpenAI GPT-4.1-mini | ~$0.001–$0.003 per lead |
| Serper.dev (web search) | Free tier: 2,500/month; $50/month for 50,000 |
| Hosting (Railway) | ~$5–$10/month |
| **Total per lead** | **< $0.01 per lead at most volumes** |
