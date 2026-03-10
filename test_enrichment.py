"""
Local Test Script — Manus Lead Enrichment Service v2
=====================================================
Tests the full AI-powered enrichment pipeline with a sample lead.
Does NOT send data to GHL webhooks — just prints the enrichment output.

Usage:
  python test_enrichment.py

Requires: OPENAI_API_KEY and SERPER_API_KEY in your .env file.
"""

import json
from dotenv import load_dotenv
load_dotenv()

from main import gather_search_results, analyze_lead_with_ai

# --- Sample lead (similar to what Adstra GHL would send) ---
TEST_LEAD = {
    "first_name": "Kimberly",
    "last_name":  "Rossbach",
    "email":      "rossbachkim06@gmail.com",
    "phone":      "+18047698851",
}

print("=" * 65)
print("Manus Lead Enrichment Service v2 — Local Test")
print("=" * 65)
print(f"\nInput Lead:")
print(json.dumps(TEST_LEAD, indent=2))

print("\n[1/2] Running web searches...")
search_results = gather_search_results(
    TEST_LEAD["first_name"],
    TEST_LEAD["last_name"],
    TEST_LEAD["email"],
    TEST_LEAD["phone"],
)
print(f"      Found {len(search_results.splitlines())} lines of search data.")

print("\n[2/2] Analyzing with AI...")
enrichment = analyze_lead_with_ai(
    TEST_LEAD["first_name"],
    TEST_LEAD["last_name"],
    TEST_LEAD["email"],
    TEST_LEAD["phone"],
    search_results,
)

print("\nEnrichment Result:")
print(json.dumps(enrichment, indent=2))
print("\n" + "=" * 65)
print("Test complete. This is what would be sent to both GHL accounts.")
print("=" * 65)
