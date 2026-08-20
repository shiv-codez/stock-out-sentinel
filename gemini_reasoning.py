"""
gemini_reasoning.py
Takes the trend engine's output for at-risk clinics and asks Gemini to:
1. Explain the risk in plain language
2. Recommend a specific redistribution action (donor clinic + units)
3. Frame it explicitly as AI-recommended, human-approved

Gemini does NOT see raw daily data and does NOT calculate trends —
that's already done in trend_engine.py using plain statistics.
Gemini's job here is interpretation and recommendation, not number-crunching.
"""
import time
import os
import json
import pandas as pd
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def find_donor_clinic(at_risk_clinic, all_clinics_df):
    """
    Finds the best candidate donor clinic: same country (logistics realism),
    healthy stock, not itself. Returns None if no good candidate exists.
    This is plain filtering logic, not AI — Gemini reasons over this shortlist,
    it doesn't invent candidates from nowhere.
    """
    candidates = all_clinics_df[
        (all_clinics_df["country"] == at_risk_clinic["country"]) &
        (all_clinics_df["clinic_id"] != at_risk_clinic["clinic_id"]) &
        (all_clinics_df["risk_level"] == "Healthy") &
        (all_clinics_df["stock_pct"] > 60)
    ].sort_values("stock_pct", ascending=False)

    if len(candidates) == 0:
        return None
    return candidates.iloc[0]

def build_prompt(at_risk_clinic, donor_clinic):
    """Builds a structured prompt with only the computed summary data — no raw logs."""
    donor_info = "No suitable donor clinic found in-country with surplus stock." if donor_clinic is None else f"""
Potential donor clinic: {donor_clinic['clinic_name']} ({donor_clinic['clinic_id']})
- Current stock: {donor_clinic['stock_pct']}% of capacity ({donor_clinic['current_stock']} units)
- Risk level: {donor_clinic['risk_level']}
"""

    prompt = f"""You are assisting a district health administrator reviewing an automated stock-out risk report. You are NOT making decisions — you are producing a recommendation for a human dispatcher to review and approve.

AT-RISK CLINIC:
- Name: {at_risk_clinic['clinic_name']} ({at_risk_clinic['clinic_id']}), {at_risk_clinic['country']}
- Current stock: {at_risk_clinic['stock_pct']}% of capacity ({at_risk_clinic['current_stock']} units)
- Daily trend: {at_risk_clinic['daily_trend_units']} units/day (negative = declining)
- Projected days until stock-out: {at_risk_clinic['days_until_stockout']}
- Risk level: {at_risk_clinic['risk_level']}
- Recent average daily patient footfall: {at_risk_clinic['recent_avg_footfall']}
- Beds available: {at_risk_clinic['beds_available']}/{at_risk_clinic['beds_total']}
- Staff present: {at_risk_clinic['staff_present']}/{at_risk_clinic['staff_total']}

{donor_info}

Respond ONLY with valid JSON in exactly this structure, no other text:
{{
  "plain_language_warning": "A 2-3 sentence explanation of the risk a non-technical health administrator would understand, referencing the actual trend and timeframe.",
  "recommended_action": "A specific, actionable recommendation. If a donor clinic was provided, recommend a specific unit transfer with a suggested quantity. If no donor was found, recommend an alternative action like emergency procurement or escalation.",
  "urgency_note": "One sentence stating this is an AI-generated recommendation requiring human dispatcher approval before any action is taken.",
  "confidence_basis": "One sentence stating what this recommendation is based on (the statistical trend), to keep the reasoning transparent."
}}
"""
    return prompt

def get_recommendation(at_risk_clinic, all_clinics_df):
    """Calls Gemini for one at-risk clinic and returns parsed JSON."""
    donor = find_donor_clinic(at_risk_clinic, all_clinics_df)
    prompt = build_prompt(at_risk_clinic, donor)

    raw_text = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )
            raw_text = response.text.strip()
            break  # success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s backoff
                print(f"  Gemini request failed ({e.__class__.__name__}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Gemini request failed after {max_retries} attempts: {e}")
                raw_text = None

    if raw_text is None:
        return {
            "plain_language_warning": "Unable to reach Gemini after multiple attempts. Please retry.",
            "recommended_action": "Manual review needed — AI service temporarily unavailable.",
            "urgency_note": "N/A",
            "confidence_basis": "N/A",
            "donor_clinic_name": None,
            "donor_clinic_id": None,
        }

    # Gemini sometimes wraps JSON in ```json fences — strip if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
        parsed["donor_clinic_name"] = donor["clinic_name"] if donor is not None else None
        parsed["donor_clinic_id"] = donor["clinic_id"] if donor is not None else None
        return parsed
    except json.JSONDecodeError:
        return {
            "plain_language_warning": "AI response could not be parsed. Raw output logged for review.",
            "recommended_action": "Manual review needed.",
            "urgency_note": "N/A",
            "confidence_basis": "N/A",
            "donor_clinic_name": None,
            "donor_clinic_id": None,
            "_raw_error": raw_text
        }

if __name__ == "__main__":
    df = pd.read_csv("clinic_trend_summary.csv")
    at_risk = df[df["risk_level"].isin(["Critical", "Warning"])]

    for _, clinic in at_risk.iterrows():
        print(f"\n{'='*60}")
        print(f"Analyzing: {clinic['clinic_name']} ({clinic['risk_level']})")
        print('='*60)
        result = get_recommendation(clinic, df)
        print(json.dumps(result, indent=2))