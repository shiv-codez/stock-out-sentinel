"""
generate_data.py
Generates mock clinic data for Stock-Out Sentinel.
Simulates 21 days of daily reports across 8 clinics in 3 countries,
with deliberately shaped stock trends so the forecasting demo has
clear, visually obvious patterns to catch.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Reproducible randomness — same output every time we run this
np.random.seed(42)

# --- Clinic definitions ---
# trend_type controls how each clinic's stock behaves over the 21 days:
#   "declining"      -> steady, catchable decline (hero case for early warning)
#   "sudden_drop"    -> stable, then sharp drop in the last few days (emergency spike)
#   "stable"         -> healthy with minor natural noise (control group)
clinics = [
    {"clinic_id": "IN-01", "clinic_name": "Sitapur PHC",      "country": "India",        "trend_type": "declining",   "capacity": 1000, "beds_total": 20, "staff_total": 8},
    {"clinic_id": "IN-02", "clinic_name": "Rampur PHC",       "country": "India",        "trend_type": "stable",      "capacity": 800,  "beds_total": 15, "staff_total": 6},
    {"clinic_id": "IN-03", "clinic_name": "Deoria PHC",       "country": "India",        "trend_type": "sudden_drop", "capacity": 1200, "beds_total": 25, "staff_total": 10},
    {"clinic_id": "BR-01", "clinic_name": "Vila Nova PHC",    "country": "Brazil",       "trend_type": "declining",   "capacity": 900,  "beds_total": 18, "staff_total": 7},
    {"clinic_id": "BR-02", "clinic_name": "Santa Fe PHC",     "country": "Brazil",       "trend_type": "stable",      "capacity": 1100, "beds_total": 22, "staff_total": 9},
    {"clinic_id": "BR-03", "clinic_name": "Boa Vista PHC",    "country": "Brazil",       "trend_type": "stable",      "capacity": 750,  "beds_total": 14, "staff_total": 6},
    {"clinic_id": "ZA-01", "clinic_name": "Khayelitsha PHC",  "country": "South Africa", "trend_type": "stable",      "capacity": 950,  "beds_total": 19, "staff_total": 8},
    {"clinic_id": "ZA-02", "clinic_name": "Soweto PHC",       "country": "South Africa", "trend_type": "sudden_drop", "capacity": 1050, "beds_total": 21, "staff_total": 9},
]

NUM_DAYS = 21
start_date = datetime(2026, 8, 1)

def generate_stock_series(capacity, trend_type):
    """Generates a 21-day stock series based on the clinic's trend type."""
    stock = np.zeros(NUM_DAYS)
    starting_stock = capacity * 0.85  # clinics start reasonably well-stocked

    if trend_type == "declining":
        # Steady linear decline with small daily noise — a real, catchable trend
        daily_drop = (starting_stock * 0.70) / NUM_DAYS
        for day in range(NUM_DAYS):
            noise = np.random.normal(0, capacity * 0.01)
            stock[day] = starting_stock - (daily_drop * day) + noise

    elif trend_type == "sudden_drop":
        # Stable for ~15 days, then a sharp drop in the last week (emergency spike)
        for day in range(NUM_DAYS):
            if day < 15:
                stock[day] = starting_stock + np.random.normal(0, capacity * 0.01)
            else:
                days_into_drop = day - 15
                stock[day] = starting_stock - (starting_stock * 0.12 * days_into_drop) + np.random.normal(0, capacity * 0.01)

    else:  # stable
        for day in range(NUM_DAYS):
            stock[day] = starting_stock + np.random.normal(0, capacity * 0.015)

    return np.clip(stock, 0, capacity)  # stock can't go negative or above capacity

def generate_footfall(trend_type, day):
    """Patient footfall — higher footfall drives faster stock depletion.
    sudden_drop clinics show a footfall spike in the last week to explain the drop."""
    base = np.random.randint(30, 60)
    if trend_type == "sudden_drop" and day >= 15:
        base += np.random.randint(40, 70)  # emergency surge
    return base

rows = []
for clinic in clinics:
    for day in range(NUM_DAYS):
        date = start_date + timedelta(days=day)
        stock_series = generate_stock_series(clinic["capacity"], clinic["trend_type"])

        beds_available = max(0, clinic["beds_total"] - np.random.randint(0, clinic["beds_total"] // 2 + 1))
        staff_present = max(1, clinic["staff_total"] - np.random.randint(0, 3))

        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "clinic_id": clinic["clinic_id"],
            "clinic_name": clinic["clinic_name"],
            "country": clinic["country"],
            "medicine_stock_units": round(stock_series[day], 1),
            "medicine_stock_capacity": clinic["capacity"],
            "beds_available": beds_available,
            "beds_total": clinic["beds_total"],
            "staff_present": staff_present,
            "staff_total": clinic["staff_total"],
            "patient_footfall": generate_footfall(clinic["trend_type"], day),
        })

df = pd.DataFrame(rows)
df.to_csv("clinic_data.csv", index=False)
print(f"Generated {len(df)} rows across {len(clinics)} clinics in {df['country'].nunique()} countries.")
print(f"Saved to clinic_data.csv")