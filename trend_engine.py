"""
trend_engine.py
Calculates moving average, linear trend, and days-until-stockout
for each clinic, using plain statistics — no AI involved here.
This output becomes the structured input Gemini reasons over in Block 3.
"""

import pandas as pd
import numpy as np

def calculate_trend(stock_series):
    """
    Fits a simple linear regression (y = mx + b) over a stock series.
    Returns the slope (units/day) — negative means declining stock.
    """
    days = np.arange(len(stock_series))
    # np.polyfit with degree 1 = linear regression, returns [slope, intercept]
    slope, intercept = np.polyfit(days, stock_series, 1)
    return slope, intercept

def calculate_moving_average(stock_series, window=5):
    """Smooths the series with a rolling average to reduce noise impact."""
    return pd.Series(stock_series).rolling(window=window, min_periods=1).mean().iloc[-1]

def days_until_stockout(current_stock, slope):
    """
    Projects forward using the linear trend.
    If slope >= 0 (stock stable/rising), there's no stockout risk from trend.
    If slope < 0, calculate how many days until stock hits zero.
    """
    if slope >= 0:
        return None  # not declining, no projection needed
    days_left = current_stock / abs(slope)
    return round(days_left, 1)

def classify_risk(days_left):
    """Simple rule-based risk tiers based on projected days until stockout."""
    if days_left is None:
        return "Healthy"
    elif days_left <= 7:
        return "Critical"
    elif days_left <= 14:
        return "Warning"
    else:
        return "Healthy"

def analyze_clinics(csv_path="clinic_data.csv", recent_days=10):
    """
    Main function: reads the clinic CSV, calculates trend metrics per clinic
    using the most recent `recent_days` of data, and returns a summary DataFrame.
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    results = []
    for clinic_id in df["clinic_id"].unique():
        clinic_df = df[df["clinic_id"] == clinic_id].sort_values("date")
        recent = clinic_df.tail(recent_days)

        stock_series = recent["medicine_stock_units"].values
        current_stock = stock_series[-1]
        capacity = recent["medicine_stock_capacity"].iloc[-1]

        slope, _ = calculate_trend(stock_series)
        moving_avg = calculate_moving_average(stock_series)
        days_left = days_until_stockout(current_stock, slope)
        risk = classify_risk(days_left)

        # Latest snapshot of the other tracked fields (beds, staff, footfall)
        latest = clinic_df.iloc[-1]

        results.append({
            "clinic_id": clinic_id,
            "clinic_name": latest["clinic_name"],
            "country": latest["country"],
            "current_stock": round(current_stock, 1),
            "stock_capacity": capacity,
            "stock_pct": round((current_stock / capacity) * 100, 1),
            "moving_avg_stock": round(moving_avg, 1),
            "daily_trend_units": round(slope, 2),  # negative = declining
            "days_until_stockout": days_left,
            "risk_level": risk,
            "beds_available": int(latest["beds_available"]),
            "beds_total": int(latest["beds_total"]),
            "staff_present": int(latest["staff_present"]),
            "staff_total": int(latest["staff_total"]),
            "recent_avg_footfall": round(recent["patient_footfall"].mean(), 1),
        })

    return pd.DataFrame(results).sort_values("days_until_stockout", na_position="last")

if __name__ == "__main__":
    summary = analyze_clinics()
    print(summary.to_string(index=False))
    summary.to_csv("clinic_trend_summary.csv", index=False)
    print("\nSaved trend summary to clinic_trend_summary.csv")