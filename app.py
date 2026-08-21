import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

from trend_engine import analyze_clinics
from gemini_reasoning import get_recommendation

load_dotenv()

st.set_page_config(
    page_title="Stock-Out Sentinel | Health Command Center",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Stock-Out Sentinel: AI-Driven Health Logistics")
st.markdown("### District Primary Health Centre (PHC) Command & Predictive Alert Center")

@st.cache_data
def load_data():
    if not os.path.exists("clinic_data.csv"):
        return None, None
    raw_df = pd.read_csv("clinic_data.csv")
    summary_df = analyze_clinics("clinic_data.csv")  # runs the REAL trend engine
    return raw_df, summary_df

raw_df, summary_df = load_data()

if raw_df is None:
    st.error("⚠️ `clinic_data.csv` not found. Run `python generate_data.py` first.")
else:
    st.sidebar.header("Command Controls")
    selected_country = st.sidebar.selectbox("Select Country Region", summary_df["country"].unique())

    filtered = summary_df[summary_df["country"] == selected_country]
    selected_clinic_name = st.sidebar.selectbox("Select Primary Health Centre", filtered["clinic_name"].unique())

    clinic_row = filtered[filtered["clinic_name"] == selected_clinic_name].iloc[0]

    st.subheader(f"📊 Live Status: {selected_clinic_name}")

    risk_colors = {"Critical": "🔴", "Warning": "🟡", "Healthy": "🟢"}
    st.markdown(f"**Risk Level:** {risk_colors.get(clinic_row['risk_level'], '')} {clinic_row['risk_level']}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Medicine Stock", f"{clinic_row['current_stock']} units", f"{clinic_row['stock_pct']}% of capacity")
    with col2:
        st.metric("Beds Available", f"{clinic_row['beds_available']} / {clinic_row['beds_total']}")
    with col3:
        st.metric("Staff Present", f"{clinic_row['staff_present']} / {clinic_row['staff_total']}")
    with col4:
        st.metric("Avg Daily Footfall", int(clinic_row['recent_avg_footfall']))

    if pd.notna(clinic_row['days_until_stockout']):
        st.warning(f"⏳ Projected stock-out in **{clinic_row['days_until_stockout']} days** at current consumption trend.")

    st.markdown("---")
    st.subheader("📈 Stock Trend (last 10 days)")

    clinic_history = raw_df[raw_df["clinic_id"] == clinic_row["clinic_id"]].sort_values("date").tail(10)
    fig = px.line(
        clinic_history, x="date", y="medicine_stock_units", markers=True,
        title=f"Stock Level Trend — {selected_clinic_name}",
        labels={"medicine_stock_units": "Stock Units", "date": "Date"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 Gemini AI Risk Assessment & Redistribution Plan")

    if clinic_row["risk_level"] == "Healthy":
        st.info("This clinic is currently healthy — no AI intervention recommendation needed.")
    else:
        if st.button("Generate AI Recommendation"):
            with st.spinner("Analyzing trend data and generating recommendation..."):
                try:
                    result = get_recommendation(clinic_row, summary_df)
                    st.success("Analysis complete")
                    st.markdown(f"**Warning:** {result['plain_language_warning']}")
                    st.markdown(f"**Recommended Action:** {result['recommended_action']}")
                    if result.get("donor_clinic_name"):
                        st.markdown(f"**Suggested Donor Clinic:** {result['donor_clinic_name']}")
                    st.caption(f"⚠️ {result['urgency_note']}")
                    st.caption(f"📊 Basis: {result['confidence_basis']}")
                except Exception as e:
                    st.error(f"Error calling Gemini: {e}")