import streamlit as st
import pandas as pd

import os

st.set_page_config(page_title="Delhi Aadhaar Anomaly Dashboard", layout="wide")

st.title("Delhi Aadhaar Anomaly Detection Dashboard")
st.write("Prototype for UIDAI Data Hackathon (Delhi Scope)")

# Load data
df = pd.read_csv("hybrid_output.csv")


st.info("Dataset contains timestamp, district, auth type, device ID, anomaly flags and risk scores.")






st.subheader("🚨 High Risk Anomalies (Hybrid Model)")
high_df = df[df["final_label"] == "HIGH"]
st.dataframe(high_df)


col1, col2, col3 = st.columns(3)

col1.markdown(
    f"""
    <div style="background-color:#1f2933;padding:20px;border-radius:10px;text-align:center">
    <h2>{len(df)}</h2>
    <p>Total Records</p>
    </div>
    """,
    unsafe_allow_html=True
)

col2.markdown(
    f"""
    <div style="background-color:#3b2f2f;padding:20px;border-radius:10px;text-align:center">
    <h2>{df[df["anomaly_flag"]==1].shape[0]}</h2>
    <p>Total Anomalies</p>
    </div>
    """,
    unsafe_allow_html=True
)

col3.markdown(
    f"""
    <div style="background-color:#2f3b2f;padding:20px;border-radius:10px;text-align:center">
    <h2>{df[df["risk_score"]>=70].shape[0]}</h2>
    <p>High Risk Cases</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "⚠️ Alert Logic: Any authentication pattern with risk score ≥ 70 "
    "is flagged for immediate review by monitoring teams."
)


st.divider()

# Anomaly category count
st.subheader("Anomaly Categories")
category_count = df["anomaly_category"].value_counts()
st.bar_chart(category_count)

st.divider()

# High risk table
st.subheader("High Risk Anomaly Details")
high_risk_df = df[df["risk_score"] >= 70]
st.dataframe(high_risk_df)

st.divider()

st.caption("Note: All data is anonymized. No Aadhaar numbers are used.")





