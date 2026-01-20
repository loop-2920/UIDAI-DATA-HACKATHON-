import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Delhi Aadhaar Anomaly Dashboard",
    layout="wide"
)

st.title("Delhi Aadhaar Anomaly Detection Dashboard")
st.write("Live Anomaly Detection using Isolation Forest")

# Load raw dataset
uploaded_file = st.file_uploader("Upload UIDAI CSV File", type=["csv"])

if uploaded_file is None:
    st.warning(
        "⚠️ This dashboard is currently optimized for Delhi-specific Aadhaar data. "
        "Performance and insights for other regions may vary, as further iterations "
        "are still in progress."
    )
    st.stop()


df = pd.read_csv(uploaded_file)

st.subheader("Raw Dataset Preview")
st.dataframe(df.head())

# Select numeric features for model
features = df[[
    'pincode',
    'age_0_5',
    'age_5_17',
    'age_18_greater',
    'demo_age_5_17',
    'demo_age_17_',
    'bio_age_5_17',
    'bio_age_17_'
]].fillna(0)

# Sidebar options
st.sidebar.header("Model Settings")

contamination = st.sidebar.slider(
    "Expected Anomaly Percentage",
    min_value=0.01,
    max_value=0.2,
    value=0.05
)

# Train Isolation Forest
model = IsolationForest(
    contamination=contamination,
    random_state=42
)

model.fit(features)

# Generate predictions
df["anomaly_flag"] = model.predict(features)
df["risk_score"] = model.decision_function(features)

# Convert -1/1 to readable labels
df["anomaly_flag"] = df["anomaly_flag"].apply(lambda x: 1 if x == -1 else 0)

# Create risk category
df["final_label"] = df["risk_score"].apply(
    lambda x: "HIGH" if x < -0.05 else "NORMAL"
)

st.divider()

# Metrics Section
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
    <h2>{df[df["final_label"]=="HIGH"].shape[0]}</h2>
    <p>High Risk Cases</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Show anomalies
st.subheader("🚨 Detected Anomalies")
anomalies = df[df["anomaly_flag"] == 1]
st.dataframe(anomalies)

st.divider()

# Bar chart of districts with anomalies
st.subheader("Anomalies by District")

if "district" in df.columns:
    district_count = anomalies["district"].value_counts()
    st.bar_chart(district_count)

st.divider()

# High Risk Details
st.subheader("High Risk Anomaly Details")
high_risk = df[df["final_label"] == "HIGH"]
st.dataframe(high_risk)

st.caption("Note: All data is anonymized. No Aadhaar numbers are used.")
