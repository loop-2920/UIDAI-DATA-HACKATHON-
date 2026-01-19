import pandas as pd
from sklearn.ensemble import IsolationForest

# Load processed data
df = pd.read_csv("anomaly_output_delhi.csv")

# ---------- ML FEATURES ----------
# Simple + safe features
df["failure_rate"] = 1 - df["success_fail"]
df["log_request_count"] = df["request_count"].apply(lambda x: 0 if x == 0 else x)

features = df[[
    "request_count",
    "failure_rate",
    "risk_score"
]]

# ---------- ISOLATION FOREST ----------
model = IsolationForest(
    n_estimators=100,
    contamination=0.25,
    random_state=42
)

model.fit(features)



# ML predictions
df["ml_flag"] = model.predict(features)
df["ml_flag"] = df["ml_flag"].map({-1: 1, 1: 0})

# ML anomaly score (0–100)
df["ml_score"] = (-model.score_samples(features))
df["ml_score"] = (df["ml_score"] - df["ml_score"].min()) / (
    df["ml_score"].max() - df["ml_score"].min()
)
df["ml_score"] = (df["ml_score"] * 100).round(2)

# Save ML output
df.to_csv("ml_output.csv", index=False)

print("STEP 1 DONE: ML anomalies saved to ml_output.csv")

# ---------- HYBRID LOGIC ----------

def hybrid_label(row):
    if row["anomaly_flag"] == 1 and row["ml_flag"] == 1:
        return "HIGH"
    elif row["anomaly_flag"] == 1 or row["ml_flag"] == 1:
        return "MEDIUM"
    else:
        return "NORMAL"

df["final_label"] = df.apply(hybrid_label, axis=1)

# Final hybrid risk score
df["final_risk"] = (
    0.6 * df["risk_score"] + 0.4 * df["ml_score"]
).round(2)

# Save final hybrid output
df.to_csv("hybrid_output.csv", index=False)

print("STEP 2 DONE: Hybrid output saved to hybrid_output.csv")

