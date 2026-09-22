import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Configure the page appearance
st.set_page_config(page_title="Risk Agent Dashboard", page_icon="🛡️", layout="wide")

# Custom styling for risk badges
st.markdown("""
<style>
    .High { background-color: #ff4b4b; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .Medium { background-color: #ffa500; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .Low { background-color: #28a745; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Cache the model loading so it doesn't reload on every click
@st.cache_resource
def load_models():
    xgb = joblib.load("xgb_model.pkl")
    iso = joblib.load("iso_model.pkl")
    features = joblib.load("feature_cols.pkl")
    explainer = shap.TreeExplainer(xgb)
    return xgb, iso, features, explainer

xgb_model, iso_model, feature_cols, explainer = load_models()

def generate_explanation(row_features, shap_values, feature_names):
    """Converts complex SHAP math into plain English for the analyst."""
    # Find the top 2 features that pushed the fraud score the highest
    top_indices = np.argsort(shap_values)[-2:][::-1]
    reasons = []
    
    for idx in top_indices:
        feat = feature_names[idx]
        val = row_features[feat]
        
        if feat == "amount_ratio_to_avg" and val > 1.5:
            reasons.append(f"Amount is {val:.1f}x higher than the user's normal average.")
        elif feat == "time_since_prev_tx" and val < 300:
            reasons.append(f"Unusually fast succession ({int(val)}s since last transaction).")
        elif feat == "tx_freq_last_hour" and val >= 3:
            reasons.append(f"High velocity: {int(val)} transactions in the last hour.")
        elif feat == "scaled_amount" and val > 3:
            reasons.append("Raw transaction amount is unusually large.")
        elif feat.startswith("V"):
            reasons.append(f"Latent pattern anomaly detected in vector {feat}.")
            
    if not reasons:
        return "Transaction parameters are slightly irregular but lack a specific dominant trigger."
    return " | ".join(reasons)

st.title("🛡️ AI-Powered Fraud Detection & Risk Agent")
st.write("Upload a batch of transactions to automatically flag high-risk activity using our Dual-Engine AI (XGBoost + Isolation Forest).")

# File Uploader
uploaded_file = st.file_uploader("Upload Transaction CSV for Analysis", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Agent is analyzing transactions..."):
        df = pd.read_csv(uploaded_file)
        
        # Ensure we only pass the trained features into the model
        X_eval = df[feature_cols]
        
        # 1. Supervised Predictions
        xgb_probs = xgb_model.predict_proba(X_eval)[:, 1]
        
        # 2. Unsupervised Predictions
        iso_scores = iso_model.decision_function(X_eval)
        iso_probs = 1.0 / (1.0 + np.exp(iso_scores * 10))
        
        # 3. Hybrid Risk Score Fusion
        hybrid_scores = (0.7 * xgb_probs) + (0.3 * iso_probs)
        df["Risk_Score"] = np.clip(hybrid_scores, 0.0, 1.0)
        
        # 4. Assign Risk Tiers
        conditions = [df["Risk_Score"] > 0.7, df["Risk_Score"] > 0.3]
        choices = ["High", "Medium"]
        df["Risk_Level"] = np.select(conditions, choices, default="Low")
        
        # 5. Extract Explainability for High-Risk Items
        df["Agent_Explanation"] = "N/A"
        high_risk_idx = df[df["Risk_Level"] == "High"].index
        
        if len(high_risk_idx) > 0:
            # Only calculate SHAP for the dangerous transactions to save time
            high_risk_data = X_eval.loc[high_risk_idx]
            shap_vals = explainer.shap_values(high_risk_data)
            
            explanations = []
            for i in range(len(high_risk_idx)):
                exp = generate_explanation(high_risk_data.iloc[i], shap_vals[i], feature_cols)
                explanations.append(exp)
            
            df.loc[high_risk_idx, "Agent_Explanation"] = explanations

        # Display Metrics
        st.markdown("### 📊 Batch Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions Analyzed", len(df))
        col2.metric("High Risk Flagged", len(high_risk_idx), delta_color="inverse")
        col3.metric("Action Required", "YES" if len(high_risk_idx) > 0 else "NO")

        # Display Suspicious Queue
        st.markdown("### 🚨 Suspicious Transaction Queue")
        suspicious_df = df[df["Risk_Level"].isin(["High", "Medium"])].sort_values(by="Risk_Score", ascending=False)
        
        if not suspicious_df.empty:
            display_cols = ["user_id", "Amount", "Risk_Score", "Risk_Level", "Agent_Explanation"]
            # Format the output table nicely
            st.dataframe(
                suspicious_df[display_cols].style.format({
                    "Amount": "${:.2f}",
                    "Risk_Score": "{:.3f}"
                }).map(lambda x: f"background-color: #ff4b4b; color: white" if x == "High" else 
                                 (f"background-color: #ffa500; color: white" if x == "Medium" else ""), 
                       subset=["Risk_Level"]),
                use_container_width=True
            )
        else:
            st.success("No high-risk transactions detected in this batch.")