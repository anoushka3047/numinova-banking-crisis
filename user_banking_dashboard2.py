import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from io import BytesIO

# ================================
# 🏦 Banking Crisis Prevention Dashboard v2
# ================================
st.set_page_config(page_title="Banking Crisis Prevention System", layout="wide")

st.title("🏦 Banking Crisis Prevention System")
st.write("""
AI/ML-based system for detecting abnormal fund movements, insolvency risk, fraud, and operational mismanagement.  
Upload your own dataset or input custom data to get real-time AI risk alerts.
""")

# ================================
# 📥 File Upload
# ================================
st.sidebar.header("📂 Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
else:
    st.info("Using default dataset for demo (bank_financial_risk_dataset_14000_10.csv).")
    df = pd.read_csv("bank_financial_risk_dataset_14000_10.csv")

st.sidebar.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")

# Sidebar menu
menu = st.sidebar.radio(
    "Select Module",
    [
        "📊 Abnormal Fund Detection",
        "⚠️ Insolvency & Fraud Risk Prediction",
        "🧠 Operational Mismanagement Check",
        "🧩 User Input / Custom Analysis",
        "📑 Download Report"
    ]
)

# Global variable to store results
results = df.copy()

# ================================
# 📊 Abnormal Fund Detection
# ================================
if menu == "📊 Abnormal Fund Detection":
    st.header("📊 Abnormal Fund Detection")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 3:
        st.warning("Need at least 3 numeric columns for anomaly detection.")
    else:
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['abnormal_fund_flag'] = iso.fit_predict(df[num_cols[:3]])
        df['abnormal_fund_flag'] = df['abnormal_fund_flag'].apply(lambda x: 1 if x == -1 else 0)

        st.metric("Banks Flagged as Abnormal", int(df['abnormal_fund_flag'].sum()))
        st.dataframe(df[df['abnormal_fund_flag'] == 1].head(10))

        results['abnormal_fund_flag'] = df['abnormal_fund_flag']

# ================================
# ⚠️ Insolvency & Fraud Risk Prediction
# ================================
elif menu == "⚠️ Insolvency & Fraud Risk Prediction":
    st.header("⚠️ Insolvency & Fraud Risk Prediction")

    target_candidates = [c for c in df.columns if 'risk' in c.lower() or 'flag' in c.lower()]
    if not target_candidates:
        st.warning("⚠️ No target risk/flag column found. Please include one like 'bank_failure_risk_flag'.")
    else:
        target = target_candidates[0]
        st.write(f"Detected target column: `{target}`")

        X = df.drop(columns=[target])
        y = df[target]

        X = X.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model.fit(X_scaled, y)

        df['fraud_risk_score'] = model.predict_proba(X_scaled)[:, 1]
        df['fraud_risk_flag'] = (df['fraud_risk_score'] > 0.5).astype(int)

        st.metric("Average Predicted Risk (%)", f"{df['fraud_risk_score'].mean()*100:.2f}")
        st.dataframe(df[['fraud_risk_score', 'fraud_risk_flag']].head(10))

        results['fraud_risk_score'] = df['fraud_risk_score']
        results['fraud_risk_flag'] = df['fraud_risk_flag']

# ================================
# 🧠 Operational Mismanagement Check
# ================================
elif menu == "🧠 Operational Mismanagement Check":
    st.header("🧠 Operational Mismanagement Analysis")

    required = ['liquidity_ratio', 'loan_default_ratio', 'capital_adequacy_ratio']
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.warning(f"Missing required columns: {missing}")
    else:
        df['mismanagement_score'] = (
            (1 - df['liquidity_ratio']) * 0.4 +
            df['loan_default_ratio'] * 0.3 +
            (10 - df['capital_adequacy_ratio']) / 10 * 0.3
        )
        df['mismanagement_flag'] = (df['mismanagement_score'] > 0.5).astype(int)

        st.metric("Banks Flagged for Mismanagement", int(df['mismanagement_flag'].sum()))
        st.dataframe(df[['mismanagement_score', 'mismanagement_flag']].head(10))

        results['mismanagement_score'] = df['mismanagement_score']
        results['mismanagement_flag'] = df['mismanagement_flag']

# ================================
# 🧩 User Input / Custom Analysis
# ================================
elif menu == "🧩 User Input / Custom Analysis":
    st.header("🧩 Custom Data Entry for Risk Evaluation")

    loan_default_ratio = st.number_input("Loan Default Ratio", 0.0, 1.0, 0.25)
    liquidity_ratio = st.number_input("Liquidity Ratio", 0.0, 1.0, 0.8)
    withdrawal_rate = st.number_input("Withdrawal Rate", 0.0, 1.0, 0.3)
    investment_risk_score = st.number_input("Investment Risk Score (0–10)", 0.0, 10.0, 4.0)
    fraud_alerts_count = st.number_input("Fraud Alerts Count", 0, 10, 1)
    capital_adequacy_ratio = st.number_input("Capital Adequacy Ratio", 0.0, 1.0, 0.2)
    customer_sentiment_score = st.number_input("Customer Sentiment (0–10)", 0.0, 10.0, 7.0)

    if st.button("🚨 Run AI Risk Analysis"):
        sample = pd.DataFrame({
            "loan_default_ratio": [loan_default_ratio],
            "liquidity_ratio": [liquidity_ratio],
            "withdrawal_rate": [withdrawal_rate],
            "investment_risk_score": [investment_risk_score],
            "fraud_alerts_count": [fraud_alerts_count],
            "capital_adequacy_ratio": [capital_adequacy_ratio],
            "customer_sentiment_score": [customer_sentiment_score]
        })

        X = df.select_dtypes(include=[np.number])
        if 'bank_failure_risk_flag' in df.columns:
            y = df['bank_failure_risk_flag']
        else:
            y = (X.iloc[:, -1] > X.iloc[:, -1].median()).astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        model.fit(X_scaled, y)

        pred_prob = model.predict_proba(scaler.transform(sample))[0][1]

        if pred_prob > 0.8:
            st.error(f"🚨 High Risk Detected! Probability: {pred_prob:.2f}")
        elif pred_prob > 0.5:
            st.warning(f"⚠️ Moderate Risk Detected! Probability: {pred_prob:.2f}")
        else:
            st.success(f"✅ Low Risk. Probability: {pred_prob:.2f}")

        st.dataframe(sample)

# ================================
# 📑 Download Report
# ================================
elif menu == "📑 Download Report":
    st.header("📑 Download Analysis Report")

    st.write("""
    After performing any analysis, you can download the complete report below.
    This report includes flags, risk scores, and other AI-based evaluations.
    """)

    csv_data = results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Report",
        data=csv_data,
        file_name="banking_analysis_report.csv",
        mime="text/csv"
    )

    st.success("✅ Report ready for download. Use it in your ML report or presentation.")