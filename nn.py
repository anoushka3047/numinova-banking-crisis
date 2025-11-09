import pandas as pd
from sklearn.ensemble import IsolationForest

# Load the dataset
df = pd.read_csv('bank_financial_risk_dataset_14000_10.csv')

# Specify features for anomaly detection
anomaly_features = ['withdrawal_rate', 'liquidity_ratio', 'fraud_alerts_count']

# Initialize Isolation Forest for anomaly detection
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_pred = iso_forest.fit_predict(df[anomaly_features])

# Flag anomalies
df['abnormal_fund_flag'] = (anomaly_pred == -1).astype(int)

# Output summary statistics
print(f"Number of banks flagged as abnormal: {df['abnormal_fund_flag'].sum()}")

# Print details for first 5 flagged banks
flagged_indices = df[df['abnormal_fund_flag'] == 1].index[:5]
print("Details for the first 5 flagged banks:")
print(df.loc[flagged_indices, [
    'loan_default_ratio', 'liquidity_ratio', 'withdrawal_rate',
    'investment_risk_score', 'fraud_alerts_count', 'capital_adequacy_ratio',
    'customer_sentiment_score', 'bank_failure_risk_flag'
]])






