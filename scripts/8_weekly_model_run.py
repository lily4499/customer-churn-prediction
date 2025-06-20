# 8_weekly_model_run.py
# Re-trains the model weekly and updates the churn risk list

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import datetime
import os

# Ensure output folder exists
os.makedirs('output', exist_ok=True)

# Load cleaned data
df = pd.read_csv('data/telco_churn_clean.csv')

# Features and target
features = ['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]
y = df['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Generate RiskScore for all customers
df['RiskScore'] = model.predict_proba(X)[:, 1]

# Add date for tracking
today = datetime.date.today().isoformat()
df['RunDate'] = today

# Save latest risk list
df[['customerID', 'RiskScore', 'RunDate']].to_csv('output/weekly_churn_risk.csv', index=False)

print(f"\n✅ Weekly churn risk list updated: output/weekly_churn_risk.csv ({today})")

