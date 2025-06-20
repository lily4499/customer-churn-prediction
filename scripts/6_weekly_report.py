import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/telco_churn_clean.csv')

features = ['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]
y = df['Churn']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

df['RiskScore'] = model.predict_proba(X)[:, 1]

df[['customerID', 'RiskScore']].to_csv('output/weekly_churn_risk.csv', index=False)
print("\n✅ Weekly churn risk report saved to output/weekly_churn_risk.csv")

