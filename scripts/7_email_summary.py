# Summary for email:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind

# Load cleaned data first!
df = pd.read_csv('data/telco_churn_clean.csv')

# Total customers
total_customers = len(df)
churn_rate = df['Churn'].mean() * 100

# Model training (you must define X, y, split, model)
features = ['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test) * 100

# Tenure significance
churned = df[df['Churn'] == 1]['tenure']
retained = df[df['Churn'] == 0]['tenure']
_, p_value = ttest_ind(churned, retained)

# Print summary
print("\n📋 Summary for Email Report:")
print(f"✅ Total customers analyzed: {total_customers}")
print(f"✅ Overall churn rate: {churn_rate:.1f}%")
print(f"✅ Model accuracy: {accuracy:.1f}%")

if p_value < 0.001:
    print("✅ Significant factor: Tenure (p-value < 0.001 — lower tenure → higher churn risk)")
else:
    print(f"Tenure p-value = {p_value:.4f} — not significant.")

