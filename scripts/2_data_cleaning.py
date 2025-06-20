import pandas as pd

df = pd.read_csv('data/telco_churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print("\n✅ Cleaned Data Sample:")
print(df[['customerID', 'gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']].head())

df.to_csv('data/telco_churn_clean.csv', index=False)
print("\n✅ Saved cleaned data to data/telco_churn_clean.csv")

