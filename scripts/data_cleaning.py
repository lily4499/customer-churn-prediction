import pandas as pd

# Load data
df = pd.read_csv('data/telco_churn.csv')

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing TotalCharges
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode Gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Encode SeniorCitizen (already numeric)

# Encode Churn as 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Save cleaned data
df.to_csv('data/telco_churn_clean.csv', index=False)
