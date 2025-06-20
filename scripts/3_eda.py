import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/telco_churn_clean.csv')

# Plot 1 — Tenure
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack')
plt.title('Tenure Distribution by Churn')
plt.savefig('output/tenure_churn_distribution.png')
plt.show()

# Plot 2 — Monthly Charges
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges vs Churn')
plt.savefig('output/monthly_charges_churn.png')
plt.show()

