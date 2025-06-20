import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv('data/telco_churn_clean.csv')

churned = df[df['Churn'] == 1]['tenure']
retained = df[df['Churn'] == 0]['tenure']

t_stat, p_value = ttest_ind(churned, retained)
print(f"\n✅ T-test p-value for tenure (churned vs retained): {p_value:.4f}")
