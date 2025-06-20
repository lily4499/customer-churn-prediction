# customer-churn-prediction

---

## 📂 Project Structure

```bash
customer-churn-prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── scripts/
│   └── data_cleaning.py
│   └── modeling.py
├── output/
│   └── churn_dashboard.pbix  # Exported Power BI file (optional)
├── README.md
├── requirements.txt
└── venv/  # Python virtual environment
```

---

## 🖥️ CLI Steps

### 1️⃣ Create Project Folder

```bash
mkdir customer-churn-prediction
cd customer-churn-prediction
```

### 2️⃣ Set up Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR on Windows:
# venv\Scripts\activate
```

### 3️⃣ Create Requirements File

```bash
cat > requirements.txt <<EOF
pandas
matplotlib
seaborn
scikit-learn
jupyter
EOF
----------------OR------------------------
echo -e "pandas\nmatplotlib\nseaborn\nscikit-learn\njupyter" > requirements.txt
pip install -r requirements.txt
```

### 4️⃣ Add CSV Dataset

Save your provided CSV snippet into:
`data/telco_churn.csv`
```
pip install kaggle
mkdir data && cd data
kaggle datasets download blastchar/telco-customer-churn
unzip telco-customer-churn.zip
mv WA_Fn-UseC_-Telco-Customer-Churn.csv  telco_churn.csv
```
![image](https://github.com/user-attachments/assets/97bebdaa-1ec5-4818-8038-4594516d1caf)


---

## 📝 Data Analysis Workflow (solved step-by-step)

---

### 1️⃣ Define Problem

👉 Predict which customers are likely to churn using historical data.

---

### 2️⃣ Get Data

**Load CSV:**

```python
# notebooks/churn_analysis.ipynb
import pandas as pd

df = pd.read_csv('../data/telco_churn.csv')
df.head()
```

---

### 3️⃣ Clean Data

```python
# scripts/data_cleaning.py

import pandas as pd

# Load data
df = pd.read_csv('data/telco_churn.csv')

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Handle missing TotalCharges
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Encode Gender
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Encode SeniorCitizen (already numeric)

# Encode Churn as 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Save cleaned data
df.to_csv('data/telco_churn_clean.csv', index=False)
```

Run:

```bash
python scripts/data_cleaning.py
```

---

### 4️⃣ Explore

```python
# notebooks/churn_analysis.ipynb

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/telco_churn_clean.csv')

# Distribution of Tenure by Churn
sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack')
plt.title('Tenure Distribution by Churn Status')
plt.show()

# Compare Monthly Charges
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Monthly Charges by Churn Status')
plt.show()
```

---

### 5️⃣ Statistical Test

```python
from scipy.stats import ttest_ind

# T-test for average tenure
churned = df[df['Churn'] == 1]['tenure']
retained = df[df['Churn'] == 0]['tenure']

t_stat, p_value = ttest_ind(churned, retained)
print(f"T-test p-value = {p_value:.4f}")
```

---

### 6️⃣ Build Model (Logistic Regression)

```python
# scripts/modeling.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load cleaned data
df = pd.read_csv('data/telco_churn_clean.csv')

# Select features
features = ['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Run:

```bash
python scripts/modeling.py
```

---

### 7️⃣ Visualization (Power BI)

* Export CSV `telco_churn_clean.csv`
* Import to Power BI Desktop.
* Create:

  * Churn by Segment (Contract type, Gender)
  * Risk Scores by Customer ID
  * Monthly Charges vs Churn
 
![image](https://github.com/user-attachments/assets/c9a4baa2-7621-447e-b63c-c00a7cf3802d)


---

### 8️⃣ Recommendation

**Finding:**

* Customers with *low tenure* and *high monthly charges* → high risk of churn.
  **Action:**
* Target them with loyalty offers (discounts or longer contracts).

---

### 9️⃣ Automation

Create `weekly_report.py` to run modeling weekly and output risk scores:

```python
# scripts/weekly_report.py

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/telco_churn_clean.csv')
features = ['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]
y = df['Churn']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predict risk scores
df['RiskScore'] = model.predict_proba(X)[:, 1]
df[['customerID', 'RiskScore']].to_csv('output/weekly_churn_risk.csv', index=False)
```

Run:

```bash
python scripts/weekly_report.py
```

---

## ✅ Summary

You now have:

* **Cleaned Data**
* **Exploratory Visuals**
* **Statistical Insights**
* **Logistic Regression Model**
* **Power BI Dashboard**
* **Automated Weekly Report**

---

If you want, I can:

1️⃣ Zip this whole project for you
2️⃣ Generate a PDF Stakeholder Report + PowerPoint template

Just tell me! 🚀
