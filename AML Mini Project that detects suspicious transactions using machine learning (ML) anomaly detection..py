#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Generating a synthetic dataset
np.random.seed(42)
n_samples = 1000

# Normal transactions (Legitimate)
normal_txns = pd.DataFrame({
    'transaction_id': range(1, n_samples+1),
    'amount': np.random.normal(loc=5000, scale=2000, size=n_samples),  # Normal amounts
    'transaction_type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer'], size=n_samples, p=[0.5, 0.3, 0.2]),
    'account_age': np.random.randint(1, 20, size=n_samples),  # Account age in years
    'num_transactions': np.random.randint(5, 50, size=n_samples)  # Transactions per month
})

# Creating suspicious transactions (Anomalies)
n_anomalies = 20
anomalous_txns = pd.DataFrame({
    'transaction_id': range(n_samples+1, n_samples+1+n_anomalies),
    'amount': np.random.uniform(50000, 200000, n_anomalies),  # High-value transactions
    'transaction_type': np.random.choice(['Deposit', 'Withdrawal', 'Transfer'], size=n_anomalies),
    'account_age': np.random.randint(0, 2, size=n_anomalies),  # New accounts
    'num_transactions': np.random.randint(50, 100, size=n_anomalies)  # High frequency
})

# Combining both datasets
aml_data = pd.concat([normal_txns, anomalous_txns]).sample(frac=1).reset_index(drop=True)

# Adding labels (0 = Normal, 1 = Suspicious)
aml_data['label'] = [0] * n_samples + [1] * n_anomalies

# Displaying sample data
aml_data.head()


# In[2]:


from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Selecting features for model training
features = ['amount', 'account_age', 'num_transactions']
X = aml_data[features]

# Training Isolation Forest model
model = IsolationForest(contamination=0.02, random_state=42)  # Assuming ~2% anomalies
aml_data['predicted_label'] = model.fit_predict(X)

# Mapping -1 (Anomalies) to 1 (Suspicious) and 1 (Normal) to 0
aml_data['predicted_label'] = aml_data['predicted_label'].map({1: 0, -1: 1})

# Evaluating model performance
from sklearn.metrics import classification_report

print("Classification Report:\n")
print(classification_report(aml_data['label'], aml_data['predicted_label']))

# Visualizing transactions (amount vs. num_transactions)
plt.figure(figsize=(10, 6))
plt.scatter(aml_data['amount'], aml_data['num_transactions'], c=aml_data['predicted_label'], cmap='coolwarm', alpha=0.6)
plt.xlabel("Transaction Amount")
plt.ylabel("Number of Transactions")
plt.title("AML Detection: Normal (Blue) vs Suspicious (Red) Transactions")
plt.colorbar(label="Predicted Label")
plt.show()


# In[ ]:




