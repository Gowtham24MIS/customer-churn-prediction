import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import json
from imblearn.over_sampling import SMOTE

# =====================
# Step 1 - Create Dataset
# =====================
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print(df.head())
print(df.shape)
print(df.columns.tolist())

print("✅ Dataset created!")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# =====================
# =====================
# Step 2 - Clean Data
# =====================

# Drop useless columns
df = df.drop(['customerID'], axis=1)

# Target column is 'Churn' - convert Yes/No to 1/0
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# TotalCharges has some empty strings - fix it
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values
df = df.fillna(df.median(numeric_only=True))

# Check imbalance
print("\nTarget distribution:")
print(df['Churn'].value_counts())

# =====================
# Step 3 - Encode Categorical Columns
# =====================

# Find all text columns automatically
cat_columns = df.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns to encode:")
print(cat_columns)

# One Hot Encoding
df = pd.get_dummies(df, columns=cat_columns)

print("\n✅ Encoding done!")
print("Shape after encoding:", df.shape)

# =====================
# Step 4 - Split Data
# =====================
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\n✅ Data split done!")
print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)

# =====================
# Step 5 - Train Model
# =====================

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(pd.Series(y_train_smote).value_counts())

model = XGBClassifier(
    scale_pos_weight=3,
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train_smote, y_train_smote)
print("\n✅ Model trained successfully!")

# =====================
# Step 6 - Evaluate Model
# =====================
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =====================
# Step 7 - Save Model
# =====================
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('columns.json', 'w') as f:
    json.dump(list(X.columns), f)

print("\n✅ Model saved successfully!")
print("🔥 customer_prediction.py complete!")