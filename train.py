import pandas as pd
import numpy as np
import joblib
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from model_architecture import TKARiskModel

# Step 1: Load data
print("Step 1: Loading knee replacement dataset...")
df = pd.read_parquet("data/data copy/interim/knee-provider.parquet")

# Step 2: Create target column (1 = poor outcome, 0 = good outcome)
# Oxford Knee Score < 27 after surgery = poor outcome (NHS / medical standard)
df['poor_outcome'] = (df['oks_t1_score'] < 27).astype(int)

# Step 3: Select pre-operation features only (what a doctor knows BEFORE surgery)
features = [
    # Oxford Knee Score individual questions (pre-op)
    'oks_t0_score', 'oks_t0_pain', 'oks_t0_night_pain', 'oks_t0_washing',
    'oks_t0_transport', 'oks_t0_walking', 'oks_t0_standing', 'oks_t0_limping',
    'oks_t0_kneeling', 'oks_t0_work', 'oks_t0_confidence', 'oks_t0_shopping',
    'oks_t0_stairs',
    # Quality of life scores (pre-op EuroQoL 5D + VAS)
    't0_mobility', 't0_self_care', 't0_activity', 't0_discomfort', 't0_anxiety',
    't0_eq_vas',
    # Patient profile
    'age_band', 'gender',
    # Comorbidities
    'heart_disease', 'high_bp', 'diabetes', 'stroke', 'lung_disease',
    # Pre-op status
    't0_disability', 't0_previous_surgery'
]

df = df[features + ['poor_outcome']].copy()

# Step 4: Encode and clean features
# age_band: text -> number (1 to 6)
age_map = {'40 to 49': 1, '50 to 59': 2, '60 to 69': 3,
           '70 to 79': 4, '80 to 89': 5, '90 to 120': 6}
df['age_band'] = df['age_band'].map(age_map)

# comorbidities: 1=Yes, 9=No/missing -> convert to 1=Yes, 0=No
for col in ['heart_disease', 'high_bp', 'diabetes', 'stroke', 'lung_disease']:
    df[col] = df[col].replace({1: 1, 9: 0, 0: 0}).fillna(0)

# pre-op disability and previous surgery: 9=No -> 0
for col in ['t0_disability', 't0_previous_surgery']:
    df[col] = df[col].replace({9: 0})

# OKS individual questions: 9=missing -> replace with NaN (will be dropped)
oks_cols = ['oks_t0_pain', 'oks_t0_night_pain', 'oks_t0_washing', 'oks_t0_transport',
            'oks_t0_walking', 'oks_t0_standing', 'oks_t0_limping', 'oks_t0_kneeling',
            'oks_t0_work', 'oks_t0_confidence', 'oks_t0_shopping', 'oks_t0_stairs']
for col in oks_cols:
    df[col] = df[col].replace({9: None})

# eq5d quality of life: 9=missing -> replace with NaN
for col in ['t0_mobility', 't0_self_care', 't0_activity', 't0_discomfort', 't0_anxiety']:
    df[col] = df[col].replace({9: None})

# t0_eq_vas: 999=missing -> replace with NaN
df['t0_eq_vas'] = df['t0_eq_vas'].replace({999: None})

# Step 5: Convert all columns to numeric, then drop rows with missing values
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

print(f"Dataset size after cleaning: {df.shape}")
print(f"Poor outcome %: {df['poor_outcome'].mean() * 100:.2f}%")

# Step 6: Split into X (features) and y (target) with stratification
X = df[features]
y = df['poor_outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 7: Train High-Precision Ensemble Model (LightGBM + XGBoost + Clinical Engineering)
print("\nStep 7: Training High-Precision Ensemble Model...")
model = TKARiskModel(lgb_weight=0.5, default_threshold=0.45)
model.fit(X_train, y_train)

# Step 8: Evaluate Model Performance
print("\nStep 8: Evaluating Model Performance...")
probs = model.predict_proba(X_test)[:, 1]
preds_default = model.predict(X_test, threshold=0.45)

auc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)

print("=" * 65)
print(f"Overall ROC-AUC Score:          {auc:.4f}")
print(f"Average Precision (PR-AUC):     {pr_auc:.4f}")
print("=" * 65)

print("\nDetailed Classification Report at Operating Threshold (0.45):")
print(classification_report(y_test, preds_default, target_names=["Good Outcome (0)", "Poor Outcome (1)"], digits=4))

cm = confusion_matrix(y_test, preds_default)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix (Threshold 0.45):")
print(f"  True Negatives (TN):  {tn}")
print(f"  False Positives (FP): {fp}")
print(f"  False Negatives (FN): {fn}")
print(f"  True Positives (TP):  {tp}")

print("\n" + "=" * 70)
print(f"{'Decision Threshold':<20} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<10}")
print("=" * 70)
for t in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    p_pred = (probs >= t).astype(int)
    p = precision_score(y_test, p_pred, zero_division=0)
    r = recall_score(y_test, p_pred, zero_division=0)
    f1 = f1_score(y_test, p_pred, zero_division=0)
    print(f"Risk Score >= {t:<7.2f} | {p*100:<11.2f}% | {r*100:<11.2f}% | {f1:<10.4f}")
print("=" * 70)

# Step 9: Save model
joblib.dump(model, 'model.pkl')
print("\nStep 9: Model saved to model.pkl")

# Copy to tka-risk-predictor folder
shutil.copy('model.pkl', 'tka-risk-predictor/model.pkl')
print("Model copied to tka-risk-predictor/model.pkl")
