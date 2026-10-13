import pandas as pd
import joblib
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

print("Loading model.pkl...")
model = joblib.load("model.pkl")

print("Loading test data...")
df = pd.read_parquet("data/data copy/interim/knee-provider.parquet")
df['poor_outcome'] = (df['oks_t1_score'] < 27).astype(int)

features = [
    'oks_t0_score', 'oks_t0_pain', 'oks_t0_night_pain', 'oks_t0_washing',
    'oks_t0_transport', 'oks_t0_walking', 'oks_t0_standing', 'oks_t0_limping',
    'oks_t0_kneeling', 'oks_t0_work', 'oks_t0_confidence', 'oks_t0_shopping',
    'oks_t0_stairs',
    't0_mobility', 't0_self_care', 't0_activity', 't0_discomfort', 't0_anxiety',
    't0_eq_vas',
    'age_band', 'gender',
    'heart_disease', 'high_bp', 'diabetes', 'stroke', 'lung_disease',
    't0_disability', 't0_previous_surgery'
]

age_map = {'40 to 49': 1, '50 to 59': 2, '60 to 69': 3,
           '70 to 79': 4, '80 to 89': 5, '90 to 120': 6}
df['age_band'] = df['age_band'].map(age_map)

for col in ['heart_disease', 'high_bp', 'diabetes', 'stroke', 'lung_disease']:
    df[col] = df[col].replace({1: 1, 9: 0, 0: 0}).fillna(0)

for col in ['t0_disability', 't0_previous_surgery']:
    df[col] = df[col].replace({9: 0})

oks_cols = ['oks_t0_pain', 'oks_t0_night_pain', 'oks_t0_washing', 'oks_t0_transport',
            'oks_t0_walking', 'oks_t0_standing', 'oks_t0_limping', 'oks_t0_kneeling',
            'oks_t0_work', 'oks_t0_confidence', 'oks_t0_shopping', 'oks_t0_stairs']
for col in oks_cols:
    df[col] = df[col].replace({9: None})

for col in ['t0_mobility', 't0_self_care', 't0_activity', 't0_discomfort', 't0_anxiety']:
    df[col] = df[col].replace({9: None})

df['t0_eq_vas'] = df['t0_eq_vas'].replace({999: None})

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

X = df[features]
y = df['poor_outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

probs = model.predict_proba(X_test)[:, 1]

print("=" * 60)
print(f"Total test samples: {len(y_test)}")
print(f"Actual Poor Outcome count: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")
print(f"Actual Good Outcome count: {len(y_test) - sum(y_test)} ({(1 - sum(y_test)/len(y_test))*100:.2f}%)")
print("=" * 60)
print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
print(f"Average Precision (PR-AUC): {average_precision_score(y_test, probs):.4f}")
print("=" * 60)

print("\nPrecision at Different Risk Thresholds (Poor Outcome):")
for thresh in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    preds_t = (probs >= thresh).astype(int)
    p = precision_score(y_test, preds_t, pos_label=1, zero_division=0)
    r = recall_score(y_test, preds_t, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, preds_t, pos_label=1, zero_division=0)
    print(f"Threshold >= {thresh:.2f} -> Precision: {p*100:.2f}% | Recall: {r*100:.2f}% | F1: {f1:.4f} | Patients Flagged: {sum(preds_t)}")
