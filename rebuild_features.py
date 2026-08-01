import sys
sys.path.append("/Users/shobhitagnihotri/Downloads/DL GEN AI/src")
import pandas as pd
import numpy as np
from joblib import load, dump
from train import build_features, build_feature_row

BASE = "/Users/shobhitagnihotri/Downloads/DL GEN AI"
train_df = pd.read_csv(f"{BASE}/train (1).csv")
train_df.columns = [c.strip() for c in train_df.columns]
test_df = pd.read_csv(f"{BASE}/test (1).csv")
test_df.columns = [c.strip() for c in test_df.columns]

# Fit and transform
X_train_new, tfidf_vec = build_features(train_df, fit=True)
X_test_new, _ = build_features(test_df, tfidf_vectorizer=tfidf_vec, fit=False)

print("Rebuilt shapes:")
print("  X_train_new:", X_train_new.shape)
print("  X_test_new:", X_test_new.shape)

# Load LGBM full model
lgbm_model = load(f"{BASE}/models/lgbm_full_model.joblib")
print("LGBM input features:", getattr(lgbm_model, "n_features_in_", None))

# Try predicting to see if it works now
try:
    probs = lgbm_model.predict_proba(X_test_new)
    print("✓ Successfully predicted with LGBM!")
except Exception as e:
    print("✗ Prediction failed:", e)
