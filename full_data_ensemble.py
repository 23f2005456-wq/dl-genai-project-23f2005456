import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from joblib import load
import optuna

BASE = Path("/Users/shobhitagnihotri/Downloads/DL GEN AI")
OUT  = BASE / "outputs"
MODEL_DIR = BASE / "models"

OPTION_COLS = ["A", "B", "C", "D", "E"]
L2I = {c: i for i, c in enumerate(OPTION_COLS)}
I2L = {i: c for c, i in L2I.items()}

# Load actual answers
train_df = pd.read_csv(BASE / "train (1).csv")
train_df.columns = [c.strip() for c in train_df.columns]
test_df = pd.read_csv(BASE / "test (1).csv")
test_df.columns = [c.strip() for c in test_df.columns]
LA = train_df["answer"].map(L2I).values

def norm(a):
    mn = a.min(1, keepdims=True)
    mx = a.max(1, keepdims=True)
    return (a - mn) / (mx - mn + 1e-9)

def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score

def mapk(actuals, predictions, k=3):
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)]))

def top3(sc):
    return [np.argsort(-row)[:3].tolist() for row in sc]

# Load features for full model predictions
X_test = np.load(OUT / "X_test.npy")

def get_full_probs(model_name):
    model = load(MODEL_DIR / f"{model_name}_full_model.joblib")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)
        probs = (probs - probs.min()) / (probs.ptp() + 1e-9)
    return probs.reshape(-1, 5)

# Load full model predictions (trained on 100% of train data!)
test_full_probs = {
    "lgbm": norm(get_full_probs("lgbm")),
    "xgb": norm(get_full_probs("xgb")),
    "cat": norm(get_full_probs("cat")),
    "lr": norm(get_full_probs("lr")),
    "svm": norm(get_full_probs("svm")),
}

# Load fold OOF predictions (which correspond to the CV weights)
models_tr = {
    "lgbm": norm(np.load(OUT / "oof_lgbm.npy")),
    "xgb": norm(np.load(OUT / "oof_xgb.npy")),
    "cat": norm(np.load(OUT / "oof_cat.npy")),
    "lr": norm(np.load(OUT / "oof_lr.npy")),
    "svm": norm(np.load(OUT / "oof_svm.npy")),
    "st_mpnet": norm(np.load(OUT / "st_train_mpnet.npy")),
    "st_minilm": norm(np.load(OUT / "st_train_minilm.npy")),
}

models_te = {
    "lgbm": test_full_probs["lgbm"],
    "xgb": test_full_probs["xgb"],
    "cat": test_full_probs["cat"],
    "lr": test_full_probs["lr"],
    "svm": test_full_probs["svm"],
    "st_mpnet": norm(np.load(OUT / "st_test_mpnet.npy")),
    "st_minilm": norm(np.load(OUT / "st_test_minilm.npy")),
}

MN = list(models_tr.keys())

# Split into 5 folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds_indices = list(skf.split(np.zeros(len(train_df)), LA))

def evaluate_weights_cv(weights):
    wa = np.array(weights)
    wa /= (wa.sum() + 1e-9)
    fold_scores = []
    for tr_idx, vl_idx in folds_indices:
        ens_vl = sum(wa[i] * models_tr[MN[i]][vl_idx] for i in range(len(MN)))
        score = mapk(LA[vl_idx].tolist(), top3(ens_vl))
        fold_scores.append(score)
    return np.mean(fold_scores)

# Deeper optimization (10,000 trials for perfect weights)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def obj(trial):
    w = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in MN]
    return evaluate_weights_cv(w)

print("🚀 Running deep optimization on full data models (10,000 trials)...")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
)
study.optimize(obj, n_trials=10000)

BW = np.array([study.best_trial.params[f"w_{n}"] for n in MN])
BW /= (BW.sum() + 1e-9)

print(f"\n🏆 Deep CV Ensemble MAP@3 = {study.best_trial.value:.5f}")
for n, w in sorted(zip(MN, BW), key=lambda x: -x[1]):
    print(f"  {n:<20} {w:.5f}")

# Temperature search on CV
best_T = 1.0
best_T_score = 0.0
for T in np.linspace(0.01, 1.5, 60):
    def sft(sc, temp):
        e = np.exp(sc / temp)
        return e / (e.sum(1, keepdims=True) + 1e-9)
        
    scores_T = []
    for tr_idx, vl_idx in folds_indices:
        ens_vl = sum(BW[i] * models_tr[MN[i]][vl_idx] for i in range(len(MN)))
        ens_vl_sc = sft(ens_vl, T)
        score = mapk(LA[vl_idx].tolist(), top3(ens_vl_sc))
        scores_T.append(score)
    mean_score = np.mean(scores_T)
    if mean_score > best_T_score:
        best_T_score = mean_score
        best_T = T

print(f"\n🌡️ Best Softmax Temperature: {best_T:.3f} (CV MAP@3: {best_T_score:.5f})")

def sft(sc, temp):
    e = np.exp(sc / temp)
    return e / (e.sum(1, keepdims=True) + 1e-9)

FTE_SC = sft(sum(BW[i] * models_te[MN[i]] for i in range(len(MN))), best_T)

# Save submission
preds = [" ".join([I2L[i] for i in np.argsort(-r)[:3]]) for r in FTE_SC]
sub = pd.DataFrame({"ID": test_df["id"].values, "Prediction": preds})
sub.to_csv(BASE / "submission.csv", index=False)
sub.to_csv(OUT / "submission_full_data_ensemble.csv", index=False)
sub.to_csv("/Users/shobhitagnihotri/Desktop/dl_genai2/submission.csv", index=False)
print("🎉 Optimized 100% data ensemble saved to desktop successfully!")
