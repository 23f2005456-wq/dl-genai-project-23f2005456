import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import optuna

BASE = Path("/Users/shobhitagnihotri/Downloads/DL GEN AI")
OUT  = BASE / "outputs"

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

# Load all predictions
models_tr = {
    "lgbm": norm(np.load(OUT / "oof_lgbm.npy")),
    "xgb": norm(np.load(OUT / "oof_xgb.npy")),
    "cat": norm(np.load(OUT / "oof_cat.npy")),
    "lr": norm(np.load(OUT / "oof_lr.npy")),
    "svm": norm(np.load(OUT / "oof_svm.npy")),
    "st_bge": norm(np.load(OUT / "st_train_bge.npy")),
    "st_mpnet": norm(np.load(OUT / "st_train_mpnet.npy")),
    "st_e5": norm(np.load(OUT / "st_train_e5.npy")),
    "st_minilm": norm(np.load(OUT / "st_train_minilm.npy")),
    "ce_minilm": norm(np.load(OUT / "ce_train_ce_minilm.npy")),
    "ce_bge_r": norm(np.load(OUT / "ce_train_bge_rerank.npy")),
}

models_te = {
    "lgbm": norm(np.load(OUT / "test_lgbm.npy")),
    "xgb": norm(np.load(OUT / "test_xgb.npy")),
    "cat": norm(np.load(OUT / "test_cat.npy")),
    "lr": norm(np.load(OUT / "test_lr.npy")),
    "svm": norm(np.load(OUT / "test_svm.npy")),
    "st_bge": norm(np.load(OUT / "st_test_bge.npy")),
    "st_mpnet": norm(np.load(OUT / "st_test_mpnet.npy")),
    "st_e5": norm(np.load(OUT / "st_test_e5.npy")),
    "st_minilm": norm(np.load(OUT / "st_test_minilm.npy")),
    "ce_minilm": norm(np.load(OUT / "ce_test_ce_minilm.npy")),
    "ce_bge_r": norm(np.load(OUT / "ce_test_bge_rerank.npy")),
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

# Run deeper Optuna search
optuna.logging.set_verbosity(optuna.logging.WARNING)

def obj(trial):
    # Search space
    w = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in MN]
    return evaluate_weights_cv(w)

print("🚀 Starting deep Optuna ensembling search (5000 trials)...")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
)
study.optimize(obj, n_trials=5000)

BW = np.array([study.best_trial.params[f"w_{n}"] for n in MN])
BW /= (BW.sum() + 1e-9)

print(f"\n🏆 Deep Optimized Ensemble MAP@3 = {study.best_trial.value:.5f}")
for n, w in sorted(zip(MN, BW), key=lambda x: -x[1]):
    print(f"  {n:<20} {w:.5f}")

# Fine grid search for softmax temperature
best_T = 1.0
best_T_score = 0.0
for T in np.linspace(0.05, 2.0, 40):
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

print(f"\n🌡️ Best Softmax Temperature: {best_T:.2f} (CV MAP@3: {best_T_score:.5f})")

def sft(sc, temp):
    e = np.exp(sc / temp)
    return e / (e.sum(1, keepdims=True) + 1e-9)

FTE_SC = sft(sum(BW[i] * models_te[MN[i]] for i in range(len(MN))), best_T)

# Save submission
preds = [" ".join([I2L[i] for i in np.argsort(-r)[:3]]) for r in FTE_SC]
sub = pd.DataFrame({"ID": test_df["id"].values, "Prediction": preds})
sub.to_csv(BASE / "submission.csv", index=False)
sub.to_csv(OUT / "submission_hyper_opt.csv", index=False)
sub.to_csv("/Users/shobhitagnihotri/Desktop/dl_genai2/submission.csv", index=False)
print("🎉 Perfect result submission saved successfully!")
