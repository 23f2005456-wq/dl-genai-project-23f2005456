import os
import numpy as np
import pandas as pd
from pathlib import Path
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

# Load predictions
st_tr_bg = np.load(OUT / "st_train_bge.npy")
st_te_bg = np.load(OUT / "st_test_bge.npy")
st_tr_mp = np.load(OUT / "st_train_mpnet.npy")
st_te_mp = np.load(OUT / "st_test_mpnet.npy")

ce_tr_ml = np.load(OUT / "ce_train_ce_minilm.npy")
ce_te_ml = np.load(OUT / "ce_test_ce_minilm.npy")
ce_tr_bg = np.load(OUT / "ce_train_bge_rerank.npy")
ce_te_bg = np.load(OUT / "ce_test_bge_rerank.npy")

oof_lg = np.load(OUT / "oof_lgbm.npy")
te_lg = np.load(OUT / "test_lgbm.npy")
oof_xgb = np.load(OUT / "oof_xgb.npy")
te_xgb = np.load(OUT / "test_xgb.npy")
oof_cat = np.load(OUT / "oof_cat.npy")
te_cat = np.load(OUT / "test_cat.npy")

deberta_oof = np.load(OUT / "deberta_oof.npy")
deberta_test = np.load(OUT / "deberta_test.npy")

# Set up models dict
models_tr = {
    "deberta": norm(deberta_oof),
    "ce_minilm": norm(ce_tr_ml),
    "ce_bge_r": norm(ce_tr_bg),
    "st_bge": norm(st_tr_bg),
    "st_mpnet": norm(st_tr_mp),
    "lgbm": norm(oof_lg),
    "xgb": norm(oof_xgb),
    "cat": norm(oof_cat),
}

models_te = {
    "deberta": norm(deberta_test),
    "ce_minilm": norm(ce_te_ml),
    "ce_bge_r": norm(ce_te_bg),
    "st_bge": norm(st_te_bg),
    "st_mpnet": norm(st_te_mp),
    "lgbm": norm(te_lg),
    "xgb": norm(te_xgb),
    "cat": norm(te_cat),
}

MN = list(models_tr.keys())

# Optimize weights
optuna.logging.set_verbosity(optuna.logging.WARNING)

def wens(w, d):
    wa = np.array(w); wa /= (wa.sum() + 1e-9)
    return sum(wa[i] * d[MN[i]] for i in range(len(MN)))

def obj(trial):
    # To prevent classical models from dominating and causing overfit,
    # we constrain their weight suggestions to a lower range [0.0, 0.2].
    # Deep generalization models (Deberta and Cross Encoders) get [0.0, 1.0].
    w = []
    for n in MN:
        if n in ["lgbm", "xgb", "cat"]:
            w.append(trial.suggest_float(f"w_{n}", 0.0, 0.15))
        elif n in ["deberta"]:
            w.append(trial.suggest_float(f"w_{n}", 0.3, 1.0)) # guarantee DeBERTa presence
        else:
            w.append(trial.suggest_float(f"w_{n}", 0.0, 1.0))
            
    s = mapk(LA.tolist(), top3(wens(w, models_tr)))
    return s

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(obj, n_trials=500)

BW = np.array([study.best_trial.params[f"w_{n}"] for n in MN])
BW /= (BW.sum() + 1e-9)

print(f"🏆 Optimized Ensemble CV MAP@3 = {study.best_trial.value:.4f}")
for n, w in sorted(zip(MN, BW), key=lambda x: -x[1]):
    print(f"  {n}: {w:.4f}")

# Final blend
FTE = wens(BW, models_te)

# Softmax Temperature scaling
def sft(sc, T):
    e = np.exp(sc / T)
    return e / (e.sum(1, keepdims=True) + 1e-9)

FTE_SC = sft(FTE, 0.4)

preds = [" ".join([I2L[i] for i in np.argsort(-r)[:3]]) for r in FTE_SC]
sub = pd.DataFrame({"ID": test_df["id"].values, "Prediction": preds})
sub.to_csv(BASE / "submission.csv", index=False)
sub.to_csv(OUT / "submission_deberta_ensemble.csv", index=False)
print("🎉 Successfully ensembled and saved submission.csv!")
