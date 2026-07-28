import os, json, time
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("/Users/shobhitagnihotri/Downloads/DL GEN AI")
OUT  = BASE / "outputs"
OPTION_COLS = list("ABCDE")
L2I = {c:i for i,c in enumerate(OPTION_COLS)}
I2L = {i:c for c,i in L2I.items()}

# Load cached predictions
st_tr_bg = np.load(OUT / "st_train_bge.npy")
st_te_bg = np.load(OUT / "st_test_bge.npy")
st_tr_mp = np.load(OUT / "st_train_mpnet.npy")
st_te_mp = np.load(OUT / "st_test_mpnet.npy")
st_tr_e5 = np.load(OUT / "st_train_e5.npy")
st_te_e5 = np.load(OUT / "st_test_e5.npy")
ce_tr_ml = np.load(OUT / "ce_train_ce_minilm.npy")
ce_te_ml = np.load(OUT / "ce_test_ce_minilm.npy")
ce_tr_bg = np.load(OUT / "ce_train_bge_rerank.npy")
ce_te_bg = np.load(OUT / "ce_test_bge_rerank.npy")

train_df = pd.read_csv(BASE / "train (1).csv")
train_df.columns = [c.strip() for c in train_df.columns]
test_df = pd.read_csv(BASE / "test (1).csv")
test_df.columns = [c.strip() for c in test_df.columns]
LA = train_df["answer"].map(L2I).values

def norm(a):
    mn = a.min(1, keepdims=True)
    mx = a.max(1, keepdims=True)
    return (a - mn) / (mx - mn + 1e-9)

def apk(a, p, k=3):
    p = p[:k]; s = hits = 0
    for i, x in enumerate(p):
        if x == a and x not in p[:i]: hits += 1; s += hits / (i + 1)
    return s

def mapk(acts, preds, k=3):
    return float(np.mean([apk(a, p, k) for a, p in zip(acts, preds)]))

def top3(sc):
    return [np.argsort(-r)[:3].tolist() for r in sc]

# Optuna search for PURE embedding + CE weights
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

models_tr = {
    "bge": norm(st_tr_bg),
    "mpnet": norm(st_tr_mp),
    "e5": norm(st_tr_e5),
    "ce_minilm": norm(ce_tr_ml),
    "ce_bge_r": norm(ce_tr_bg)
}

models_te = {
    "bge": norm(st_te_bg),
    "mpnet": norm(st_te_mp),
    "e5": norm(st_te_e5),
    "ce_minilm": norm(ce_te_ml),
    "ce_bge_r": norm(ce_te_bg)
}

MN = list(models_tr.keys())

def wens(w, d):
    wa = np.array(w); wa /= (wa.sum() + 1e-9)
    return sum(wa[i] * d[MN[i]] for i in range(len(MN)))

def obj(trial):
    w = [trial.suggest_float(f"w_{n}", 0., 1.) for n in MN]
    s = mapk(LA.tolist(), top3(wens(w, models_tr)))
    return s

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(obj, n_trials=500)

BW = np.array([study.best_trial.params[f"w_{n}"] for n in MN])
BW /= (BW.sum() + 1e-9)

print(f"🏆 Pure Embedding/CE Ensemble OOF MAP@3 = {study.best_trial.value:.4f}")
for n, w in sorted(zip(MN, BW), key=lambda x: -x[1]):
    print(f"  {n}: {w:.4f}")

# Generate test predictions
FTE = wens(BW, models_te)
# Apply gentle scaling
def sft(sc, T):
    e = np.exp(sc / T)
    return e / (e.sum(1, keepdims=True) + 1e-9)

FTE_SC = sft(FTE, 0.4)

preds = [" ".join([I2L[i] for i in np.argsort(-r)[:3]]) for r in FTE_SC]
sub = pd.DataFrame({"ID": test_df["id"].values, "Prediction": preds})
sub.to_csv(BASE / "submission.csv", index=False)
sub.to_csv(BASE / "outputs/submission_pure_embeddings.csv", index=False)
print("🎉 Saved pure embedding submission to base dir and outputs/")
