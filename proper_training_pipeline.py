import os, sys, re, gc, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from joblib import dump, load
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
random_state = SEED

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

# ── Rebuild the exact 24 features using train.py functions ───────────────────────
sys.path.append(str(BASE / "src"))
from train import build_features

print("🔧 Rebuilding the proper 24-feature dataset...")
X_train_raw, tfidf_vec = build_features(train_df, fit=True)
X_test_raw, _ = build_features(test_df, tfidf_vectorizer=tfidf_vec, fit=False)

y_train = np.array([
    1.0 if OPTION_COLS[i % 5] == train_df.iloc[i // 5]["answer"] else 0.0
    for i in range(len(train_df) * 5)
])

def option_probs_from_flat(model, X_flat: np.ndarray, n_options: int = 5) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_flat)[:, 1]
    else:
        probs = model.decision_function(X_flat)
        probs = (probs - probs.min()) / (probs.ptp() + 1e-9)
    n_samples = X_flat.shape[0] // n_options
    return probs.reshape(n_samples, n_options)

# ── Setup tuned booster models ──────────────────────────────────────────────────
models_to_train = {
    "lgbm": lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.015,
        max_depth=5, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=2.0, reg_lambda=2.0,
        min_child_samples=20, random_state=SEED,
        verbose=-1, n_jobs=-1
    ),
    "xgb": xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.015,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=2.0,
        eval_metric="logloss", random_state=SEED,
        verbosity=0, n_jobs=-1,
        early_stopping_rounds=50
    ),
    "cat": cb.CatBoostClassifier(
        iterations=1000, learning_rate=0.015,
        depth=6, l2_leaf_reg=5.0,
        subsample=0.8, random_seed=SEED,
        verbose=0
    ),
    "lr": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)),
    ]),
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    CalibratedClassifierCV(
                       LinearSVC(C=0.5, max_iter=2000, random_state=SEED),
                       cv=3, method="isotonic")),
    ]),
}

oof_probs = {}
test_preds = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
n_samples_train = len(train_df)
label_per_q = y_train.reshape(n_samples_train, 5).argmax(axis=1)

print("\n🚀 Training models with 5-fold CV...")
for name, model in models_to_train.items():
    print(f"  Training {name.upper()}...")
    oof_prob = np.zeros((n_samples_train, 5))
    test_prob_folds = np.zeros((len(test_df), 5, 5))
    
    for fold, (train_q_idx, val_q_idx) in enumerate(skf.split(np.zeros(n_samples_train), label_per_q)):
        train_opt_idx = np.concatenate([np.arange(qi*5, qi*5+5) for qi in train_q_idx])
        val_opt_idx   = np.concatenate([np.arange(qi*5, qi*5+5) for qi in val_q_idx])
        
        X_tr, y_tr = X_train_raw[train_opt_idx], y_train[train_opt_idx]
        X_vl, y_vl = X_train_raw[val_opt_idx],   y_train[val_opt_idx]
        
        if name == "lgbm":
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=[lgb.early_stopping(50, verbose=False)])
        elif name == "xgb":
            model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
        elif name == "cat":
            model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), early_stopping_rounds=50, verbose=False)
        else:
            model.fit(X_tr, y_tr)
            
        oof_prob[val_q_idx] = option_probs_from_flat(model, X_vl)
        test_prob_folds[:, :, fold] = option_probs_from_flat(model, X_test_raw)
        
    cv_score = mapk(label_per_q.tolist(), top3(oof_prob))
    print(f"    {name.upper()} CV MAP@3 = {cv_score:.5f}")
    oof_probs[name] = oof_prob
    test_preds[name] = test_prob_folds.mean(axis=2)

# ── Fit full models on all data (100% training data!) ────────────────────────────
print("\n🔄 Fitting full models on 100% of training data...")
test_full_preds = {}
for name, model in models_to_train.items():
    if name == "lgbm":
        model.fit(X_train_raw, y_train)
    elif name == "xgb":
        model.set_params(early_stopping_rounds=None)
        model.fit(X_train_raw, y_train, verbose=False)
    elif name == "cat":
        model.fit(X_train_raw, y_train, verbose=False)
    else:
        model.fit(X_train_raw, y_train)
    
    test_full_preds[name] = option_probs_from_flat(model, X_test_raw)
    print(f"  ✓ {name} full model fit complete.")

# ── Load sentence transformers ─────────────────────────────────────────────────
st_tr_mp = norm(np.load(OUT / "st_train_mpnet.npy"))
st_te_mp = norm(np.load(OUT / "st_test_mpnet.npy"))
st_tr_ml = norm(np.load(OUT / "st_train_minilm.npy"))
st_te_ml = norm(np.load(OUT / "st_test_minilm.npy"))

# ── Setup ensemble mapping ─────────────────────────────────────────────────────
models_tr = {
    "lgbm": norm(oof_probs["lgbm"]),
    "xgb": norm(oof_probs["xgb"]),
    "cat": norm(oof_probs["cat"]),
    "lr": norm(oof_probs["lr"]),
    "svm": norm(oof_probs["svm"]),
    "st_mpnet": st_tr_mp,
    "st_minilm": st_tr_ml,
}

models_te = {
    "lgbm": norm(test_full_preds["lgbm"]),
    "xgb": norm(test_full_preds["xgb"]),
    "cat": norm(test_full_preds["cat"]),
    "lr": norm(test_full_preds["lr"]),
    "svm": norm(test_full_preds["svm"]),
    "st_mpnet": st_te_mp,
    "st_minilm": st_te_ml,
}

MN = list(models_tr.keys())

# Split into 5 folds for ensembling
folds_indices = list(skf.split(np.zeros(n_samples_train), LA))

def evaluate_weights_cv(weights):
    wa = np.array(weights)
    wa /= (wa.sum() + 1e-9)
    fold_scores = []
    for tr_idx, vl_idx in folds_indices:
        ens_vl = sum(wa[i] * models_tr[MN[i]][vl_idx] for i in range(len(MN)))
        score = mapk(LA[vl_idx].tolist(), top3(ens_vl))
        fold_scores.append(score)
    return np.mean(fold_scores)

print("\n🚀 Running 5000 trials of Optuna weight search...")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True)
)
study.optimize(lambda t: evaluate_weights_cv([t.suggest_float(f"w_{n}", 0.0, 1.0) for n in MN]), n_trials=5000)

BW = np.array([study.best_trial.params[f"w_{n}"] for n in MN])
BW /= (BW.sum() + 1e-9)

print(f"\n🏆 Tuned Ensemble CV MAP@3 = {study.best_trial.value:.5f}")
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
sub.to_csv(OUT / "submission_proper_tuned.csv", index=False)
sub.to_csv("/Users/shobhitagnihotri/Desktop/dl_genai2/submission.csv", index=False)
print("\n🎉 Submission file saved successfully to desktop!")
