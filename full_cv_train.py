#!/usr/bin/env python3
"""
full_cv_train.py — 5-Fold Stratified CV Training Pipeline
===========================================================
Strategy:
  1. Load all 6 cached embedding models (best generalizers)
  2. Train MLP + LightGBM + XGBoost + CatBoost in 5-fold CV — no leakage
  3. Use OOF to Optuna-weight the ensemble
  4. Predict test with all-fold average
  5. Log every fold + final metrics to W&B
  6. Save submission_cv_final.csv to DL GEN AI folder
"""
import gc, warnings, random, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna, wandb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── W&B ────────────────────────────────────────────────────────────────────────
WANDB_API_KEY = "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"
wandb.login(key=WANDB_API_KEY, relogin=True)
run = wandb.init(
    project="mcq-map3-boost",
    name="full_cv_5fold_mlp_gbm",
    config={
        "n_folds": 5,
        "mlp_epochs": 80,
        "mlp_lr": 1e-3,
        "batch_size": 256,
        "optuna_trials": 500,
        "n_emb_models": 6,
        "seed": SEED
    }
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/shobhitagnihotri/Desktop/DL GEN AI")
OUT = BASE / "outputs"

OPTION_COLS = list("ABCDE")
L2I = {c: i for i, c in enumerate(OPTION_COLS)}
I2L = {i: c for c, i in L2I.items()}

# ── Metrics ───────────────────────────────────────────────────────────────────
def apk(a, p, k=3):
    p = p[:k]; s = hits = 0
    for i, x in enumerate(p):
        if x == a and x not in p[:i]: hits += 1; s += hits / (i + 1)
    return s
def mapk(acts, preds, k=3): return float(np.mean([apk(a, p, k) for a, p in zip(acts, preds)]))
def top3(sc): return [np.argsort(-r)[:3].tolist() for r in sc]
def norm(a):
    mn = a.min(1, keepdims=True); mx = a.max(1, keepdims=True)
    return (a - mn) / (mx - mn + 1e-9)

# ── Load data ─────────────────────────────────────────────────────────────────
print("📦 Loading data...")
tr = pd.read_csv(BASE / "train (1).csv"); tr.columns = [c.strip() for c in tr.columns]
te = pd.read_csv(BASE / "test (1).csv");  te.columns = [c.strip() for c in te.columns]
tr["label"] = tr["answer"].map(L2I)
LA = tr["label"].values; NT = len(tr); NTE = len(te)
print(f"   Train={NT}  Test={NTE}")

# ── Load cached embeddings ────────────────────────────────────────────────────
print("📡 Loading cached embeddings (6 models)...")
emb_names_tr = ["st_train_bge.npy", "st_train_e5.npy", "st_train_mpnet.npy",
                 "st_train_minilm.npy", "ce_train_ce_minilm.npy", "ce_train_bge_rerank.npy"]
emb_names_te = ["st_test_bge.npy", "st_test_e5.npy", "st_test_mpnet.npy",
                 "st_test_minilm.npy", "ce_test_ce_minilm.npy", "ce_test_bge_rerank.npy"]

emb_tr = []; emb_te = []
for name_tr, name_te in zip(emb_names_tr, emb_names_te):
    arr_tr = np.load(OUT / name_tr); emb_tr.append(arr_tr)
    arr_te = np.load(OUT / name_te); emb_te.append(arr_te)
    print(f"  ✓ {name_tr}  {arr_tr.shape}")

X_all = np.stack([m.flatten() for m in emb_tr], axis=1)   # (NT*5, 6)
X_test = np.stack([m.flatten() for m in emb_te], axis=1)  # (NTE*5, 6)
y_all = np.zeros(NT * 5)
for i, l in enumerate(LA): y_all[i * 5 + l] = 1.0

# ── MLP definition ─────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class DeepMLP(nn.Module):
    def __init__(self, in_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.35),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.GELU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

# ── 5-Fold Cross-Validation ───────────────────────────────────────────────────
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

mlp_oof = np.zeros((NT, 5)); lgbm_oof = np.zeros((NT, 5))
xgb_oof = np.zeros((NT, 5)); cat_oof = np.zeros((NT, 5))

mlp_test_folds  = np.zeros((NTE, 5, N_FOLDS))
lgbm_test_folds = np.zeros((NTE, 5, N_FOLDS))
xgb_test_folds  = np.zeros((NTE, 5, N_FOLDS))
cat_test_folds  = np.zeros((NTE, 5, N_FOLDS))

print(f"\n🔁 Starting {N_FOLDS}-Fold CV...")

for fold, (tr_q_idx, vl_q_idx) in enumerate(skf.split(np.arange(NT), LA)):
    tr_opt = np.concatenate([np.arange(q*5, q*5+5) for q in tr_q_idx])
    vl_opt = np.concatenate([np.arange(q*5, q*5+5) for q in vl_q_idx])

    X_tr, y_tr = X_all[tr_opt], y_all[tr_opt]
    X_vl, y_vl = X_all[vl_opt], y_all[vl_opt]
    vl_labels  = LA[vl_q_idx]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_vl_s = sc.transform(X_vl)
    X_te_s = sc.transform(X_test)

    # MLP
    model = DeepMLP(X_tr.shape[1]).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    loss_fn = nn.BCEWithLogitsLoss()

    tr_ds = TensorDataset(torch.FloatTensor(X_tr_s), torch.FloatTensor(y_tr))
    tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True)

    best_map3 = 0.0; best_state = None
    for epoch in range(80):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb).squeeze(), yb).backward()
            opt.step()
        sched.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl_sc = torch.sigmoid(model(torch.FloatTensor(X_vl_s).to(DEVICE))).cpu().numpy().squeeze().reshape(-1, 5)
            preds = top3(vl_sc)
            m3 = mapk(vl_labels.tolist(), preds)
            if m3 > best_map3: best_map3 = m3; best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mlp_oof[vl_q_idx] = torch.sigmoid(model(torch.FloatTensor(X_vl_s).to(DEVICE))).cpu().numpy().squeeze().reshape(-1, 5)
        mlp_test_folds[:, :, fold] = torch.sigmoid(model(torch.FloatTensor(X_te_s).to(DEVICE))).cpu().numpy().squeeze().reshape(-1, 5)
    del model; gc.collect()

    # LightGBM
    lg = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, max_depth=8, num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, min_child_samples=10, random_state=SEED, verbose=-1)
    lg.fit(X_tr, y_tr)
    lgbm_oof[vl_q_idx] = lg.predict_proba(X_vl)[:, 1].reshape(-1, 5)
    lgbm_test_folds[:, :, fold] = lg.predict_proba(X_test)[:, 1].reshape(-1, 5)
    del lg; gc.collect()

    # XGBoost
    xgbm = xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, eval_metric="logloss", verbosity=0)
    xgbm.fit(X_tr, y_tr)
    xgb_oof[vl_q_idx] = xgbm.predict_proba(X_vl)[:, 1].reshape(-1, 5)
    xgb_test_folds[:, :, fold] = xgbm.predict_proba(X_test)[:, 1].reshape(-1, 5)
    del xgbm; gc.collect()

    # CatBoost
    cat = cb.CatBoostClassifier(iterations=800, learning_rate=0.02, depth=6, l2_leaf_reg=3, random_seed=SEED, verbose=0)
    cat.fit(X_tr, y_tr)
    cat_oof[vl_q_idx] = cat.predict_proba(X_vl)[:, 1].reshape(-1, 5)
    cat_test_folds[:, :, fold] = cat.predict_proba(X_test)[:, 1].reshape(-1, 5)
    del cat; gc.collect()

mlp_test  = mlp_test_folds.mean(2); lgbm_test = lgbm_test_folds.mean(2)
xgb_test  = xgb_test_folds.mean(2); cat_test  = cat_test_folds.mean(2)

OOF_PREDS = {"mlp": norm(mlp_oof), "lgbm": norm(lgbm_oof), "xgb": norm(xgb_oof), "cat": norm(cat_oof)}
MN = list(OOF_PREDS.keys())

def obj(trial):
    w = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in MN])
    w /= (w.sum() + 1e-9)
    p = sum(w[i] * OOF_PREDS[MN[i]] for i in range(len(MN)))
    return mapk(LA.tolist(), top3(p))

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=50))
study.optimize(obj, n_trials=500)

BW = np.array([study.best_trial.params[f"w_{n}"] for n in MN])
BW /= (BW.sum() + 1e-9)

TEST_PREDS = {"mlp": norm(mlp_test), "lgbm": norm(lgbm_test), "xgb": norm(xgb_test), "cat": norm(cat_test)}
final_test = sum(BW[i] * TEST_PREDS[MN[i]] for i in range(len(MN)))

def sft(sc, T): e = np.exp(sc / T); return e / (e.sum(1, keepdims=True) + 1e-9)
final_test_sc = sft(final_test, 0.3)
preds_str = [" ".join([I2L[i] for i in np.argsort(-r)[:3]]) for r in final_test_sc]
id_col = "id" if "id" in te.columns else "ID"
sub = pd.DataFrame({"id": te[id_col].values, "prediction": preds_str})

out_path = BASE / "submission_cv_final.csv"
sub.to_csv(out_path, index=False)
print(f"🎉 Saved: {out_path}")
