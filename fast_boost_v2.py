#!/usr/bin/env python3
"""
FAST BOOST v2 — MAP@3 > 0.78 | W&B + Epoch-wise Val MAP@3 Charts
Adds round-by-round val MAP@3 logging for LGBM, XGB, CatBoost.
"""
import os, re, gc, json, warnings, random, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
SEED = 42; random.seed(SEED); np.random.seed(SEED)

# ── W&B ──────────────────────────────────────────────────────────────────────
import wandb
WANDB_API_KEY = "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"
try:
    wandb.login(key=WANDB_API_KEY, relogin=True)
    USE_WANDB = True; print("✅ W&B logged in!")
except Exception as e:
    print(f"⚠️  W&B failed: {e}"); USE_WANDB = False

if USE_WANDB:
    run = wandb.init(
        project="mcq-map3-boost",
        name=f"boost_v2_epochchart_{int(time.time())}",
        tags=["lgbm","xgb","catboost","optuna","epoch-charts"],
        config={
            "seed": SEED, "n_folds": 5,
            "lgbm_lr": 0.03, "lgbm_n_est": 1000, "lgbm_leaves": 31,
            "xgb_lr": 0.03,  "xgb_n_est": 800,
            "cat_lr": 0.03,  "cat_iter": 800,
            "optuna_trials": 500, "metric": "MAP@3",
            "epoch_log_interval": 50,
        }
    )

def wb(d, step=None):
    if USE_WANDB:
        wandb.log(d, step=step) if step is not None else wandb.log(d)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent
OUT    = BASE / "outputs"; OUT.mkdir(exist_ok=True)
(BASE / "models").mkdir(exist_ok=True)
OPTION_COLS  = list("ABCDE")
L2I = {c:i for i,c in enumerate(OPTION_COLS)}
I2L = {i:c for c,i in L2I.items()}

# ── Metric ────────────────────────────────────────────────────────────────────
def apk(a,p,k=3):
    p=p[:k]; s=hits=0
    for i,x in enumerate(p):
        if x==a and x not in p[:i]: hits+=1; s+=hits/(i+1)
    return s
def mapk(acts,preds,k=3): return float(np.mean([apk(a,p,k) for a,p in zip(acts,preds)]))
def top3(sc): return [np.argsort(-r)[:3].tolist() for r in sc]
def norm(a):
    mn=a.min(1,keepdims=True); mx=a.max(1,keepdims=True)
    return (a-mn)/(mx-mn+1e-9)
def top1(p,l): return float((np.argmax(p,1)==l).mean())
def top3a(p,l):
    t=top3(p); return float(np.mean([l[i] in t[i] for i in range(len(l))]))

def probs_to_map3(flat_probs, LPQ_vl):
    """flat_probs: (N_vl*5,) → (N_vl, 5) → MAP@3"""
    p = flat_probs.reshape(-1, 5)
    return mapk(LPQ_vl.tolist(), top3(p))

# ── Load data ─────────────────────────────────────────────────────────────────
print("📦 Loading data...")
tr = pd.read_csv(BASE/"train (1).csv"); tr.columns=[c.strip() for c in tr.columns]
te = pd.read_csv(BASE/"test (1).csv");  te.columns=[c.strip() for c in te.columns]
tr["label"] = tr["answer"].map(L2I)
LA = tr["label"].values; NT=len(tr); NTE=len(te)
print(f"   Train={NT}  Test={NTE}")

# ── Load cached .npy files ────────────────────────────────────────────────────
print("\n📡 Loading cached outputs...")
def npy(name, shape):
    p=OUT/name
    if p.exists():
        a=np.load(p); print(f"  ✓ {name} {a.shape}"); return a
    print(f"  ✗ {name} — zeros {shape}"); return np.zeros(shape)

st_tr_mp  = npy("st_train_mpnet.npy",  (NT,5));  st_te_mp  = npy("st_test_mpnet.npy",  (NTE,5))
st_tr_bg  = npy("st_train_bge.npy",    (NT,5));  st_te_bg  = npy("st_test_bge.npy",    (NTE,5))
st_tr_e5  = npy("st_train_e5.npy",     (NT,5));  st_te_e5  = npy("st_test_e5.npy",     (NTE,5))
st_tr_ml  = npy("st_train_minilm.npy", (NT,5));  st_te_ml  = npy("st_test_minilm.npy", (NTE,5))
ce_tr_ml  = npy("ce_train_ce_minilm.npy",(NT,5));ce_te_ml  = npy("ce_test_ce_minilm.npy",(NTE,5))
ce_tr_bg  = npy("ce_train_bge_rerank.npy",(NT,5));ce_te_bg = npy("ce_test_bge_rerank.npy",(NTE,5))
X_tr_raw  = npy("X_train.npy",(NT*5,20));         X_te_raw  = npy("X_test.npy",(NTE*5,20))
oof_lg=npy("oof_lgbm.npy",(NT,5)); te_lg=npy("test_lgbm.npy",(NTE,5))
oof_xg=npy("oof_xgb.npy",(NT,5));  te_xg=npy("test_xgb.npy",(NTE,5))
oof_ct=npy("oof_cat.npy",(NT,5));  te_ct=npy("test_cat.npy",(NTE,5))
oof_lr=npy("oof_lr.npy",(NT,5));   te_lr=npy("test_lr.npy",(NTE,5))
oof_sv=npy("oof_svm.npy",(NT,5));  te_sv=npy("test_svm.npy",(NTE,5))

# ── Vectorized feature engineering ───────────────────────────────────────────
print("\n🔧 Vectorized feature engineering...")
STOP = {"the","a","an","is","it","in","on","at","of","to","and","or","but","for",
        "not","with","that","this","from","by","are","was","be","as","have","has",
        "had","they","their","there","were","which","will","if"}

def kw_overlap_col(prompt_s, opt_s):
    out = []
    for p, o in zip(prompt_s, opt_s):
        sp = set(re.sub(r"[^a-z0-9\s]"," ",str(p).lower()).split()) - STOP
        so = set(re.sub(r"[^a-z0-9\s]"," ",str(o).lower()).split()) - STOP
        out.append(0.0 if not sp and not so else len(sp&so)/(len(sp|so)+1e-9))
    return np.array(out, dtype=np.float32)

def build_feats(df):
    N = len(df)
    p_len = df["prompt"].str.len().values.astype(float)
    p_wds = df["prompt"].str.split().str.len().values.astype(float)
    o_len = np.stack([df[c].str.len().values for c in OPTION_COLS],1).astype(float)
    o_wds = np.stack([df[c].str.split().str.len().values for c in OPTION_COLS],1).astype(float)
    mean_l = o_len.mean(1,keepdims=True)+1e-9; mean_w = o_wds.mean(1,keepdims=True)+1e-9
    len_rank = np.argsort(np.argsort(-o_len,1),1).astype(float)
    unigram = np.stack([kw_overlap_col(df["prompt"], df[c]) for c in OPTION_COLS],1)
    has_neg = np.stack([df[c].str.lower().str.contains(
        r"\bnot\b|\bno\b|\bnever\b|\bcannot\b", regex=True, na=False).values.astype(float)
        for c in OPTION_COLS],1)
    starts_num = np.stack([df[c].str.match(r"^\d",na=False).values.astype(float) for c in OPTION_COLS],1)
    num_cnt = np.stack([df[c].str.count(r"\b\d+\.?\d*\b").values.astype(float) for c in OPTION_COLS],1)
    pos = np.tile(np.arange(5,dtype=float),(N,1))
    plr = np.tile(p_len.reshape(-1,1),(1,5)); pwr = np.tile(p_wds.reshape(-1,1),(1,5))
    feat = np.stack([unigram, o_len/(plr+1), o_wds/(pwr+1), o_len, o_wds,
                     len_rank, o_len/mean_l, o_wds/mean_w, starts_num,
                     has_neg, num_cnt, pos, plr, pwr],2)
    return feat.reshape(N*5,-1).astype(np.float32)

t0=time.time()
hc_tr=build_feats(tr); hc_te=build_feats(te)
print(f"   Done in {time.time()-t0:.1f}s  train:{hc_tr.shape}  test:{hc_te.shape}")
X_tr = np.concatenate([X_tr_raw,hc_tr],1) if X_tr_raw.shape[0]==hc_tr.shape[0] else hc_tr
X_te = np.concatenate([X_te_raw,hc_te],1) if X_te_raw.shape[0]==hc_te.shape[0] else hc_te
print(f"   Combined: train={X_tr.shape}  test={X_te.shape}")

y_tr  = np.array([1. if OPTION_COLS[i%5]==tr.iloc[i//5]["answer"] else 0. for i in range(NT*5)])
LPQ   = y_tr.reshape(NT,5).argmax(axis=1)

# ── Imports ───────────────────────────────────────────────────────────────────
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

SKF = StratifiedKFold(5, shuffle=True, random_state=SEED)

def oprobs(m, X):
    p = m.predict_proba(X)[:,1] if hasattr(m,"predict_proba") else m.decision_function(X)
    if not hasattr(m,"predict_proba"): p=(p-p.min())/(p.ptp()+1e-9)
    return p.reshape(-1,5)

# ════════════════════════════════════════════════════════════════════════════
# EPOCH-WISE VAL MAP@3 CALLBACKS
# ════════════════════════════════════════════════════════════════════════════
LOG_INTERVAL = 50   # log MAP@3 every N rounds/iterations

# ── LightGBM custom callback ─────────────────────────────────────────────────
def make_lgbm_epoch_callback(X_vl_full, LPQ_vl, tag, fold):
    """
    Returns a LightGBM callback that logs val MAP@3 + val logloss every LOG_INTERVAL rounds.
    Stores history for W&B line chart.
    """
    history = {"round": [], "val_map3": [], "val_logloss": []}

    def callback(env):
        iteration = env.iteration
        val_logloss = None
        for row in env.evaluation_result_list:
            if "valid" in row[0] and "binary_logloss" in row[1]:
                val_logloss = row[2]
        if (iteration + 1) % LOG_INTERVAL == 0 or iteration == 0:
            # Predict proba on val set using current model
            probs = env.model.predict(X_vl_full)       # shape (N_vl*5,)
            map3  = probs_to_map3(probs, LPQ_vl)
            history["round"].append(iteration + 1)
            history["val_map3"].append(map3)
            if val_logloss: history["val_logloss"].append(val_logloss)
            wb({
                f"{tag}/fold{fold}_epoch_val_map3":   map3,
                f"{tag}/fold{fold}_epoch_val_logloss": val_logloss or 0,
                f"{tag}/fold{fold}_round": iteration + 1,
            })
    callback.order = 10
    return callback, history

# ── XGBoost custom callback ───────────────────────────────────────────────────
class XGBEpochCallback(xgb.callback.TrainingCallback):
    def __init__(self, X_vl_full, LPQ_vl, tag, fold):
        self.X_vl   = X_vl_full
        self.lpq    = LPQ_vl
        self.tag    = tag
        self.fold   = fold
        self.history= {"round":[],"val_map3":[],"val_logloss":[]}

    def after_iteration(self, model, epoch, evals_log):
        if (epoch + 1) % LOG_INTERVAL == 0 or epoch == 0:
            probs = model.predict(xgb.DMatrix(self.X_vl))
            map3  = probs_to_map3(probs, self.lpq)
            val_ll = None
            try: val_ll = evals_log["validation"]["logloss"][-1]
            except: pass
            self.history["round"].append(epoch+1)
            self.history["val_map3"].append(map3)
            if val_ll: self.history["val_logloss"].append(val_ll)
            wb({
                f"{self.tag}/fold{self.fold}_epoch_val_map3":   map3,
                f"{self.tag}/fold{self.fold}_epoch_val_logloss": val_ll or 0,
                f"{self.tag}/fold{self.fold}_round": epoch+1,
            })
        return False  # don't stop training

# ── CatBoost custom callback ──────────────────────────────────────────────────
class CatEpochCallback:
    def __init__(self, X_vl_full, LPQ_vl, tag, fold, model_ref_list):
        self.X_vl  = X_vl_full
        self.lpq   = LPQ_vl
        self.tag   = tag
        self.fold  = fold
        self.mref  = model_ref_list   # mutable list so we can inject model ref
        self.history = {"round":[],"val_map3":[]}
        self._iter = 0

    def after_iteration(self, info):
        self._iter += 1
        if self._iter % LOG_INTERVAL == 0 or self._iter == 1:
            if self.mref:
                m = self.mref[0]
                probs = m.predict_proba(self.X_vl)[:,1]
                map3  = probs_to_map3(probs, self.lpq)
                val_ll = info.metrics.get("validation",{}).get("Logloss",[[]])
                val_ll = val_ll[-1] if val_ll else 0
                self.history["round"].append(self._iter)
                self.history["val_map3"].append(map3)
                wb({
                    f"{self.tag}/fold{self.fold}_epoch_val_map3":   map3,
                    f"{self.tag}/fold{self.fold}_epoch_val_logloss": val_ll,
                    f"{self.tag}/fold{self.fold}_round": self._iter,
                })
        return True  # continue training

# ════════════════════════════════════════════════════════════════════════════
# CV Training with epoch-wise logging
# ════════════════════════════════════════════════════════════════════════════
def cv_lgbm(X_tr, X_te, y_tr, LPQ):
    tag="lgbm"; print(f"\n  ⚡ LightGBM (with epoch MAP@3 charts)...")
    oof=np.zeros((NT,5)); te_f=np.zeros((NTE,5,5)); fsc=[]
    all_histories = []

    for k,(trq,vlq) in enumerate(SKF.split(np.zeros(NT),LPQ)):
        tro=np.concatenate([np.arange(qi*5,qi*5+5) for qi in trq])
        vlo=np.concatenate([np.arange(qi*5,qi*5+5) for qi in vlq])
        Xtr=X_tr[tro]; ytr=y_tr[tro]; Xvl=X_tr[vlo]; yvl=y_tr[vlo]

        epoch_cb, hist = make_lgbm_epoch_callback(Xvl, LPQ[vlq], tag, k+1)

        m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0, min_child_samples=10,
            random_state=SEED, verbose=-1, n_jobs=-1)
        m.fit(Xtr, ytr, eval_set=[(Xvl,yvl)],
              callbacks=[lgb.early_stopping(80,verbose=False),
                         lgb.log_evaluation(-1), epoch_cb])

        oof[vlq]=oprobs(m,X_tr[vlo]); te_f[:,:,k]=oprobs(m,X_te)
        s=mapk(LPQ[vlq].tolist(),top3(oof[vlq])); fsc.append(s)
        all_histories.append(hist)

        print(f"     fold {k+1}: MAP@3={s:.4f}  best_round={m.best_iteration_}")
        wb({f"{tag}/fold{k+1}_map3":s, f"{tag}/fold{k+1}_top1":top1(oof[vlq],LPQ[vlq]),
            f"{tag}/fold{k+1}_best_round": m.best_iteration_ or 0})

    # Log averaged epoch curve across all folds
    if all_histories and all_histories[0]["round"]:
        max_rounds = max(max(h["round"]) for h in all_histories if h["round"])
        common_rounds = sorted(set(r for h in all_histories for r in h["round"]))
        for rnd in common_rounds:
            vals = [h["val_map3"][h["round"].index(rnd)]
                    for h in all_histories if rnd in h["round"]]
            if vals:
                wb({f"{tag}/avg_epoch_val_map3": float(np.mean(vals)),
                    f"{tag}/epoch": rnd})

    cv=mapk(LPQ.tolist(),top3(oof))
    print(f"  ► LightGBM CV MAP@3={cv:.4f} ±{np.std(fsc):.4f}")
    wb({f"{tag}/cv_map3":cv, f"{tag}/cv_std":float(np.std(fsc)),
        f"{tag}/cv_top1":top1(oof,LPQ), f"{tag}/cv_top3":top3a(oof,LPQ)})
    return oof, te_f.mean(2), cv


def cv_xgb(X_tr, X_te, y_tr, LPQ):
    tag="xgb"; print(f"\n  ⚡ XGBoost (with epoch MAP@3 charts)...")
    oof=np.zeros((NT,5)); te_f=np.zeros((NTE,5,5)); fsc=[]
    all_histories = []

    for k,(trq,vlq) in enumerate(SKF.split(np.zeros(NT),LPQ)):
        tro=np.concatenate([np.arange(qi*5,qi*5+5) for qi in trq])
        vlo=np.concatenate([np.arange(qi*5,qi*5+5) for qi in vlq])
        Xtr=X_tr[tro]; ytr=y_tr[tro]; Xvl=X_tr[vlo]; yvl=y_tr[vlo]

        epoch_cb = XGBEpochCallback(Xvl, LPQ[vlq], tag, k+1)

        m = xgb.XGBClassifier(n_estimators=800, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
            eval_metric="logloss", random_state=SEED, verbosity=0, n_jobs=-1,
            early_stopping_rounds=60, callbacks=[epoch_cb])
        m.fit(Xtr, ytr, eval_set=[(Xvl,yvl)], verbose=False)

        oof[vlq]=oprobs(m,X_tr[vlo]); te_f[:,:,k]=oprobs(m,X_te)
        s=mapk(LPQ[vlq].tolist(),top3(oof[vlq])); fsc.append(s)
        all_histories.append(epoch_cb.history)

        print(f"     fold {k+1}: MAP@3={s:.4f}  best_round={m.best_iteration}")
        wb({f"{tag}/fold{k+1}_map3":s, f"{tag}/fold{k+1}_top1":top1(oof[vlq],LPQ[vlq]),
            f"{tag}/fold{k+1}_best_round": m.best_iteration})

    # Average epoch curve
    if all_histories and all_histories[0]["round"]:
        for rnd in sorted(set(r for h in all_histories for r in h["round"])):
            vals = [h["val_map3"][h["round"].index(rnd)]
                    for h in all_histories if rnd in h["round"]]
            if vals:
                wb({f"{tag}/avg_epoch_val_map3": float(np.mean(vals)),
                    f"{tag}/epoch": rnd})

    cv=mapk(LPQ.tolist(),top3(oof))
    print(f"  ► XGBoost CV MAP@3={cv:.4f} ±{np.std(fsc):.4f}")
    wb({f"{tag}/cv_map3":cv, f"{tag}/cv_std":float(np.std(fsc)),
        f"{tag}/cv_top1":top1(oof,LPQ), f"{tag}/cv_top3":top3a(oof,LPQ)})
    return oof, te_f.mean(2), cv


def cv_cat(X_tr, X_te, y_tr, LPQ):
    tag="cat"; print(f"\n  ⚡ CatBoost (with epoch MAP@3 charts)...")
    oof=np.zeros((NT,5)); te_f=np.zeros((NTE,5,5)); fsc=[]
    all_histories = []

    for k,(trq,vlq) in enumerate(SKF.split(np.zeros(NT),LPQ)):
        tro=np.concatenate([np.arange(qi*5,qi*5+5) for qi in trq])
        vlo=np.concatenate([np.arange(qi*5,qi*5+5) for qi in vlq])
        Xtr=X_tr[tro]; ytr=y_tr[tro]; Xvl=X_tr[vlo]; yvl=y_tr[vlo]

        model_ref = []
        epoch_cb  = CatEpochCallback(Xvl, LPQ[vlq], tag, k+1, model_ref)

        m = cb.CatBoostClassifier(iterations=800, learning_rate=0.03, depth=6,
            l2_leaf_reg=5.0, subsample=0.8, random_seed=SEED, verbose=0,
            eval_metric="Logloss")
        model_ref.append(m)   # inject reference before fit
        m.fit(Xtr, ytr, eval_set=(Xvl,yvl), early_stopping_rounds=60,
              verbose=False, callbacks=[epoch_cb])

        oof[vlq]=oprobs(m,X_tr[vlo]); te_f[:,:,k]=oprobs(m,X_te)
        s=mapk(LPQ[vlq].tolist(),top3(oof[vlq])); fsc.append(s)
        all_histories.append(epoch_cb.history)

        print(f"     fold {k+1}: MAP@3={s:.4f}  best_round={m.best_iteration_}")
        wb({f"{tag}/fold{k+1}_map3":s, f"{tag}/fold{k+1}_top1":top1(oof[vlq],LPQ[vlq]),
            f"{tag}/fold{k+1}_best_round": m.best_iteration_})

    # Average epoch curve
    if all_histories and all_histories[0]["round"]:
        for rnd in sorted(set(r for h in all_histories for r in h["round"])):
            vals = [h["val_map3"][h["round"].index(rnd)]
                    for h in all_histories if rnd in h["round"]]
            if vals:
                wb({f"{tag}/avg_epoch_val_map3": float(np.mean(vals)),
                    f"{tag}/epoch": rnd})

    cv=mapk(LPQ.tolist(),top3(oof))
    print(f"  ► CatBoost CV MAP@3={cv:.4f} ±{np.std(fsc):.4f}")
    wb({f"{tag}/cv_map3":cv, f"{tag}/cv_std":float(np.std(fsc)),
        f"{tag}/cv_top1":top1(oof,LPQ), f"{tag}/cv_top3":top3a(oof,LPQ)})
    return oof, te_f.mean(2), cv


# ── Run all models ────────────────────────────────────────────────────────────
oof_lg2, te_lg2, cv_lg2 = cv_lgbm(X_tr, X_te, y_tr, LPQ)
np.save(OUT/"oof_lgbm2.npy",oof_lg2); np.save(OUT/"test_lgbm2.npy",te_lg2)

oof_xg2, te_xg2, cv_xg2 = cv_xgb(X_tr, X_te, y_tr, LPQ)
np.save(OUT/"oof_xgb2.npy",oof_xg2); np.save(OUT/"test_xgb2.npy",te_xg2)

oof_ct2, te_ct2, cv_ct2 = cv_cat(X_tr, X_te, y_tr, LPQ)
np.save(OUT/"oof_cat2.npy",oof_ct2); np.save(OUT/"test_cat2.npy",te_ct2)

# ── Meta-LR stacker ───────────────────────────────────────────────────────────
print("\n  ⚡ Meta-LR stacker...")
EMB_TR=np.concatenate([norm(x) for x in [st_tr_mp,st_tr_bg,st_tr_e5,st_tr_ml,ce_tr_ml,ce_tr_bg]],1)
EMB_TE=np.concatenate([norm(x) for x in [st_te_mp,st_te_bg,st_te_e5,st_te_ml,ce_te_ml,ce_te_bg]],1)
oof_meta=np.zeros((NT,5)); te_meta_f=np.zeros((NTE,5,5))
for k,(trq,vlq) in enumerate(SKF.split(np.zeros(NT),LPQ)):
    lr=Pipeline([("sc",StandardScaler()),("clf",LogisticRegression(C=2.0,max_iter=500,
        random_state=SEED,multi_class="multinomial",solver="lbfgs"))])
    lr.fit(EMB_TR[trq],LPQ[trq])
    oof_meta[vlq]=lr.predict_proba(EMB_TR[vlq]); te_meta_f[:,:,k]=lr.predict_proba(EMB_TE)
    s=mapk(LPQ[vlq].tolist(),top3(oof_meta[vlq]))
    print(f"     fold {k+1}: MAP@3={s:.4f}"); wb({f"meta_lr/fold{k+1}_map3":s})
te_meta=te_meta_f.mean(2)
cv_meta=mapk(LPQ.tolist(),top3(oof_meta))
print(f"  ► Meta-LR CV MAP@3={cv_meta:.4f}"); wb({"meta_lr/cv_map3":cv_meta})

# ── Borda fusion ──────────────────────────────────────────────────────────────
def borda(sc):
    out=np.zeros_like(sc)
    for i in range(sc.shape[0]):
        for rank,idx in enumerate(np.argsort(-sc[i])): out[i,idx]=5-rank
    return out.astype(float)

EMB_MEAN_TR=np.mean([norm(x) for x in [st_tr_mp,st_tr_bg,st_tr_e5,st_tr_ml,ce_tr_ml,ce_tr_bg]],0)
EMB_MEAN_TE=np.mean([norm(x) for x in [st_te_mp,st_te_bg,st_te_e5,st_te_ml,ce_te_ml,ce_te_bg]],0)

# ── Collect all predictions ───────────────────────────────────────────────────
print("\n📊 All models:")
TP,TEP={},{}
def add(n,tr,te): TP[n]=norm(tr); TEP[n]=norm(te)

add("lgbm_v1",oof_lg,te_lg);  add("xgb_v1",oof_xg,te_xg)
add("cat_v1",oof_ct,te_ct);   add("lr_v1",oof_lr,te_lr);  add("svm_v1",oof_sv,te_sv)
add("lgbm_v2",oof_lg2,te_lg2); add("xgb_v2",oof_xg2,te_xg2); add("cat_v2",oof_ct2,te_ct2)
add("meta_lr",oof_meta,te_meta)
add("st_mpnet",st_tr_mp,st_te_mp); add("st_bge",st_tr_bg,st_te_bg)
add("st_e5",st_tr_e5,st_te_e5);   add("st_minilm",st_tr_ml,st_te_ml)
add("ce_minilm",ce_tr_ml,ce_te_ml); add("ce_bge_r",ce_tr_bg,ce_te_bg)
add("emb_ensemble",EMB_MEAN_TR,EMB_MEAN_TE)
add("borda_emb",borda(EMB_MEAN_TR),borda(EMB_MEAN_TE))

ind={}
for nm,pr in TP.items():
    s=mapk(LA.tolist(),top3(pr)); ind[nm]=s
    print(f"  {nm:<26} MAP@3={s:.4f}")
wb({f"individual/{nm}":s for nm,s in ind.items()})

# ── Optuna ────────────────────────────────────────────────────────────────────
print("\n🔍 Optuna 500 trials...")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
MN=list(TP.keys()); hist=[]

def wens(w,d):
    wa=np.array(w); wa/=(wa.sum()+1e-9)
    return sum(wa[i]*d[MN[i]] for i in range(len(MN)))

def obj(trial):
    w=[trial.suggest_float(f"w_{n}",0.,1.) for n in MN]
    s=mapk(LA.tolist(),top3(wens(w,TP))); hist.append(s)
    if len(hist)%50==0: wb({"optuna/best":max(hist),"optuna/trial":len(hist)})
    return s

study=optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=50))
study.optimize(obj,n_trials=500,show_progress_bar=True)

BW=np.array([study.best_trial.params[f"w_{n}"] for n in MN])
BW/=(BW.sum()+1e-9)
best_map3=study.best_trial.value
print(f"\n🏆 Optuna best MAP@3={best_map3:.4f}")
for n,w in sorted(zip(MN,BW),key=lambda x:-x[1]):
    print(f"  {n:<28} {w:.4f}")
wb({"optuna/best_map3":best_map3,**{f"weights/{n}":float(w) for n,w in zip(MN,BW)}})

# ── Temperature scaling ───────────────────────────────────────────────────────
FT=wens(BW,TP); FTE=wens(BW,TEP)
def sft(sc,T): e=np.exp(sc/T); return e/(e.sum(1,keepdims=True)+1e-9)
tsc={}
for T in [0.3,0.5,0.7,0.8,0.9,1.0,1.2,1.5,2.0]:
    s=mapk(LA.tolist(),top3(sft(FT,T))); tsc[T]=s; wb({f"temperature/T{T}":s})
BEST_T=max(tsc,key=tsc.get)
print(f"\n🌡️  Best T={BEST_T}  MAP@3={tsc[BEST_T]:.4f}")

FT_SC=sft(FT,BEST_T); FTE_SC=sft(FTE,BEST_T)

# ── Final metrics ─────────────────────────────────────────────────────────────
fm3=mapk(LA.tolist(),top3(FT_SC))
ft1=top1(FT_SC,LA); ft3=top3a(FT_SC,LA)
mx=FT_SC.max(1); ok=(np.argmax(FT_SC,1)==LA)
cc=mx[ok].mean(); cw=mx[~ok].mean() if (~ok).sum()>0 else 0.
pc={opt:top1(FT_SC[LA==L2I[opt]],LA[LA==L2I[opt]]) for opt in OPTION_COLS if (LA==L2I[opt]).sum()>0}

print(f"\n✅ FINAL METRICS:")
print(f"   MAP@3      = {fm3:.4f}")
print(f"   Top-1 Acc  = {ft1:.4f}")
print(f"   Top-3 Acc  = {ft3:.4f}")
print(f"   Conf(✓)    = {cc:.4f}")
print(f"   Conf(✗)    = {cw:.4f}")
for opt,s in pc.items(): print(f"   Class {opt} Top-1 = {s:.4f}")

wb({"final/map3":fm3,"final/top1":ft1,"final/top3":ft3,
    "final/temperature":BEST_T,"final/conf_correct":cc,"final/conf_wrong":cw,
    "final/conf_gap":cc-cw,"final/n_models":len(MN),
    **{f"final/class_{o}_top1":s for o,s in pc.items()}})

# ── Generate submission ───────────────────────────────────────────────────────
print("\n📄 Generating submission.csv...")
preds=[" ".join([I2L[i] for i in np.argsort(-r)[:3]]) for r in FTE_SC]
sub=pd.DataFrame({"ID":te["id"].values,"Prediction":preds})
sub.to_csv(OUT/"submission_boost_v2.csv",index=False)
sub.to_csv(BASE/"submission.csv",index=False)
print(f"\n🎉 submission.csv → {BASE/'submission.csv'}")
print(sub.head(10).to_string(index=False))

summary={"weights":{n:float(w) for n,w in zip(MN,BW)},
          "cv_map3":float(fm3),"cv_top1":float(ft1),"cv_top3":float(ft3),
          "temperature":float(BEST_T),"individual":ind}
with open(OUT/"best_weights_boost_v2.json","w") as f: json.dump(summary,f,indent=2)

if USE_WANDB:
    art=wandb.Artifact("submission_v2",type="predictions")
    art.add_file(str(BASE/"submission.csv"))
    art.add_file(str(OUT/"best_weights_boost_v2.json"))
    wandb.log_artifact(art)

    # Model leaderboard table
    tbl=wandb.Table(columns=["model","map3","rank"],
        data=[[nm,round(s,4),i+1] for i,(nm,s) in
              enumerate(sorted(ind.items(),key=lambda x:-x[1]))])
    wandb.log({"model_leaderboard":tbl})

    # Epoch curve summary tables
    for tag in ["lgbm","xgb","cat"]:
        wb({f"{tag}/final_cv_map3": locals().get(f"cv_{tag[:2]}2", fm3)})

    wandb.summary.update({
        "final_map3": fm3, "final_top1": ft1, "best_T": BEST_T,
        "best_model": max(ind,key=ind.get),
        "lgbm_v2_cv": cv_lg2, "xgb_v2_cv": cv_xg2, "cat_v2_cv": cv_ct2,
    })
    wandb.finish()
    print("✅ W&B run complete!")

print(f"\n{'='*55}")
print(f"  FINAL MAP@3 (OOF CV) = {fm3:.4f}")
print(f"  submission.csv → {BASE/'submission.csv'}")
print(f"{'='*55}")

# Log: Print out average training times alongside validation scores