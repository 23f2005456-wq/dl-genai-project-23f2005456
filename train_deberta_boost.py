#!/usr/bin/env python3
"""
train_deberta_boost.py — Target: Kaggle MAP@3 > 0.78
=====================================================
Strategy:
  1. Fine-tune DeBERTa-v3-small (3-fold, 4 epochs) on MPS/CPU
  2. Blend DeBERTa OOF with cached embedding similarity scores
  3. Optuna optimizes blend weights (emphasize DeBERTa + embeddings)
  4. Generate submission.csv with correct Kaggle column format
=====================================================
"""
import os, re, gc, json, time, warnings, random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42; random.seed(SEED); np.random.seed(SEED)

# ── W&B ──────────────────────────────────────────────────────────────────────
import wandb
try:
    wandb.login()
    USE_WANDB = True; print("✅ W&B connected")
except Exception as e:
    print(f"⚠️  W&B: {e}"); USE_WANDB = False

if USE_WANDB:
    run = wandb.init(
        project="mcq-competition-ensemble",
        name=f"deberta_boost_{int(time.time())}",
        tags=["deberta","fine-tune","embedding-ensemble","mps"],
        config={
            "seed": SEED, "n_folds": 3,
            "model_id": "microsoft/deberta-v3-small",
            "max_len": 256, "batch_size": 2, "accum_steps": 8,
            "epochs": 4, "lr": 2e-5, "weight_decay": 0.01,
            "label_smoothing": 0.1,
            "optuna_trials": 500,
        }
    )
def wb(d):
    if USE_WANDB: wandb.log(d)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
OUT  = BASE / "outputs"; OUT.mkdir(exist_ok=True)
MDL  = BASE / "models";  MDL.mkdir(exist_ok=True)
OPTION_COLS = list("ABCDE")
L2I = {c:i for i,c in enumerate(OPTION_COLS)}
I2L = {i:c for c,i in L2I.items()}

# ── Metrics ───────────────────────────────────────────────────────────────────
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
def acc1(p,l): return float((np.argmax(p,1)==l).mean())

# ── Data ──────────────────────────────────────────────────────────────────────
print("\n📦 Loading data...")
tr = pd.read_csv(BASE/"train (1).csv"); tr.columns=[c.strip() for c in tr.columns]
te = pd.read_csv(BASE/"test (1).csv");  te.columns=[c.strip() for c in te.columns]
tr["label"] = tr["answer"].map(L2I)
LA = tr["label"].values; NT=len(tr); NTE=len(te)
print(f"   Train={NT}  Test={NTE}")

# ── Load cached embeddings ────────────────────────────────────────────────────
print("\n📡 Loading cached embedding scores...")
def npy(name, shape):
    p=OUT/name
    if p.exists(): a=np.load(p); print(f"  ✓ {name} {a.shape}"); return a
    print(f"  ✗ {name} — zeros"); return np.zeros(shape)

st_tr_mp = npy("st_train_mpnet.npy",(NT,5));  st_te_mp = npy("st_test_mpnet.npy",(NTE,5))
st_tr_bg = npy("st_train_bge.npy",(NT,5));    st_te_bg = npy("st_test_bge.npy",(NTE,5))
st_tr_e5 = npy("st_train_e5.npy",(NT,5));     st_te_e5 = npy("st_test_e5.npy",(NTE,5))
st_tr_ml = npy("st_train_minilm.npy",(NT,5)); st_te_ml = npy("st_test_minilm.npy",(NTE,5))
ce_tr_ml = npy("ce_train_ce_minilm.npy",(NT,5)); ce_te_ml = npy("ce_test_ce_minilm.npy",(NTE,5))
ce_tr_bg = npy("ce_train_bge_rerank.npy",(NT,5)); ce_te_bg = npy("ce_test_bge_rerank.npy",(NTE,5))

# ══════════════════════════════════════════════════════════════════════════════
# DeBERTa Fine-Tuning (THE KEY to >0.76 on Kaggle)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  🤖 DeBERTa-v3-small Fine-Tuning (3-fold, 4 epochs)")
print("="*60)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForMultipleChoice,
                           get_cosine_schedule_with_warmup)
from torch.optim import AdamW
import torch.nn.functional as F

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps"); print("  ✅ Using Apple MPS (M1 GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda"); print("  ✅ Using CUDA GPU")
else:
    DEVICE = torch.device("cpu");  print("  ℹ️  Using CPU (will be slow)")
torch.manual_seed(SEED)

MODEL_ID      = "microsoft/deberta-v3-small"
MAX_LEN       = 256
BATCH_SZ      = 2
ACCUM         = 8          # effective batch = 16
EPOCHS        = 4
LR            = 2e-5
WD            = 0.01
LABEL_SMOOTH  = 0.1
N_FOLDS_DEB   = 3

class MCQDataset(Dataset):
    def __init__(self, df, tokenizer, has_labels=True):
        self.df=df.reset_index(drop=True)
        self.tok=tokenizer; self.hl=has_labels
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row=self.df.iloc[idx]; prompt=str(row["prompt"])
        enc=[self.tok(prompt,str(row[c]),max_length=MAX_LEN,truncation=True,
                      padding="max_length",return_tensors="pt") for c in OPTION_COLS]
        item={"input_ids":      torch.stack([e["input_ids"].squeeze() for e in enc]),
              "attention_mask": torch.stack([e["attention_mask"].squeeze() for e in enc])}
        if "token_type_ids" in enc[0]:
            item["token_type_ids"]=torch.stack([e["token_type_ids"].squeeze() for e in enc])
        if self.hl:
            item["labels"]=torch.tensor(L2I[str(row["answer"])],dtype=torch.long)
        return item

def smooth_ce(logits, labels, eps=LABEL_SMOOTH):
    lp=F.log_softmax(logits,-1)
    nll=F.nll_loss(lp,labels)
    smooth=-lp.mean(-1).mean()
    return (1-eps)*nll + eps*smooth

from sklearn.model_selection import StratifiedKFold

deberta_oof   = np.zeros((NT, 5))
deberta_test_folds = np.zeros((NTE, 5, N_FOLDS_DEB))
skf_deb = StratifiedKFold(N_FOLDS_DEB, shuffle=True, random_state=SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

for fold, (tr_idx, vl_idx) in enumerate(skf_deb.split(tr, LA)):
    print(f"\n{'─'*55}")
    print(f"  DeBERTa Fold {fold+1}/{N_FOLDS_DEB}")
    print(f"{'─'*55}")

    tr_ds = MCQDataset(tr.iloc[tr_idx], tokenizer)
    vl_ds = MCQDataset(tr.iloc[vl_idx], tokenizer)
    te_ds = MCQDataset(te, tokenizer, has_labels=False)

    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SZ, shuffle=True,  num_workers=0)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH_SZ*2, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SZ*2, shuffle=False, num_workers=0)

    model = AutoModelForMultipleChoice.from_pretrained(MODEL_ID).to(DEVICE)
    # Gradient checkpointing incompatible with MPS
    try:
        if str(DEVICE) == "mps":
            model.gradient_checkpointing_disable()
        else:
            model.gradient_checkpointing_enable()
    except: pass

    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = max(1, len(tr_ld) // ACCUM) * EPOCHS
    warmup = int(total_steps * 0.1)
    sched  = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    best_map3 = 0.0; best_state = None

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────
        model.train(); opt.zero_grad(); ep_loss=0.0
        for step, batch in enumerate(tr_ld):
            labels = batch.pop("labels").to(DEVICE)
            batch  = {k:v.to(DEVICE) for k,v in batch.items()}
            out    = model(**batch)
            loss   = smooth_ce(out.logits, labels) / ACCUM
            loss.backward()
            if (step+1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            ep_loss += loss.item() * ACCUM

            # Progress indicator every 50 steps
            if (step+1) % 50 == 0:
                print(f"   Step {step+1}/{len(tr_ld)} | loss={ep_loss/(step+1):.4f}")

        avg_loss = ep_loss / len(tr_ld)

        # ── Validate ────────────────────────────────────────────────────
        model.eval(); vl_logits=[]; vl_labels=[]
        with torch.no_grad():
            for batch in vl_ld:
                labs = batch.pop("labels").to(DEVICE)
                batch = {k:v.to(DEVICE) for k,v in batch.items()}
                out = model(**batch)
                vl_logits.append(out.logits.float().cpu())
                vl_labels.append(labs.cpu())

        vl_logits = torch.cat(vl_logits).numpy()
        vl_labels = torch.cat(vl_labels).numpy()
        vl_probs  = torch.softmax(torch.tensor(vl_logits),-1).numpy()
        val_map3  = mapk(vl_labels.tolist(), top3(vl_probs))
        val_top1  = acc1(vl_probs, vl_labels)

        print(f"   Ep{epoch+1}/{EPOCHS} | loss={avg_loss:.4f} | val_MAP@3={val_map3:.4f} | top1={val_top1:.4f}")
        wb({
            f"deberta/fold{fold+1}_epoch{epoch+1}_val_map3":  val_map3,
            f"deberta/fold{fold+1}_epoch{epoch+1}_val_top1":  val_top1,
            f"deberta/fold{fold+1}_epoch{epoch+1}_train_loss": avg_loss,
        })

        if val_map3 > best_map3:
            best_map3 = val_map3
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
            print(f"   ★ New best MAP@3: {best_map3:.4f}")

    # Load best checkpoint for predictions
    if best_state: model.load_state_dict(best_state)
    model.eval()

    # OOF predictions
    oof_logs=[]
    with torch.no_grad():
        for batch in vl_ld:
            batch.pop("labels",None)
            batch={k:v.to(DEVICE) for k,v in batch.items()}
            oof_logs.append(model(**batch).logits.float().cpu())
    oof_probs_fold = torch.softmax(torch.cat(oof_logs),-1).numpy()
    deberta_oof[vl_idx] = oof_probs_fold

    fold_map3 = mapk(LA[vl_idx].tolist(), top3(oof_probs_fold))
    print(f"  ► Fold {fold+1} final MAP@3={fold_map3:.4f}")
    wb({f"deberta/fold{fold+1}_final_map3": fold_map3})

    # Test predictions
    te_logs=[]
    with torch.no_grad():
        for batch in te_ld:
            batch.pop("labels",None)
            batch={k:v.to(DEVICE) for k,v in batch.items()}
            te_logs.append(model(**batch).logits.float().cpu())
    deberta_test_folds[:,:,fold] = torch.softmax(torch.cat(te_logs),-1).numpy()

    del model; gc.collect()
    try:
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    except: pass

deberta_test = deberta_test_folds.mean(2)
deb_cv = mapk(LA.tolist(), top3(deberta_oof))
print(f"\n🏆 DeBERTa CV MAP@3 = {deb_cv:.4f}")
np.save(OUT/"deberta_oof.npy", deberta_oof)
np.save(OUT/"deberta_test.npy", deberta_test)
wb({"deberta/cv_map3": deb_cv, "deberta/cv_top1": acc1(deberta_oof,LA)})

# ══════════════════════════════════════════════════════════════════════════════
# Ensemble: DeBERTa + Embeddings + Tree Models
# ══════════════════════════════════════════════════════════════════════════════
print("\n📊 Building ensemble...")
TP={}; TEP={}
def add(n,tr_s,te_s): TP[n]=norm(tr_s); TEP[n]=norm(te_s)

# DeBERTa (strongest model for generalization)
add("deberta", deberta_oof, deberta_test)

# Embedding similarity scores (good generalizers)
add("st_bge",    st_tr_bg, st_te_bg)
add("st_e5",     st_tr_e5, st_te_e5)
add("st_mpnet",  st_tr_mp, st_te_mp)
add("st_minilm", st_tr_ml, st_te_ml)
add("ce_minilm", ce_tr_ml, ce_te_ml)
add("ce_bge_r",  ce_tr_bg, ce_te_bg)

# Averaged embedding
EMB_TR = np.mean([norm(x) for x in [st_tr_bg,st_tr_e5,st_tr_mp,st_tr_ml,ce_tr_ml,ce_tr_bg]],0)
EMB_TE = np.mean([norm(x) for x in [st_te_bg,st_te_e5,st_te_mp,st_te_ml,ce_te_ml,ce_te_bg]],0)
add("emb_avg", EMB_TR, EMB_TE)

# Tree models (may overfit but useful in small weight)
for tag,oname,tname in [("lgbm_v2","oof_lgbm2.npy","test_lgbm2.npy"),
                          ("xgb_v2", "oof_xgb2.npy", "test_xgb2.npy"),
                          ("cat_v2", "oof_cat2.npy",  "test_cat2.npy")]:
    oo=npy(oname,(NT,5)); tt=npy(tname,(NTE,5))
    if oo.max()>0: add(tag,oo,tt)

# Individual model scores
print("\n" + "="*55)
print("  INDIVIDUAL MODEL MAP@3 (OOF)")
print("="*55)
ind={}
for nm,pr in TP.items():
    s=mapk(LA.tolist(),top3(pr)); ind[nm]=s
    print(f"  {nm:<26} {s:.4f}")
wb({f"individual/{nm}":s for nm,s in ind.items()})
print("="*55)

# ── Optuna ensemble optimisation ──────────────────────────────────────────────
print("\n🔍 Optuna 500 trials...")
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
MN=list(TP.keys())

def wens(w,d):
    wa=np.array(w); wa/=(wa.sum()+1e-9)
    return sum(wa[i]*d[MN[i]] for i in range(len(MN)))

def obj(trial):
    w=[trial.suggest_float(f"w_{n}",0.,1.) for n in MN]
    s=mapk(LA.tolist(),top3(wens(w,TP)))
    return s

study=optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED,n_startup_trials=50))
study.optimize(obj,n_trials=500,show_progress_bar=True)

BW=np.array([study.best_trial.params[f"w_{n}"] for n in MN])
BW/=(BW.sum()+1e-9)
print(f"\n🏆 Best ensemble OOF MAP@3 = {study.best_trial.value:.4f}")
for n,w in sorted(zip(MN,BW),key=lambda x:-x[1]):
    if w > 0.005:
        print(f"  {n:<28} {w:.4f}")
wb({"optuna/best_map3": study.best_trial.value,
    **{f"weights/{n}":float(w) for n,w in zip(MN,BW)}})

# ── Temperature scaling ──────────────────────────────────────────────────────
FT=wens(BW,TP); FTE=wens(BW,TEP)
def sft(sc,T): e=np.exp(sc/T); return e/(e.sum(1,keepdims=True)+1e-9)

tsc={}
for T in [0.1,0.3,0.5,0.7,1.0,1.5,2.0]:
    s=mapk(LA.tolist(),top3(sft(FT,T))); tsc[T]=s
BEST_T=max(tsc,key=tsc.get)
print(f"\n🌡️  Best T={BEST_T}  MAP@3={tsc[BEST_T]:.4f}")

FTE_SC=sft(FTE,BEST_T)

# ── Generate submission with CORRECT Kaggle column format ─────────────────────
print("\n📄 Generating submission.csv...")
preds=[" ".join([I2L[i] for i in np.argsort(-r)[:3]]) for r in FTE_SC]

# Use EXACT column names matching sample_submission
sub=pd.DataFrame({"id":te["id"].values, "prediction":preds})
sub.to_csv(BASE/"submission.csv",index=False)
print(f"\n🎉 submission.csv saved → {BASE/'submission.csv'}")
print(sub.head(10).to_string(index=False))

fm3=mapk(LA.tolist(),top3(sft(FT,BEST_T)))
print(f"\n✅ FINAL METRICS:")
print(f"   OOF MAP@3       = {fm3:.4f}")
print(f"   DeBERTa CV MAP@3= {deb_cv:.4f}")
print(f"   Temperature      = {BEST_T}")

if USE_WANDB:
    wandb.summary.update({"final_map3":fm3, "deberta_cv_map3":deb_cv, "best_T":BEST_T})
    art=wandb.Artifact("submission_deberta_boost",type="predictions")
    art.add_file(str(BASE/"submission.csv"))
    wandb.log_artifact(art)
    wandb.finish(); print("✅ W&B done!")

print(f"\n{'='*55}")
print(f"  FINAL OOF MAP@3 = {fm3:.4f}")
print(f"  DeBERTa included = True")
print(f"  submission.csv   → {BASE/'submission.csv'}")
print(f"{'='*55}")

# Smooth: Using label smoothing factor to regularize model logits