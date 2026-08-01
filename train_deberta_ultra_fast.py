#!/usr/bin/env python3
import os, sys, re, gc, time, warnings, random, json
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Using Apple MPS (M1/M2 GPU)")
else:
    DEVICE = torch.device("cpu")
    print("ℹ️ Using CPU")

torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/shobhitagnihotri/Downloads/DL GEN AI")
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OPTION_COLS = ["A", "B", "C", "D", "E"]
L2I = {c: i for i, c in enumerate(OPTION_COLS)}
I2L = {i: c for c, i in L2I.items()}

# ── Load data ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(BASE_DIR / "train (1).csv")
train_df.columns = [c.strip() for c in train_df.columns]
test_df = pd.read_csv(BASE_DIR / "test (1).csv")
test_df.columns = [c.strip() for c in test_df.columns]
train_df["label"] = train_df["answer"].map(L2I)
LA = train_df["label"].values

print(f"Loaded Train: {len(train_df)} | Test: {len(test_df)}")

# ── Metric functions ──────────────────────────────────────────────────────────
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

def scores_to_top3(scores):
    return [np.argsort(-row)[:3].tolist() for row in scores]

# ── Dataset Definition ────────────────────────────────────────────────────────
class MCQDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row["prompt"])
        encodings = []
        for opt in OPTION_COLS:
            enc = self.tok(
                prompt, str(row[opt]),
                max_length=self.max_len,
                truncation=True, padding="max_length",
                return_tensors="pt"
            )
            encodings.append({k: v.squeeze(0) for k, v in enc.items()})

        item = {
            "input_ids": torch.stack([e["input_ids"] for e in encodings]),
            "attention_mask": torch.stack([e["attention_mask"] for e in encodings]),
        }
        if "token_type_ids" in encodings[0]:
            item["token_type_ids"] = torch.stack([e["token_type_ids"] for e in encodings])

        if self.has_labels:
            item["labels"] = torch.tensor(L2I[str(row["answer"])], dtype=torch.long)
        return item

# ── Label smoothing loss ──────────────────────────────────────────────────────
def smooth_ce(logits, labels, eps=0.1):
    lp = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(lp, labels, reduction="mean")
    smooth = -lp.mean(dim=-1).mean()
    return (1.0 - eps) * nll + eps * smooth

# ── Hyperparameters ──────────────────────────────────────────────────────────
MODEL_ID = "microsoft/deberta-v3-small"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LR = 2e-5
N_SPLITS = 3

print(f"\nTraining DeBERTa model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(train_df), 5))
test_probs_folds = np.zeros((len(test_df), 5, N_SPLITS))

t_start = time.time()

for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, LA)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
    tr_ds = MCQDataset(train_df.iloc[tr_idx], tokenizer, MAX_LEN)
    vl_ds = MCQDataset(train_df.iloc[vl_idx], tokenizer, MAX_LEN)
    te_ds = MCQDataset(test_df, tokenizer, MAX_LEN, has_labels=False)

    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    vl_ld = DataLoader(vl_ds, batch_size=BATCH_SIZE*2, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE*2, shuffle=False)



    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(tr_ld) * EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * 0.1), total_steps)

    best_map3 = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(tr_ld):
            labels = batch.pop("labels").to(DEVICE)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            opt.zero_grad()
            out = model(**batch)
            loss = smooth_ce(out.logits, labels)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            
            total_loss += loss.item()

        # Validation
        model.eval()
        vl_logits, vl_labels = [], []
        with torch.no_grad():
            for batch in vl_ld:
                labs = batch.pop("labels").to(DEVICE)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                out = model(**batch)
                vl_logits.append(out.logits.float().cpu())
                vl_labels.append(labs.cpu())

        vl_logits = torch.cat(vl_logits).numpy()
        vl_labels = torch.cat(vl_labels).numpy()
        vl_probs = torch.softmax(torch.tensor(vl_logits), dim=-1).numpy()
        
        val_map3 = mapk(vl_labels.tolist(), scores_to_top3(vl_probs))
        print(f"  Epoch {epoch+1} | loss: {total_loss/len(tr_ld):.4f} | val MAP@3: {val_map3:.4f}")

        if val_map3 > best_map3:
            best_map3 = val_map3
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best checkpoint
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Get OOF predictions
    oof_logits = []
    with torch.no_grad():
        for batch in vl_ld:
            batch.pop("labels", None)
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            oof_logits.append(out.logits.float().cpu())
    oof_probs[vl_idx] = torch.softmax(torch.cat(oof_logits), dim=-1).numpy()

    # Get Test predictions
    te_logits = []
    with torch.no_grad():
        for batch in te_ld:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            te_logits.append(out.logits.float().cpu())
    test_probs_folds[:, :, fold] = torch.softmax(torch.cat(te_logits), dim=-1).numpy()

    del model; gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

test_probs = test_probs_folds.mean(axis=2)
cv_map3 = mapk(LA.tolist(), scores_to_top3(oof_probs))
print(f"\n🏆 Final DeBERTa CV MAP@3: {cv_map3:.4f}")
print(f"Training completed in {time.time() - t_start:.1f}s")

# Save outputs
np.save(OUT_DIR / "deberta_oof.npy", oof_probs)
np.save(OUT_DIR / "deberta_test.npy", test_probs)
print("Saved predictions successfully.")
