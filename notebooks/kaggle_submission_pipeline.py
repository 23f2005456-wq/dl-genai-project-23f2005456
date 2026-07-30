# --- CELL --- 
# Install dependencies if running on Kaggle
import sys
import os

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    print("Running on Kaggle GPU environment ...")
else:
    print("Running in Local environment ...")

# Authenticate Weights & Biases
import wandb
try:
    if IS_KAGGLE:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        os.environ["WANDB_API_KEY"] = user_secrets.get_secret("WANDB_API_KEY")
        wandb.login()
    else:
        # Local login check
        wandb.login()
    USE_WANDB = True
except Exception as e:
    print(f"⚠️ W&B credentials not found or login skipped: {e}")
    print("Running without experiment tracking.")
    USE_WANDB = False

# --- CELL --- 
import gc
import re
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import optuna

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
set_seed()

# Resolve directories
if IS_KAGGLE:
    BASE_DIR = Path("/kaggle/input/llm-science-exam")
    if not BASE_DIR.exists():
        # Fallback for playground/custom dataset mapping
        BASE_DIR = Path("/kaggle/input")
    OUT_DIR = Path("/kaggle/working/outputs")
    MODEL_DIR = Path("/kaggle/working/models")
else:
    BASE_DIR = Path("..").resolve()
    OUT_DIR = BASE_DIR / "outputs"
    MODEL_DIR = BASE_DIR / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_TO_IDX = {k: i for i, k in enumerate(OPTION_COLS)}
IDX_TO_LABEL = {i: k for k, i in LABEL_TO_IDX.items()}

# --- CELL --- 
# Load files
train_path = list(BASE_DIR.glob("**/train*.csv"))
test_path = list(BASE_DIR.glob("**/test*.csv"))

if train_path:
    train_df = pd.read_csv(train_path[0])
else:
    # Local default path
    train_df = pd.read_csv(BASE_DIR / "train (1).csv")

if test_path:
    test_df = pd.read_csv(test_path[0])
else:
    test_df = pd.read_csv(BASE_DIR / "test (1).csv")

train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns  = [c.strip() for c in test_df.columns]
train_df["label"] = train_df["answer"].map(LABEL_TO_IDX)
label_arr = train_df["label"].values


# Local execution speed-up modifications
train_df = train_df.head(50).copy()
test_df = test_df.head(10).copy()
label_arr = train_df['label'].values
print(f"Train set: {train_df.shape} | Test set: {test_df.shape}")

# --- CELL --- 
import re
from typing import List, Tuple, Dict, Any

def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing and punctuation removal."""
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split()

def ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """Compute n-gram Jaccard overlap between two texts."""
    def get_ngrams(tokens, n):
        return set(zip(*[tokens[i:] for i in range(n)]))
    t1, t2 = tokenize(text1), tokenize(text2)
    ng1, ng2 = get_ngrams(t1, n), get_ngrams(t2, n)
    if not ng1 and not ng2: return 0.0
    return len(ng1 & ng2) / (len(ng1 | ng2) + 1e-9)

def keyword_overlap(text1: str, text2: str) -> float:
    """Keyword (non-stopword) Jaccard overlap."""
    t1 = set(tokenize(text1))
    t2 = set(tokenize(text2))
    if not t1 and not t2: return 0.0
    return len(t1 & t2) / (len(t1 | t2) + 1e-9)

def length_features(prompt: str, option: str) -> Dict[str, float]:
    """Length-based features for a prompt-option pair."""
    p_chars, o_chars = len(prompt), len(option)
    p_words = len(prompt.split())
    o_words = len(option.split())
    return {
        "prompt_len_chars":  float(p_chars),
        "option_len_chars":  float(o_chars),
        "option_len_words":  float(o_words),
        "char_ratio":        float(o_chars / (p_chars + 1)),
        "word_ratio":        float(o_words / (p_words + 1)),
        "option_is_long":    float(int(o_chars > 200)),
        "option_is_short":   float(int(o_chars < 30)),
    }

def build_feature_row(prompt: str, option: str, option_idx: int, all_options: List[str]) -> Dict[str, float]:
    """Build a feature vector for one (prompt, option) pair."""
    feats = {}
    feats.update(length_features(prompt, option))
    
    # Overlap features
    feats["unigram_overlap"] = keyword_overlap(prompt, option)
    feats["bigram_overlap"]  = ngram_overlap(prompt, option, n=2)
    feats["trigram_overlap"] = ngram_overlap(prompt, option, n=3)
    
    # Positional bias feature
    feats["option_position"] = float(option_idx)
    
    # Option relative length rank
    option_lens = [len(o) for o in all_options]
    sorted_lens = sorted(option_lens, reverse=True)
    feats["length_rank"] = float(sorted_lens.index(len(option)))
    
    # Structural details
    feats["starts_with_num"] = float(int(bool(re.match(r'^\d', option))))
    feats["starts_with_cap"] = float(int(option[0].isupper()) if option else 0)
    feats["num_numbers"] = float(len(re.findall(r'\b\d+\.?\d*\b', option)))
    
    neg_words = {"not", "no", "never", "neither", "nor", "without", "cannot"}
    feats["has_negation"] = float(int(bool(neg_words & set(tokenize(option)))))
    
    return feats

def build_features(df: pd.DataFrame, tfidf_vectorizer=None, fit=False) -> Tuple[np.ndarray, Any]:
    print("🔧 Building features ...")
    all_texts = [str(row["prompt"]) + " " + str(row[opt]) for _, row in df.iterrows() for opt in OPTION_COLS]
    
    if fit or tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=40_000,
            sublinear_tf=True, strip_accents='unicode',
            analyzer='word', min_df=2, max_df=0.95
        )
        tfidf_vectorizer.fit(all_texts)
        
    # Vectorized TF-IDF Transform
    prompts = df["prompt"].astype(str).tolist()
    prompt_vecs = tfidf_vectorizer.transform(prompts)
    
    option_vecs = {}
    for opt in OPTION_COLS:
        options = df[opt].astype(str).tolist()
        option_vecs[opt] = tfidf_vectorizer.transform(options)
        
    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        prompt = str(row["prompt"])
        all_options = [str(row[c]) for c in OPTION_COLS]
        
        if HAS_BM25:
            bm25_model = BM25Okapi([tokenize(o) for o in all_options])
            bm25_scores = bm25_model.get_scores(tokenize(prompt))
        else:
            bm25_scores = np.zeros(5)
            
        p_vec = prompt_vecs[idx]
        for i, opt in enumerate(OPTION_COLS):
            option = str(row[opt])
            o_vec = option_vecs[opt][idx]
            tfidf_cos = float(p_vec.dot(o_vec.T).toarray()[0, 0])
            
            feat = build_feature_row(prompt, option, i, all_options)
            feat["tfidf_cosine"] = tfidf_cos
            feat["bm25_score"]   = float(bm25_scores[i])
            feat["bm25_rank"]    = 5 - int(np.sum(bm25_scores < bm25_scores[i]))
            rows.append(feat)
            
    X = pd.DataFrame(rows).fillna(0).values
    print(f"   Feature matrix: {X.shape}")
    return X, tfidf_vectorizer


# --- CELL --- 
class ClassicalEnsemble:
    def __init__(self, n_splits=2):
        self.n_splits = n_splits
        
    def _make_models(self):
        return {
            "lgbm": lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1
            ),
            "xgb": xgb.XGBClassifier(
                n_estimators=600, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss', early_stopping_rounds=50
            ),
            "cat": cb.CatBoostClassifier(
                iterations=600, learning_rate=0.03, depth=6,
                l2_leaf_reg=5.0, subsample=0.8, random_seed=SEED, verbose=0
            )
        }

    def fit(self, X, y, X_test, n_samples_train):
        label_per_q = y.reshape(n_samples_train, 5).argmax(axis=1)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=SEED)
        
        oof_predictions = {}
        test_predictions = {}
        
        for model_name, model in self._make_models().items():
            print(f"  Training: {model_name.upper()} ...")
            oof_prob = np.zeros((n_samples_train, 5))
            test_prob_folds = np.zeros((X_test.shape[0] // 5, 5, self.n_splits))
            
            for fold, (train_q, val_q) in enumerate(skf.split(np.zeros(n_samples_train), label_per_q)):
                train_opt = np.concatenate([np.arange(qi*5, qi*5+5) for qi in train_q])
                val_opt   = np.concatenate([np.arange(qi*5, qi*5+5) for qi in val_q])
                
                X_tr, y_tr = X[train_opt], y[train_opt]
                X_vl, y_vl = X[val_opt], y[val_opt]
                
                if model_name == "lgbm":
                    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=[lgb.early_stopping(50, verbose=False)])
                elif model_name == "xgb":
                    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                elif model_name == "cat":
                    model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), early_stopping_rounds=50, verbose=False)
                else:
                    model.fit(X_tr, y_tr)
                    
                # Predict probabilities of positive label
                preds_vl = model.predict_proba(X_vl)[:, 1]
                oof_prob[val_q] = preds_vl.reshape(-1, 5)
                if USE_WANDB and wandb.run is not None and model_name == 'xgb':
                    xgb_fold_map3 = mapk(y_vl.reshape(-1, 5).argmax(axis=1).tolist(), scores_to_top3(preds_vl.reshape(-1, 5)))
                    wandb.log({
                        'xgb/fold_map3': xgb_fold_map3,
                        'xgb/fold': fold
                    })
                
                preds_te = model.predict_proba(X_test)[:, 1]
                test_prob_folds[:, :, fold] = preds_te.reshape(-1, 5)
                
            oof_predictions[model_name] = oof_prob
            test_predictions[model_name] = test_prob_folds.mean(axis=2)
            
        return oof_predictions, test_predictions
# =============================================================================
# 🧠 MODEL BUILT FROM SCRATCH (Custom MLP in PyTorch)
# =============================================================================
import torch.nn as nn
import torch.optim as optim

class ModelFromScratch(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_model_from_scratch(X, y, X_test, n_samples_train, n_splits=2):
    print("⚡ Training Model From Scratch (Custom MLP) ...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    label_per_q = y.reshape(n_samples_train, 5).argmax(axis=1)
    
    oof_prob = np.zeros((n_samples_train, 5))
    test_prob_folds = np.zeros((X_test.shape[0] // 5, 5, n_splits))
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    for fold, (train_q, val_q) in enumerate(skf.split(np.zeros(n_samples_train), label_per_q)):
        train_opt = np.concatenate([np.arange(qi*5, qi*5+5) for qi in train_q])
        val_opt   = np.concatenate([np.arange(qi*5, qi*5+5) for qi in val_q])
        
        X_tr, y_tr = X_scaled[train_opt], y[train_opt]
        X_vl, y_vl = X_scaled[val_opt], y[val_opt]
        
        from torch.utils.data import TensorDataset, DataLoader
        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        
        model = ModelFromScratch(X.shape[1]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        
        model.train()
        for epoch in range(25):
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = model(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                if USE_WANDB and wandb.run is not None:
                    wandb.log({
                        'scratch/loss': loss.item(),
                        'scratch/epoch': fold * 25 + epoch
                    })
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            preds_vl = torch.sigmoid(model(torch.FloatTensor(X_vl).to(DEVICE))).cpu().numpy().squeeze()
            oof_prob[val_q] = preds_vl.reshape(-1, 5)
            if USE_WANDB and wandb.run is not None and model_name == 'xgb':
                xgb_fold_map3 = mapk(y_vl.reshape(-1, 5).argmax(axis=1).tolist(), scores_to_top3(preds_vl.reshape(-1, 5)))
                wandb.log({
                    'xgb/fold_map3': xgb_fold_map3,
                    'xgb/fold': fold
                })
            
            preds_te = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(DEVICE))).cpu().numpy().squeeze()
            test_prob_folds[:, :, fold] = preds_te.reshape(-1, 5)
            
    test_prob = test_prob_folds.mean(axis=2)
    return oof_prob, test_prob


# --- CELL --- 
# Automatically choose DeBERTa configuration based on device capabilities
import torch

DEVICE = torch.device('cpu')
USE_AMP = DEVICE.type == 'cuda'

if DEVICE.type == 'cuda':
    DEBERTA_MODEL_ID = '/kaggle/input/models/tailen/detect-ai-text-deberta-v3-large/pytorch/large/1'
    DEBERTA_BATCH_SIZE = 2
    DEBERTA_ACCUM_STEPS = 16
else:
    DEBERTA_MODEL_ID = 'microsoft/deberta-v3-base'
    DEBERTA_BATCH_SIZE = 1
    DEBERTA_ACCUM_STEPS = 32

DEBERTA_MAX_LEN = 64
DEBERTA_EPOCHS = 1
DEBERTA_LR = 1.5e-5
LABEL_SMOOTHING = 0.1

print(f"Using DeBERTa model: {DEBERTA_MODEL_ID} on device: {DEVICE}")

# --- CELL --- 
from transformers import AutoTokenizer, AutoModelForMultipleChoice, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

class MCQDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = str(row["prompt"])
        options = [str(row[o]) for o in OPTION_COLS]
        
        # Format text pairs: (prompt, option)
        first_sentences = [prompt] * 5
        second_sentences = options
        
        inputs = self.tokenizer(
            first_sentences, second_sentences,
            truncation=True, max_length=self.max_len, padding='max_length',
            return_tensors="pt"
        )
        
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        if self.has_labels:
            item["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return item

def train_deberta(train_df, test_df, n_splits=2):
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_ID)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    oof_probs = np.zeros((len(train_df), 5))
    test_probs_folds = np.zeros((len(test_df), 5, n_splits))
    
    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, train_df["label"])):
        print(f"  Training DeBERTa Fold {fold+1}/{n_splits} ...")
        tr_ds = MCQDataset(train_df.iloc[tr_idx], tokenizer, DEBERTA_MAX_LEN)
        vl_ds = MCQDataset(train_df.iloc[vl_idx], tokenizer, DEBERTA_MAX_LEN)
        te_ds = MCQDataset(test_df, tokenizer, DEBERTA_MAX_LEN, has_labels=False)
        
        tr_loader = DataLoader(tr_ds, batch_size=DEBERTA_BATCH_SIZE, shuffle=True)
        vl_loader = DataLoader(vl_ds, batch_size=DEBERTA_BATCH_SIZE*2, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=DEBERTA_BATCH_SIZE*2, shuffle=False)
        
        model = AutoModelForMultipleChoice.from_pretrained(DEBERTA_MODEL_ID).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=DEBERTA_LR, weight_decay=0.01)
        
        total_steps = len(tr_loader) // DEBERTA_ACCUM_STEPS * DEBERTA_EPOCHS
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
        
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
        
        # Training loop
        for epoch in range(DEBERTA_EPOCHS):
            model.train()
            optimizer.zero_grad()
            for step, batch in enumerate(tr_loader):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                labels = batch.pop("labels")
                
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(**batch)
                    loss = loss_fn(outputs.logits, labels) / DEBERTA_ACCUM_STEPS
                    
                scaler.scale(loss).backward()
                if (step + 1) % DEBERTA_ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
        # Validate fold
        model.eval()
        val_scores = []
        with torch.no_grad():
            for batch in vl_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                batch.pop("labels")
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                val_scores.append(probs)
        oof_probs[vl_idx] = np.concatenate(val_scores, axis=0)
        
        # Predict test
        test_scores = []
        with torch.no_grad():
            for batch in te_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                logits = model(**batch).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                test_scores.append(probs)
        test_probs_folds[:, :, fold] = np.concatenate(test_scores, axis=0)
        
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return oof_probs, test_probs_folds.mean(axis=2)

# --- CELL --- 
import os
from sklearn.metrics import accuracy_score, f1_score
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'mcq-competition-ensemble')

# 1. Feature Engineering
X_train_flat, tfidf_vec = build_features(train_df, fit=True)
X_test_flat, _          = build_features(test_df, tfidf_vectorizer=tfidf_vec)
y_train_flat            = np.zeros(len(train_df) * 5)
for i, l in enumerate(train_df["label"].values):
    y_train_flat[i*5 + l] = 1

# --- Run 1: Model From Scratch (Custom MLP) ---
if USE_WANDB:
    run1 = wandb.init(
        project=WANDB_PROJECT,
        name="model-from-scratch",
        config={
            "model_type": "scratch_mlp",
            "hidden_dim": 64,
            "dropout": 0.2,
            "epochs": 25,
            "learning_rate": 0.005,
            "optimizer": "Adam"
        },
        reinit=True
    )

oof_scratch, test_scratch = train_model_from_scratch(X_train_flat, y_train_flat, X_test_flat, len(train_df))
all_oof = {"scratch": oof_scratch}
all_test = {"scratch": test_scratch}

scratch_map3 = mapk(train_df["label"].values.tolist(), scores_to_top3(oof_scratch))
scratch_preds = oof_scratch.argmax(axis=1)
scratch_acc = accuracy_score(train_df["label"].values, scratch_preds)
scratch_f1 = f1_score(train_df["label"].values, scratch_preds, average="macro")

if USE_WANDB:
    wandb.log({
        "val/map3": scratch_map3,
        "val/accuracy": scratch_acc,
        "val/f1_macro": scratch_f1
    })
    run1.finish()

# --- Run 2: Classical XGBoost Model (Model of choice) ---
if USE_WANDB:
    run2 = wandb.init(
        project=WANDB_PROJECT,
        name="xgb-model-choice",
        config={
            "model_type": "xgboost",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        reinit=True
    )

print("⚡ Running Classical Ensemble ...")
ens = ClassicalEnsemble(n_splits=2)
class_oof, class_test = ens.fit(X_train_flat, y_train_flat, X_test_flat, len(train_df))
all_oof.update(class_oof)
all_test.update(class_test)

xgb_oof = all_oof["xgb"]
xgb_map3 = mapk(train_df["label"].values.tolist(), scores_to_top3(xgb_oof))
xgb_preds = xgb_oof.argmax(axis=1)
xgb_acc = accuracy_score(train_df["label"].values, xgb_preds)
xgb_f1 = f1_score(train_df["label"].values, xgb_preds, average="macro")

if USE_WANDB:
    wandb.log({
        "val/map3": xgb_map3,
        "val/accuracy": xgb_acc,
        "val/f1_macro": xgb_f1
    })
    run2.finish()

# --- Run 3: Pretrained DeBERTa Model (Fine-tuned) ---
if USE_WANDB:
    run3 = wandb.init(
        project=WANDB_PROJECT,
        name="deberta-pretrained",
        config={
            "model_type": "deberta_v3_large",
            "learning_rate": DEBERTA_LR,
            "batch_size": DEBERTA_BATCH_SIZE,
            "epochs": DEBERTA_EPOCHS,
            "max_length": DEBERTA_MAX_LEN
        },
        reinit=True
    )

print("⚡ Running DeBERTa Model ...")
try:
    deb_oof, deb_test = train_deberta(train_df, test_df, n_splits=2)
    all_oof["deberta"] = deb_oof
    all_test["deberta"] = deb_test
    
    deb_map3 = mapk(train_df["label"].values.tolist(), scores_to_top3(deb_oof))
    deb_preds = deb_oof.argmax(axis=1)
    deb_acc = accuracy_score(train_df["label"].values, deb_preds)
    deb_f1 = f1_score(train_df["label"].values, deb_preds, average="macro")
    
    if USE_WANDB:
        wandb.log({
            "val/map3": deb_map3,
            "val/accuracy": deb_acc,
            "val/f1_macro": deb_f1
        })
except Exception as e:
    print(f"⚠️ DeBERTa Training failed or skipped: {e}")

if USE_WANDB:
    run3.finish()


# --- CELL --- 
if USE_WANDB:
    wandb.init(
        project=WANDB_PROJECT,
        name="optuna-ensemble-optimization",
        config={
            "n_trials": 300,
            "models_ensembled": list(all_oof.keys())
        },
        reinit=True
    )

def apk(a: int, p: List[int], k: int = 3) -> float:
    if len(p) > k: p = p[:k]
    s, h = 0.0, 0
    for i, x in enumerate(p):
        if x == a and x not in p[:i]:
            h += 1
            s += h / (i + 1.0)
    return s

def mapk(a: List[int], p: List[List[int]], k: int = 3) -> float:
    return float(np.mean([apk(x, y, k) for x, y in zip(a, p)]))

def scores_to_top3(s):
    return [np.argsort(-r)[:3].tolist() for r in s]

def normalize(s):
    mn = s.min(axis=1, keepdims=True)
    mx = s.max(axis=1, keepdims=True)
    return (s - mn) / (mx - mn + 1e-9)

# Normalize predictions
for k in all_oof.keys():
    all_oof[k]  = normalize(all_oof[k])
    all_test[k] = normalize(all_test[k])

# smart ensemble optimization: Filter to non-duplicate prompts to avoid CV leak
mask_nodup = ~train_df["prompt"].duplicated(keep=False)
labels_nd  = label_arr[mask_nodup]
all_oof_nd = {k: v[mask_nodup] for k, v in all_oof.items()}
MODEL_NAMES = list(all_oof.keys())

print(f"Honest validation subset (no duplicates): {mask_nodup.sum()} samples.")

def objective_nd(trial):
    w = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in MODEL_NAMES])
    w = w / (w.sum() + 1e-9)
    ens_preds = sum(w[i] * all_oof_nd[MODEL_NAMES[i]] for i in range(len(MODEL_NAMES)))
    val_score = mapk(labels_nd.tolist(), scores_to_top3(ens_preds))
    
    if USE_WANDB:
        wandb.log({
            "trial/map3_no_dup": val_score,
            "trial/number": trial.number,
            **{f"trial_weight/{n}": w[idx] for idx, n in enumerate(MODEL_NAMES)}
        })
    return val_score

study = optuna.create_study(direction="maximize")
study.optimize(objective_nd, n_trials=300)

bp = study.best_trial.params
norm_w = np.array([bp[f"w_{n}"] for n in MODEL_NAMES])
norm_w = norm_w / (norm_w.sum() + 1e-9)
best_val = study.best_trial.value

print(f"\n🏆 Optimized Ensemble CV MAP@3 (no-leak) = {best_val:.4f}")
for i, n in enumerate(MODEL_NAMES):
    print(f"  Model: {n:<15} | Weight: {norm_w[i]:.4f}")

# --- CELL --- 
final_test = sum(norm_w[i] * all_test[MODEL_NAMES[i]] for i in range(len(MODEL_NAMES)))
top3_preds = scores_to_top3(final_test)
pred_strs = [" ".join(IDX_TO_LABEL[idx] for idx in p) for p in top3_preds]

sub = pd.DataFrame({
    "id": test_df["id"].values if "id" in test_df.columns else test_df["ID"].values,
    "prediction": pred_strs
})

sub.to_csv("submission.csv", index=False)
print("\n✅ submission.csv generated and saved successfully!")
print(sub.head(10).to_string(index=False))

if USE_WANDB:
    wandb.log({
        "final/ensemble_map3": best_val,
        **{f"final_weight/{n}": norm_w[idx] for idx, n in enumerate(MODEL_NAMES)}
    })
    wandb.finish()

