import json

notebook_path = 'dl-23f2005456-notebook-t22026.ipynb'

cells = [
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## ⚙️ 1. Setup Environment & Credentials']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# Install dependencies if running on Kaggle
import sys
import os

IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

if IS_KAGGLE:
    print("Running on Kaggle GPU environment ...")
    !pip install -q optuna wandb rank-bm25
else:
    print("Running in Local environment ...")

# Authenticate Weights & Biases
import wandb
WANDB_PROJECT = "mcq-competition-ensemble"
WANDB_RUN_NAME = "full-model-ensemble-run"

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
    print("✅ Weights & Biases initialized successfully!")
except Exception as e:
    print(f"⚠️ W&B credentials not found or login skipped: {e}")
    print("Running without experiment tracking.")
    USE_WANDB = False'''
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 📦 2. Libraries & Paths']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''import gc
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from sklearn.preprocessing import StandardScaler
import optuna

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Resolve directories
if IS_KAGGLE:
    BASE_DIR = Path("/kaggle/input/llm-science-exam")
    if not BASE_DIR.exists():
        BASE_DIR = Path("/kaggle/input")
    OUT_DIR = Path("/kaggle/working/outputs")
    MODEL_DIR = Path("/kaggle/working/models")
else:
    BASE_DIR = Path(".").resolve()
    OUT_DIR = BASE_DIR / "outputs"
    MODEL_DIR = BASE_DIR / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OPTION_COLS  = ["A", "B", "C", "D", "E"]
LABEL_TO_IDX = {k: i for i, k in enumerate(OPTION_COLS)}
IDX_TO_LABEL = {i: k for k, i in LABEL_TO_IDX.items()}'''
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 📄 3. Load Datasets']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# Load files
train_path = list(BASE_DIR.glob("**/train*.csv"))
test_path = list(BASE_DIR.glob("**/test*.csv"))

if train_path:
    train_df = pd.read_csv(train_path[0])
else:
    train_df = pd.read_csv(BASE_DIR / "train (1).csv")

if test_path:
    test_df = pd.read_csv(test_path[0])
else:
    test_df = pd.read_csv(BASE_DIR / "test (1).csv")

train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns  = [c.strip() for c in test_df.columns]
train_df["label"] = train_df["answer"].map(LABEL_TO_IDX)
label_arr = train_df["label"].values

print(f"Train set: {train_df.shape} | Test set: {test_df.shape}")'''
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 🛠️ 4. Feature Engineering Functions']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''def tokenize(text: str) -> List[str]:
    return re.findall(r'\\w+', text.lower())

def ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    def get_ngrams(tokens, n):
        return set(zip(*[tokens[i:] for i in range(n)]))
    toks1, toks2 = tokenize(text1), tokenize(text2)
    if len(toks1) < n or len(toks2) < n: return 0.0
    ng1, ng2 = get_ngrams(toks1, n), get_ngrams(toks2, n)
    return len(ng1 & ng2) / (len(ng1 | ng2) + 1e-9)

def keyword_overlap(text1: str, text2: str) -> float:
    stops = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at'}
    toks1 = set(tokenize(text1)) - stops
    toks2 = set(tokenize(text2)) - stops
    if not toks1 or not toks2: return 0.0
    return len(toks1 & toks2) / (len(toks1 | toks2) + 1e-9)

def length_features(prompt: str, option: str) -> Dict[str, float]:
    l_p, l_o = len(prompt), len(option)
    w_p, w_o = len(prompt.split()), len(option.split())
    return {
        "char_len_opt": float(l_o),
        "word_len_opt": float(w_o),
        "char_ratio": l_o / (l_p + 1e-9),
        "word_ratio": w_o / (w_p + 1e-9)
    }

def build_feature_row(prompt: str, option: str, option_idx: int, all_options: List[str]) -> Dict[str, float]:
    feats = {}
    feats.update(length_features(prompt, option))
    feats["unigram_overlap"] = keyword_overlap(prompt, option)
    feats["bigram_overlap"]  = ngram_overlap(prompt, option, n=2)
    feats["trigram_overlap"] = ngram_overlap(prompt, option, n=3)
    feats["option_position"] = option_idx
    
    opt_lens = [len(o) for o in all_options]
    sorted_lens = sorted(opt_lens, reverse=True)
    feats["length_rank"] = sorted_lens.index(len(option))
    feats["len_vs_mean"]  = len(option) / (np.mean(opt_lens) + 1e-9)
    
    feats["starts_with_num"] = int(bool(re.match(r'^\\d', option)))
    feats["starts_with_cap"] = int(option[0].isupper()) if option else 0
    feats["num_numbers"] = len(re.findall(r'\\b\\d+\\.?\\d*\\b', option))

    neg_words = {"not", "no", "never", "neither", "nor", "without", "cannot"}
    feats["has_negation"] = int(bool(neg_words & set(tokenize(option))))
    
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
        
    rows = []
    for _, row in df.iterrows():
        prompt = str(row["prompt"])
        all_options = [str(row[c]) for c in OPTION_COLS]
        
        if HAS_BM25:
            bm25_model = BM25Okapi([tokenize(o) for o in all_options])
            bm25_scores = bm25_model.get_scores(tokenize(prompt))
        else:
            bm25_scores = np.zeros(5)
            
        prompt_vec = tfidf_vectorizer.transform([prompt])
        for i, opt in enumerate(OPTION_COLS):
            option = str(row[opt])
            opt_vec = tfidf_vectorizer.transform([option])
            tfidf_cos = float(cosine_similarity(prompt_vec, opt_vec)[0, 0])
            
            feat = build_feature_row(prompt, option, i, all_options)
            feat["tfidf_cosine"] = tfidf_cos
            feat["bm25_score"]   = float(bm25_scores[i])
            feat["bm25_rank"]    = 5 - int(np.sum(bm25_scores < bm25_scores[i]))
            rows.append(feat)
            
    X = pd.DataFrame(rows).fillna(0).values
    return X, tfidf_vectorizer'''
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 📊 5. Metrics & Weights & Biases Logger Utilities']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''def apk(a: int, p: list, k: int = 3) -> float:
    if len(p) > k: p = p[:k]
    s, h = 0.0, 0
    for i, x in enumerate(p):
        if x == a and x not in p[:i]:
            h += 1
            s += h / (i + 1.0)
    return s

def mapk(a: list, p: list, k: int = 3) -> float:
    return float(np.mean([apk(x, y, k) for x, y in zip(a, p)]))

def scores_to_top3(s):
    return [np.argsort(-r)[:3].tolist() for r in s]

def compute_metrics(y_true, y_prob_5col):
    """
    y_true: 1D array of correct option index (0..4) for N questions
    y_prob_5col: (N, 5) array of predicted probabilities/scores for options A..E
    """
    top1_preds = y_prob_5col.argmax(axis=1)
    top3_preds = scores_to_top3(y_prob_5col)
    
    y_true_list = y_true.tolist() if hasattr(y_true, "tolist") else list(y_true)
    map3_val = mapk(y_true_list, top3_preds, k=3)
    acc = accuracy_score(y_true, top1_preds)
    f1_mac = f1_score(y_true, top1_preds, average='macro', zero_division=0)
    f1_wgt = f1_score(y_true, top1_preds, average='weighted', zero_division=0)
    prec_mac = precision_score(y_true, top1_preds, average='macro', zero_division=0)
    rec_mac = recall_score(y_true, top1_preds, average='macro', zero_division=0)
    
    probs_norm = y_prob_5col / (y_prob_5col.sum(axis=1, keepdims=True) + 1e-9)
    probs_norm = np.clip(probs_norm, 1e-15, 1 - 1e-15)
    
    y_true_oh = np.zeros_like(probs_norm)
    y_true_oh[np.arange(len(y_true)), y_true] = 1
    ll = log_loss(y_true_oh, probs_norm)
    
    return {
        "map3": float(map3_val),
        "accuracy": float(acc),
        "f1_macro": float(f1_mac),
        "f1_weighted": float(f1_wgt),
        "precision_macro": float(prec_mac),
        "recall_macro": float(rec_mac),
        "log_loss": float(ll)
    }

def log_metrics_to_wandb(metrics_dict: dict, prefix: str):
    """
    Safely log a dictionary of metrics to W&B with a prefix.
    """
    if USE_WANDB and getattr(wandb, "run", None) is not None:
        log_payload = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
        try:
            wandb.log(log_payload)
        except Exception as e:
            print(f"Failed to log to WandB: {e}")'''
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 🌲 6. Classical Machine Learning Models & Custom PyTorch MLP']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''class ClassicalEnsemble:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        
    def _make_models(self):
        return {
            "lgbm": lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1
            ),
            "xgb": xgb.XGBClassifier(
                n_estimators=600, learning_rate=0.03, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric='logloss',
                early_stopping_rounds=50
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
            train_prob = np.zeros((n_samples_train, 5))
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
                    
                preds_tr = model.predict_proba(X_tr)[:, 1]
                preds_vl = model.predict_proba(X_vl)[:, 1]
                oof_prob[val_q] = preds_vl.reshape(-1, 5)
                train_prob[train_q] = preds_tr.reshape(-1, 5)
                
                preds_te = model.predict_proba(X_test)[:, 1]
                test_prob_folds[:, :, fold] = preds_te.reshape(-1, 5)
                
            oof_predictions[model_name] = oof_prob
            test_predictions[model_name] = test_prob_folds.mean(axis=2)
            
            tr_metrics = compute_metrics(label_per_q, train_prob)
            val_metrics = compute_metrics(label_per_q, oof_prob)
            
            print(f"    [{model_name.upper()}] Train MAP@3: {tr_metrics['map3']:.4f} | F1: {tr_metrics['f1_macro']:.4f}")
            print(f"    [{model_name.upper()}] Val OOF MAP@3: {val_metrics['map3']:.4f} | F1: {val_metrics['f1_macro']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")
            
            log_metrics_to_wandb(tr_metrics, prefix=f"{model_name}/train")
            log_metrics_to_wandb(val_metrics, prefix=f"{model_name}/val")
            
        return oof_predictions, test_predictions


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

def train_model_from_scratch(X, y, X_test, n_samples_train, n_splits=5):
    print("⚡ Training Model From Scratch (Custom MLP) ...")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    label_per_q = y.reshape(n_samples_train, 5).argmax(axis=1)
    
    oof_prob = np.zeros((n_samples_train, 5))
    train_prob = np.zeros((n_samples_train, 5))
    test_prob_folds = np.zeros((X_test.shape[0] // 5, 5, n_splits))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    for fold, (train_q, val_q) in enumerate(skf.split(np.zeros(n_samples_train), label_per_q)):
        train_opt = np.concatenate([np.arange(qi*5, qi*5+5) for qi in train_q])
        val_opt   = np.concatenate([np.arange(qi*5, qi*5+5) for qi in val_q])
        
        X_tr, y_tr = X_scaled[train_opt], y[train_opt]
        X_vl, y_vl = X_scaled[val_opt], y[val_opt]
        
        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        
        model = ModelFromScratch(X.shape[1]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
        
        model.train()
        for epoch in range(25):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                out = model(xb).squeeze()
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
            epoch_loss /= len(train_ds)
            
            if USE_WANDB and getattr(wandb, "run", None) is not None:
                try:
                    wandb.log({f"scratch/fold_{fold+1}_epoch_loss": epoch_loss})
                except Exception:
                    pass
                
        model.eval()
        with torch.no_grad():
            preds_tr = torch.sigmoid(model(torch.FloatTensor(X_tr).to(DEVICE))).cpu().numpy().squeeze()
            train_prob[train_q] = preds_tr.reshape(-1, 5)
            
            preds_vl = torch.sigmoid(model(torch.FloatTensor(X_vl).to(DEVICE))).cpu().numpy().squeeze()
            oof_prob[val_q] = preds_vl.reshape(-1, 5)
            
            preds_te = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(DEVICE))).cpu().numpy().squeeze()
            test_prob_folds[:, :, fold] = preds_te.reshape(-1, 5)
            
    tr_metrics = compute_metrics(label_per_q, train_prob)
    val_metrics = compute_metrics(label_per_q, oof_prob)
    
    print(f"    [SCRATCH MLP] Train MAP@3: {tr_metrics['map3']:.4f} | F1: {tr_metrics['f1_macro']:.4f}")
    print(f"    [SCRATCH MLP] Val OOF MAP@3: {val_metrics['map3']:.4f} | F1: {val_metrics['f1_macro']:.4f} | Accuracy: {val_metrics['accuracy']:.4f}")
    
    log_metrics_to_wandb(tr_metrics, prefix="scratch/train")
    log_metrics_to_wandb(val_metrics, prefix="scratch/val")
    
    test_prob = test_prob_folds.mean(axis=2)
    return oof_prob, test_prob'''
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 🧪 7. Train & Evaluate All Models']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# 1. Start W&B Run
if USE_WANDB:
    try:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "n_splits": 5,
                "seed": SEED,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "architecture": "Ensemble (LGBM + XGBoost + CatBoost + MLP)"
            },
            reinit=True
        )
        print("🚀 Started W&B Run:", wandb.run.name)
    except Exception as e:
        print("W&B Init Notice:", e)

# 2. Extract Features
X_train, tfidf_vec = build_features(train_df, fit=True)
X_test, _          = build_features(test_df, tfidf_vectorizer=tfidf_vec, fit=False)

n_samples_train = len(train_df)
binary_y = np.zeros((n_samples_train * 5,))
for i, lbl in enumerate(train_df["label"].values):
    binary_y[i * 5 + lbl] = 1.0

# 3. Fit Classical Ensemble
classical = ClassicalEnsemble(n_splits=5)
oof_classical, test_classical = classical.fit(X_train, binary_y, X_test, n_samples_train)

# 4. Fit Custom Neural Network
oof_scratch, test_scratch = train_model_from_scratch(X_train, binary_y, X_test, n_samples_train, n_splits=5)

# 5. Assemble prediction dictionaries
all_oof = {
    "scratch": oof_scratch,
    "lgbm": oof_classical["lgbm"],
    "xgb": oof_classical["xgb"],
    "cat": oof_classical["cat"]
}

all_test = {
    "scratch": test_scratch,
    "lgbm": test_classical["lgbm"],
    "xgb": test_classical["xgb"],
    "cat": test_classical["cat"]
}'''
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## ⚖️ 8. Optuna Smart Ensembling with Comprehensive W&B Logging']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''def normalize(s):
    mn = s.min(axis=1, keepdims=True)
    mx = s.max(axis=1, keepdims=True)
    return (s - mn) / (mx - mn + 1e-9)

# Normalize predictions dictionary
for key in list(all_oof.keys()):
    all_oof[key]  = normalize(all_oof[key])
    all_test[key] = normalize(all_test[key])

# Non-duplicate sample filtering to prevent CV leak
mask_nodup = ~train_df["prompt"].duplicated(keep=False)
labels_nd  = label_arr[mask_nodup]
all_oof_nd = {key: v[mask_nodup] for key, v in all_oof.items()}
MODEL_NAMES = list(all_oof.keys())

print(f"Honest validation subset (no duplicates): {mask_nodup.sum()} samples.")
print(f"Models available for ensemble: {MODEL_NAMES}")

def objective_nd(trial):
    w = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in MODEL_NAMES])
    w = w / (w.sum() + 1e-9)
    ens_preds = sum(w[i] * all_oof_nd[MODEL_NAMES[i]] for i in range(len(MODEL_NAMES)))
    
    trial_metrics = compute_metrics(labels_nd, ens_preds)
    val_map3 = trial_metrics["map3"]

    if USE_WANDB and getattr(wandb, "run", None) is not None:
        log_payload = {
            "optuna/trial_map3": val_map3,
            "optuna/trial_f1_macro": trial_metrics["f1_macro"],
            "optuna/trial_accuracy": trial_metrics["accuracy"],
            "optuna/trial_number": trial.number
        }
        for idx, n in enumerate(MODEL_NAMES):
            log_payload[f"optuna_weight/{n}"] = float(w[idx])
        try:
            wandb.log(log_payload)
        except Exception:
            pass
            
    return val_map3

study = optuna.create_study(direction="maximize")
study.optimize(objective_nd, n_trials=300)

bp = study.best_trial.params
norm_w = np.array([bp[f"w_{n}"] for n in MODEL_NAMES])
norm_w = norm_w / (norm_w.sum() + 1e-9)
best_val = study.best_trial.value

print(f"\n🏆 Optimized Ensemble CV MAP@3 (no-leak) = {best_val:.4f}")
for i, n in enumerate(MODEL_NAMES):
    print(f"  Model: {n:<15} | Weight: {norm_w[i]:.4f}")

# Compute & log final ensemble cross-validation metrics
final_ens_oof = sum(norm_w[i] * all_oof_nd[MODEL_NAMES[i]] for i in range(len(MODEL_NAMES)))
ensemble_metrics = compute_metrics(labels_nd, final_ens_oof)

print(f"\n🏆 Final Ensemble Cross-Validation Metrics:")
print(f"   MAP@3:           {ensemble_metrics['map3']:.4f}")
print(f"   F1 Score (Macro):{ensemble_metrics['f1_macro']:.4f}")
print(f"   F1 Score (Wgtd): {ensemble_metrics['f1_weighted']:.4f}")
print(f"   Accuracy:        {ensemble_metrics['accuracy']:.4f}")
print(f"   Log Loss:        {ensemble_metrics['log_loss']:.4f}")

log_metrics_to_wandb(ensemble_metrics, prefix="final_ensemble/val")'''
    },
    {
        'cell_type': 'markdown',
        'metadata': {},
        'source': ['## 📄 9. Generate Kaggle Submission, Test Prediction Metrics & W&B Artifacts']
    },
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''final_test = sum(norm_w[i] * all_test[MODEL_NAMES[i]] for i in range(len(MODEL_NAMES)))
top3_preds = scores_to_top3(final_test)
top1_preds = final_test.argmax(axis=1)
pred_strs = [" ".join(IDX_TO_LABEL[idx] for idx in p) for p in top3_preds]

sub = pd.DataFrame({
    "id": test_df["id"].values if "id" in test_df.columns else test_df["ID"].values,
    "prediction": pred_strs
})

sub.to_csv("submission.csv", index=False)
print("\n✅ submission.csv generated and saved successfully!")
print(sub.head(10).to_string(index=False))

# Test set metrics & prediction distributions
test_counts = pd.Series([IDX_TO_LABEL[idx] for idx in top1_preds]).value_counts().to_dict()
norm_final_test = final_test / (final_test.sum(axis=1, keepdims=True) + 1e-9)
avg_confidence = float(norm_final_test.max(axis=1).mean())
pred_entropy = float(-np.sum(norm_final_test * np.log(norm_final_test + 1e-15), axis=1).mean())

print(f"\n📊 Test Set Prediction Class Counts: {test_counts}")
print(f"📊 Test Average Confidence (Top-1 prob): {avg_confidence:.4f}")
print(f"📊 Test Prediction Entropy: {pred_entropy:.4f}")

if USE_WANDB and getattr(wandb, "run", None) is not None:
    try:
        log_payload = {
            "test/avg_confidence": avg_confidence,
            "test/prediction_entropy": pred_entropy,
            "final/ensemble_cv_map3": ensemble_metrics["map3"],
            "final/ensemble_cv_f1_macro": ensemble_metrics["f1_macro"],
            "final/ensemble_cv_f1_weighted": ensemble_metrics["f1_weighted"],
            "final/ensemble_cv_accuracy": ensemble_metrics["accuracy"],
            "final/ensemble_cv_log_loss": ensemble_metrics["log_loss"],
        }
        for k, v in test_counts.items():
            log_payload[f"test_class_count/{k}"] = int(v)
            
        wandb.log(log_payload)

        # Log Class Distribution Table & Plot
        class_df = pd.DataFrame(list(test_counts.items()), columns=["Option", "Count"])
        wandb.log({"test/class_distribution": wandb.plot.bar(wandb.Table(dataframe=class_df), "Option", "Count", title="Test Predictions Distribution")})
        
        # Save & Upload submission.csv as W&B Artifact
        artifact = wandb.Artifact("mcq_submission", type="submission", description="Kaggle MCQ Competition Submission CSV")
        artifact.add_file("submission.csv")
        wandb.log_artifact(artifact)
        print("📦 Uploaded submission.csv to Weights & Biases as an Artifact!")

        wandb.finish()
        print("🎉 Weights & Biases run completed successfully!")
    except Exception as e:
        print(f"W&B final log error: {e}")'''
    }
]

nb_dict = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.12.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

with open(notebook_path, 'w') as f:
    json.dump(nb_dict, f, indent=1)

print('✅ Notebook updated successfully!')
