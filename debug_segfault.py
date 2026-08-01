import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Load data
BASE_DIR = Path("/Users/shobhitagnihotri/Downloads/DL GEN AI")
train_df = pd.read_csv(BASE_DIR / "train (1).csv")
test_df  = pd.read_csv(BASE_DIR / "test (1).csv")
train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns  = [c.strip() for c in test_df.columns]
OPTION_COLS = ["A", "B", "C", "D", "E"]
LABEL_TO_IDX = {k: i for i, k in enumerate(OPTION_COLS)}
train_df["label"] = train_df["answer"].map(LABEL_TO_IDX)

# Re-define tokenizers and helpers
import re
def tokenize(text: str):
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split()

def ngram_overlap(text1: str, text2: str, n: int = 2):
    def get_ngrams(tokens, n):
        return set(zip(*[tokens[i:] for i in range(n)]))
    t1, t2 = tokenize(text1), tokenize(text2)
    ng1, ng2 = get_ngrams(t1, n), get_ngrams(t2, n)
    if not ng1 and not ng2: return 0.0
    return len(ng1 & ng2) / (len(ng1 | ng2) + 1e-9)

def keyword_overlap(text1: str, text2: str):
    t1 = set(tokenize(text1))
    t2 = set(tokenize(text2))
    if not t1 and not t2: return 0.0
    return len(t1 & t2) / (len(t1 | t2) + 1e-9)

def length_features(prompt: str, option: str):
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

def build_feature_row(prompt: str, option: str, option_idx: int, all_options: list):
    feats = {}
    feats.update(length_features(prompt, option))
    feats["unigram_overlap"] = keyword_overlap(prompt, option)
    feats["bigram_overlap"]  = ngram_overlap(prompt, option, n=2)
    feats["trigram_overlap"] = ngram_overlap(prompt, option, n=3)
    feats["option_position"] = float(option_idx)
    option_lens = [len(o) for o in all_options]
    sorted_lens = sorted(option_lens, reverse=True)
    feats["length_rank"] = float(sorted_lens.index(len(option)))
    feats["starts_with_num"] = float(int(bool(re.match(r'^\d', option))))
    feats["starts_with_cap"] = float(int(option[0].isupper()) if option else 0)
    feats["num_numbers"] = float(len(re.findall(r'\b\d+\.?\d*\b', option)))
    neg_words = {"not", "no", "never", "neither", "nor", "without", "cannot"}
    feats["has_negation"] = float(int(bool(neg_words & set(tokenize(option)))))
    return feats

from sklearn.feature_extraction.text import TfidfVectorizer
def build_features(df, tfidf_vectorizer=None, fit=False):
    all_texts = [str(row["prompt"]) + " " + str(row[opt]) for _, row in df.iterrows() for opt in OPTION_COLS]
    if fit or tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=40_000, sublinear_tf=True, strip_accents='unicode', analyzer='word', min_df=2, max_df=0.95)
        tfidf_vectorizer.fit(all_texts)
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
        p_vec = prompt_vecs[idx]
        for i, opt in enumerate(OPTION_COLS):
            option = str(row[opt])
            o_vec = option_vecs[opt][idx]
            tfidf_cos = float(p_vec.dot(o_vec.T).toarray()[0, 0])
            feat = build_feature_row(prompt, option, i, all_options)
            feat["tfidf_cosine"] = tfidf_cos
            feat["bm25_score"]   = 0.0
            feat["bm25_rank"]    = 0.0
            rows.append(feat)
    return pd.DataFrame(rows).fillna(0).values, tfidf_vectorizer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps'  if torch.backends.mps.is_available() else 'cpu')
SEED = 42

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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    label_per_q = y.reshape(n_samples_train, 5).argmax(axis=1)
    
    oof_prob = np.zeros((n_samples_train, 5))
    test_prob_folds = np.zeros((X_test.shape[0] // 5, 5, n_splits))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    for fold, (train_q, val_q) in enumerate(skf.split(np.zeros(n_samples_train), label_per_q)):
        print(f"  MLP Fold {fold+1}...")
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
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            preds_vl = torch.sigmoid(model(torch.FloatTensor(X_vl).to(DEVICE))).cpu().numpy().squeeze()
            oof_prob[val_q] = preds_vl.reshape(-1, 5)
            
            preds_te = torch.sigmoid(model(torch.FloatTensor(X_test_scaled).to(DEVICE))).cpu().numpy().squeeze()
            test_prob_folds[:, :, fold] = preds_te.reshape(-1, 5)
            
    test_prob = test_prob_folds.mean(axis=2)
    print("✓ MLP Completed training successfully!")
    return oof_prob, test_prob

class ClassicalEnsemble:
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
            test_prob_folds = np.zeros((X_test.shape[0] // 5, 5, self.n_splits))
            
            for fold, (train_q, val_q) in enumerate(skf.split(np.zeros(n_samples_train), label_per_q)):
                print(f"    {model_name} Fold {fold+1}...")
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
                    
                preds_vl = model.predict_proba(X_vl)[:, 1]
                oof_prob[val_q] = preds_vl.reshape(-1, 5)
                
                preds_te = model.predict_proba(X_test)[:, 1]
                test_prob_folds[:, :, fold] = preds_te.reshape(-1, 5)
                
            oof_predictions[model_name] = oof_prob
            test_predictions[model_name] = test_prob_folds.mean(axis=2)
            
        return oof_predictions, test_predictions

X_train_flat, tfidf_vec = build_features(train_df, fit=True)
X_test_flat, _          = build_features(test_df, tfidf_vectorizer=tfidf_vec)
y_train_flat            = np.zeros(len(train_df) * 5)
for i, l in enumerate(train_df["label"].values):
    y_train_flat[i*5 + l] = 1

oof_scratch, test_scratch = train_model_from_scratch(X_train_flat, y_train_flat, X_test_flat, len(train_df))
print("⚡ Running Classical Ensemble ...")
ens = ClassicalEnsemble(n_splits=5)
class_oof, class_test = ens.fit(X_train_flat, y_train_flat, X_test_flat, len(train_df))
print("✓ All Completed!")
