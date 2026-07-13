#!/usr/bin/env python3
"""
================================================================================
INFERENCE SCRIPT — MCQ Competition
================================================================================
Standalone inference: loads saved models and generates submission.csv
Usage:
    python inference.py --train_path train.csv --test_path test.csv \
                        --model_dir models --output submission.csv
================================================================================
"""
import os, gc, re, json, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OPTION_COLS   = ["A", "B", "C", "D", "E"]
LABEL_TO_IDX  = {k: i for i, k in enumerate(OPTION_COLS)}
IDX_TO_LABEL  = {i: k for k, i in LABEL_TO_IDX.items()}

# ── Device detection ──────────────────────────────────────────────────────────
try:
    import torch
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else
                           "mps"  if torch.backends.mps.is_available() else "cpu")
    USE_AMP = DEVICE.type == "cuda"
    HAS_TORCH = True
except ImportError:
    DEVICE = "cpu"; USE_AMP = False; HAS_TORCH = False


# ── Metric ────────────────────────────────────────────────────────────────────
def apk(actual: int, predicted: List[int], k: int = 3) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]
    score, hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            hits  += 1
            score += hits / (i + 1.0)
    return score

def mapk(actuals: List[int], preds: List[List[int]], k: int = 3) -> float:
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, preds)]))

def scores_to_top3(scores: np.ndarray) -> List[List[int]]:
    return [np.argsort(-row)[:3].tolist() for row in scores]

def labels_to_top3_str(scores: np.ndarray) -> List[str]:
    return [" ".join(IDX_TO_LABEL[i] for i in np.argsort(-row)[:3])
            for row in scores]


# ── Feature helpers ───────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split()

def ngram_overlap(t1: str, t2: str, n: int = 2) -> float:
    def ngrams(toks, n):
        return set(zip(*[toks[i:] for i in range(n)]))
    a, b = tokenize(t1), tokenize(t2)
    ga, gb = ngrams(a, n), ngrams(b, n)
    if not ga and not gb: return 0.0
    return len(ga & gb) / (len(ga | gb) + 1e-9)

def keyword_overlap(t1: str, t2: str, stopwords=None) -> float:
    sw = stopwords or set()
    a = set(tokenize(t1)) - sw
    b = set(tokenize(t2)) - sw
    if not a and not b: return 0.0
    return len(a & b) / (len(a | b) + 1e-9)


def build_features_for_row(prompt: str, options: List[str],
                            tfidf_vec, bm25_available: bool = False) -> np.ndarray:
    """Build feature vector for one question (all 5 options)."""
    try:
        from rank_bm25 import BM25Okapi
        bm25_model  = BM25Okapi([tokenize(o) for o in options])
        bm25_scores = bm25_model.get_scores(tokenize(prompt))
    except Exception:
        bm25_scores = np.zeros(5)

    try:
        from sklearn.metrics.pairwise import cosine_similarity
        prompt_vec = tfidf_vec.transform([prompt])
    except Exception:
        prompt_vec = None

    rows = []
    option_lens = [len(o) for o in options]
    sorted_lens = sorted(option_lens, reverse=True)
    neg_words   = {"not","no","never","neither","nor","without","cannot"}

    for i, opt in enumerate(options):
        tfidf_cos = 0.0
        if prompt_vec is not None:
            try:
                opt_vec   = tfidf_vec.transform([opt])
                tfidf_cos = float(cosine_similarity(prompt_vec, opt_vec)[0, 0])
            except Exception:
                pass

        bm25_rank = 5 - int(np.sum(bm25_scores < bm25_scores[i]))
        feat = [
            len(prompt), len(opt), len(opt.split()),
            len(opt) / (len(prompt) + 1),
            len(opt.split()) / (len(prompt.split()) + 1),
            int(len(opt) > 200), int(len(opt) < 30),
            keyword_overlap(prompt, opt),
            ngram_overlap(prompt, opt, 2),
            ngram_overlap(prompt, opt, 3),
            i,                                          # option position
            sorted_lens.index(len(opt)),                # length rank
            int(bool(re.match(r"^\d", opt))),
            int(opt[0].isupper()) if opt else 0,
            int(" ".join(opt.split()[:4]).lower() in prompt.lower()),
            len(opt) / (np.mean(option_lens) + 1e-9),
            len(re.findall(r"\b\d+\.?\d*\b", opt)),
            int(bool(neg_words & set(tokenize(opt)))),
            tfidf_cos,
            float(bm25_scores[i]),
            bm25_rank,
        ]
        rows.append(feat)
    return np.array(rows)  # (5, n_feats)


# ── Classical model inference ─────────────────────────────────────────────────
def run_classical_inference(df: pd.DataFrame,
                             model_dir: Path,
                             tfidf_vec) -> np.ndarray:
    """Run saved classical models on df. Returns (N, 5) averaged scores."""
    from joblib import load
    from sklearn.metrics.pairwise import cosine_similarity

    all_preds = []
    model_names = ["lr_full", "svm_full", "lgbm_full", "xgb_full", "cat_full"]

    for mn in model_names:
        model_path = model_dir / f"{mn}_model.joblib"
        if not model_path.exists():
            print(f"  ⚠️  {mn} model not found — skipping")
            continue

        print(f"  Loading {mn} ...")
        model = load(model_path)

        X_rows = []
        for _, row in df.iterrows():
            options = [str(row[c]) for c in OPTION_COLS]
            feat_mat = build_features_for_row(str(row["prompt"]), options, tfidf_vec)
            X_rows.append(feat_mat)

        X = np.concatenate(X_rows)  # (N*5, F)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.decision_function(X)
            probs = (probs - probs.min()) / (probs.ptp() + 1e-9)

        probs_2d = probs.reshape(len(df), 5)
        # Normalize per row
        mn_val = probs_2d.min(axis=1, keepdims=True)
        mx_val = probs_2d.max(axis=1, keepdims=True)
        probs_2d = (probs_2d - mn_val) / (mx_val - mn_val + 1e-9)
        all_preds.append(probs_2d)

    if not all_preds:
        return np.ones((len(df), 5)) / 5.0
    return np.mean(all_preds, axis=0)


# ── Sentence Transformer inference ────────────────────────────────────────────
def run_st_inference(df: pd.DataFrame, model_id: str,
                     batch_size: int = 64) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(f"  ⚠️  sentence-transformers unavailable — skipping {model_id}")
        return np.zeros((len(df), 5))

    print(f"  Computing embeddings: {model_id}")
    model = SentenceTransformer(model_id, device=str(DEVICE))
    prompts  = df["prompt"].tolist()
    flat_opts = [str(row[c]) for _, row in df.iterrows() for c in OPTION_COLS]

    from tqdm.auto import tqdm
    p_embs  = model.encode(prompts, batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)
    o_embs  = model.encode(flat_opts, batch_size=batch_size,
                            show_progress_bar=True, normalize_embeddings=True)
    o_embs3d = o_embs.reshape(len(df), 5, -1)
    scores   = np.einsum("nd,nkd->nk", p_embs, o_embs3d)

    del model; gc.collect()
    return scores


# ── DeBERTa inference ─────────────────────────────────────────────────────────
def run_deberta_inference(df: pd.DataFrame, model_dir: Path,
                           max_length: int = 256,
                           batch_size: int = 8) -> np.ndarray:
    if not HAS_TORCH:
        return np.zeros((len(df), 5))

    try:
        from transformers import AutoTokenizer, AutoModelForMultipleChoice
        from torch.utils.data import Dataset, DataLoader
        from tqdm.auto import tqdm
        import torch
    except ImportError:
        return np.zeros((len(df), 5))

    # Find fold models
    fold_dirs = sorted(model_dir.glob("deberta_fold*"))
    if not fold_dirs:
        print("  ⚠️  No DeBERTa fold models found — skipping")
        return np.zeros((len(df), 5))

    class InferDataset(Dataset):
        def __init__(self, df, tokenizer, max_length):
            self.df = df.reset_index(drop=True)
            self.tok = tokenizer
            self.max_length = max_length
        def __len__(self):
            return len(self.df)
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            encs = []
            for opt in OPTION_COLS:
                e = self.tok(str(row["prompt"]), str(row[opt]),
                             max_length=self.max_length, truncation=True,
                             padding="max_length", return_tensors="pt")
                encs.append({k: v.squeeze(0) for k, v in e.items()})
            return {
                "input_ids":      torch.stack([e["input_ids"]      for e in encs]),
                "attention_mask": torch.stack([e["attention_mask"] for e in encs]),
            }

    all_fold_probs = []
    for fold_dir in fold_dirs:
        print(f"  Loading {fold_dir.name} ...")
        tokenizer = AutoTokenizer.from_pretrained(fold_dir)
        model     = AutoModelForMultipleChoice.from_pretrained(fold_dir).to(DEVICE)
        model.eval()
        ds     = InferDataset(df, tokenizer, max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        logits_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {fold_dir.name}", leave=False):
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                if USE_AMP:
                    from torch.cuda.amp import autocast
                    with autocast():
                        out = model(**batch)
                else:
                    out = model(**batch)
                logits_list.append(out.logits.float().cpu())
        logits = torch.cat(logits_list).numpy()
        probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        all_fold_probs.append(probs)
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.mean(all_fold_probs, axis=0)


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="MCQ Inference Pipeline")
    parser.add_argument("--train_path", type=str,
                        default=str(project_root / "train (1).csv"))
    parser.add_argument("--test_path",  type=str,
                        default=str(project_root / "test (1).csv"))
    parser.add_argument("--model_dir",  type=str,
                        default=str(project_root / "models"))
    parser.add_argument("--output",     type=str,
                        default=str(project_root / "submission.csv"))
    parser.add_argument("--tfidf_path", type=str,
                        default=str(project_root / "models/tfidf_vectorizer.joblib"))
    parser.add_argument("--no_deberta",   action="store_true")
    parser.add_argument("--no_sentence_transformers", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    print(f"\n{'='*60}")
    print("  MCQ INFERENCE PIPELINE")
    print(f"{'='*60}")
    print(f"  Test path  : {args.test_path}")
    print(f"  Model dir  : {model_dir}")
    print(f"  Device     : {DEVICE}")

    # Load data
    test_df = pd.read_csv(args.test_path)
    test_df.columns = [c.strip() for c in test_df.columns]
    print(f"  Test rows  : {len(test_df)}")

    # Load TF-IDF vectorizer
    tfidf_vec = None
    try:
        from joblib import load
        tfidf_path = Path(args.tfidf_path)
        if tfidf_path.exists():
            tfidf_vec = load(tfidf_path)
            print(f"  ✓ TF-IDF loaded from {tfidf_path}")
        else:
            # Re-fit on train
            from sklearn.feature_extraction.text import TfidfVectorizer
            train_df  = pd.read_csv(args.train_path)
            all_texts = [str(r["prompt"]) + " " + str(r[c])
                         for _, r in train_df.iterrows()
                         for c in OPTION_COLS]
            tfidf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000,
                                         sublinear_tf=True, min_df=2)
            tfidf_vec.fit(all_texts)
            print("  ✓ TF-IDF re-fitted on train data")
    except Exception as e:
        print(f"  ⚠️  TF-IDF unavailable: {e}")

    all_test_preds = []

    # ── 1. Classical models ───────────────────────────────────────────────────
    print("\n[1] Classical Models ...")
    if tfidf_vec is not None:
        classical_preds = run_classical_inference(test_df, model_dir, tfidf_vec)
        all_test_preds.append(("classical", classical_preds, 1.0))

    # ── 2. Sentence Transformers ──────────────────────────────────────────────
    if not args.no_sentence_transformers:
        print("\n[2] Sentence Transformers ...")
        st_models = {
            "mpnet":  ("sentence-transformers/all-mpnet-base-v2",  1.5),
            "bge":    ("BAAI/bge-large-en-v1.5",                   2.0),
            "e5":     ("intfloat/e5-large-v2",                     1.5),
            "minilm": ("sentence-transformers/all-MiniLM-L12-v2",  0.8),
        }
        for key, (model_id, weight) in st_models.items():
            # Check for cached predictions first
            cache_path = model_dir / f"st_test_{key}.npy"
            if cache_path.exists():
                preds = np.load(cache_path)
                print(f"  ✓ {key}: loaded from cache")
            else:
                preds = run_st_inference(test_df, model_id)
                np.save(cache_path, preds)
            all_test_preds.append((f"st_{key}", preds, weight))

    # ── 3. DeBERTa ────────────────────────────────────────────────────────────
    if not args.no_deberta:
        print("\n[3] DeBERTa Ensemble ...")
        cache_path = model_dir / "deberta_test_probs.npy"
        if cache_path.exists():
            deberta_preds = np.load(cache_path)
            print("  ✓ DeBERTa: loaded from cache")
        else:
            deberta_preds = run_deberta_inference(test_df, model_dir)
            if deberta_preds.max() > 0:
                np.save(cache_path, deberta_preds)
        if deberta_preds.max() > 0:
            all_test_preds.append(("deberta", deberta_preds, 3.0))

    # ── Weighted ensemble ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  Ensembling predictions ...")
    total_w = sum(w for _, _, w in all_test_preds)
    final_scores = np.zeros((len(test_df), 5))
    for name, preds, w in all_test_preds:
        # Normalize
        mn = preds.min(axis=1, keepdims=True)
        mx = preds.max(axis=1, keepdims=True)
        preds_norm = (preds - mn) / (mx - mn + 1e-9)
        final_scores += (w / total_w) * preds_norm
        print(f"    {name:<25} weight={w/total_w:.3f}")

    # ── Generate submission ────────────────────────────────────────────────────
    predictions = labels_to_top3_str(final_scores)
    sub = pd.DataFrame({"ID": test_df["id"].values, "Prediction": predictions})
    sub.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"  ✅ Submission saved → {args.output}")
    print(f"  Rows: {len(sub)}")
    print(f"\n  Preview:")
    print(sub.head(10).to_string(index=False))
    print(f"{'='*60}")
    return sub


if __name__ == "__main__":
    main()

# Note: Verifies test file path existence and triggers clear error on failure