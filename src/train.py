#!/usr/bin/env python3
"""
================================================================================
MCQ COMPETITION - COMPLETE WINNING PIPELINE
================================================================================
Author  : Kaggle Grandmaster Pipeline
Metric  : MAP@3 (Mean Average Precision @ 3)
Task    : 5-choice MCQ answering — predict top 3 ranked answers
Dataset : train.csv / test.csv (Science/Physics heavy topics)
================================================================================
ARCHITECTURE:
  1. EDA + Visualization
  2. Feature Engineering (TF-IDF, BM25, Embeddings, NLP features)
  3. Multi-Model Training (Classical ML + Transformers)
  4. 5-Fold Stratified CV with early stopping
  5. Ensembling with Optuna weight optimization
  6. MAP@3-optimized ranking & submission
================================================================================
"""

# ─────────────────────────── CELL 1: IMPORTS & SETUP ────────────────────────
import os, sys, gc, re, time, json, warnings, random
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Seed for reproducibility ─────────────────────────────────────────────────
SEED = 42
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

set_seed(SEED)

# ── Device setup ─────────────────────────────────────────────────────────────
try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    USE_AMP = DEVICE.type == "cuda"
    print(f"✅ PyTorch {torch.__version__} | Device: {DEVICE} | AMP: {USE_AMP}")
except ImportError:
    DEVICE = "cpu"; USE_AMP = False
    print("⚠️  PyTorch not found — transformer models disabled")

# ── Path configuration ────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "train (1).csv"
TEST_PATH  = BASE_DIR / "test (1).csv"
SUB_PATH   = BASE_DIR / "sample_submission (2).csv"
OUT_DIR    = BASE_DIR / "outputs"
MODEL_DIR  = BASE_DIR / "models"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

OPTION_COLS = ["A", "B", "C", "D", "E"]
LABEL_TO_IDX = {k: i for i, k in enumerate(OPTION_COLS)}
IDX_TO_LABEL = {i: k for k, i in LABEL_TO_IDX.items()}

print(f"📂 Base dir : {BASE_DIR}")
print(f"📦 Train    : {TRAIN_PATH.name}")
print(f"📦 Test     : {TEST_PATH.name}")


# ─────────────────────────── CELL 2: METRIC ──────────────────────────────────
def apk(actual: int, predicted: List[int], k: int = 3) -> float:
    """Average Precision @ K for a single sample."""
    if len(predicted) > k:
        predicted = predicted[:k]
    score, num_hits = 0.0, 0
    for i, p in enumerate(predicted):
        if p == actual and p not in predicted[:i]:
            num_hits += 1
            score += num_hits / (i + 1.0)
    return score

def mapk(actuals: List[int], predictions: List[List[int]], k: int = 3) -> float:
    """Mean Average Precision @ K."""
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predictions)])

def scores_to_top3(scores: np.ndarray) -> List[List[int]]:
    """Convert score matrix (N, 5) to list of top-3 ranked indices."""
    return [np.argsort(-row)[:3].tolist() for row in scores]

def labels_to_top3(pred_matrix: np.ndarray) -> List[str]:
    """Convert score matrix to space-separated label strings e.g. 'A C B'."""
    results = []
    for row in pred_matrix:
        top3_idx = np.argsort(-row)[:3]
        results.append(" ".join([IDX_TO_LABEL[i] for i in top3_idx]))
    return results

print("✅ Metric functions defined")


# ─────────────────────────── CELL 3: DATA LOADING ────────────────────────────
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    # Normalize column names
    train.columns = [c.strip() for c in train.columns]
    test.columns  = [c.strip() for c in test.columns]
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")
    print(f"Train columns: {list(train.columns)}")
    return train, test

train_df, test_df = load_data()

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert set(OPTION_COLS).issubset(set(train_df.columns)), "Missing option columns in train"
assert "answer" in train_df.columns, "Missing answer column in train"
assert set(OPTION_COLS).issubset(set(test_df.columns)), "Missing option columns in test"

# Add numeric label
train_df["label"] = train_df["answer"].map(LABEL_TO_IDX)
print(f"Label distribution:\n{train_df['label'].value_counts().sort_index()}")


# ─────────────────────────── CELL 4: EDA ─────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#444",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})
PALETTE = ["#7c3aed", "#06b6d4", "#f59e0b", "#10b981", "#ef4444"]

def run_eda(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(20, 22), facecolor="#0f0f1a")
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

    # ── 1. Answer distribution ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["answer"].value_counts().sort_index()
    bars = ax1.bar(counts.index, counts.values, color=PALETTE, edgecolor="#222", linewidth=0.8)
    for bar, v in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 5, str(v),
                 ha="center", va="bottom", color="#e0e0e0", fontsize=9)
    ax1.set_title("Answer Distribution", color="#7c3aed", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Option"); ax1.set_ylabel("Count")
    pct = counts / counts.sum() * 100
    ax1.set_ylim(0, counts.max() * 1.18)

    # ── 2. Prompt length distribution ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    prompt_lens = df["prompt"].str.len()
    ax2.hist(prompt_lens, bins=40, color="#06b6d4", edgecolor="#0f0f1a", alpha=0.85)
    ax2.axvline(prompt_lens.mean(), color="#f59e0b", linestyle="--", linewidth=1.5,
                label=f"Mean={prompt_lens.mean():.0f}")
    ax2.set_title("Prompt Length (chars)", color="#06b6d4", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Characters"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)

    # ── 3. Option length distributions ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for i, opt in enumerate(OPTION_COLS):
        lens = df[opt].str.len()
        ax3.hist(lens, bins=30, alpha=0.55, color=PALETTE[i], label=opt, edgecolor="#0f0f1a")
    ax3.set_title("Option Length (chars)", color="#f59e0b", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Characters"); ax3.set_ylabel("Count")
    ax3.legend(fontsize=8)

    # ── 4. Correct vs wrong option lengths ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    correct_lens, wrong_lens = [], []
    for _, row in df.iterrows():
        correct_lens.append(len(str(row[row["answer"]])))
        others = [len(str(row[c])) for c in OPTION_COLS if c != row["answer"]]
        wrong_lens.extend(others)
    ax4.hist(correct_lens, bins=30, alpha=0.75, color="#10b981", label="Correct", edgecolor="#0f0f1a")
    ax4.hist(wrong_lens,   bins=30, alpha=0.45, color="#ef4444", label="Wrong",   edgecolor="#0f0f1a")
    ax4.set_title("Correct vs Wrong Option Lengths", color="#10b981", fontweight="bold", fontsize=11)
    ax4.set_xlabel("Characters"); ax4.legend(fontsize=8)

    # ── 5. Word count stats ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    word_counts = {opt: df[opt].str.split().str.len() for opt in OPTION_COLS}
    bp = ax5.boxplot([word_counts[o] for o in OPTION_COLS],
                     labels=OPTION_COLS, patch_artist=True,
                     boxprops=dict(linewidth=1),
                     medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax5.set_title("Option Word Count Boxplot", color="#f59e0b", fontweight="bold", fontsize=11)
    ax5.set_xlabel("Option"); ax5.set_ylabel("Words")

    # ── 6. Answer class vs avg option length (position bias) ──────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    avg_len_by_answer = {}
    for opt in OPTION_COLS:
        subset = df[df["answer"] == opt]
        avg_len_by_answer[opt] = subset[opt].str.len().mean() if len(subset) > 0 else 0
    ax6.bar(avg_len_by_answer.keys(), avg_len_by_answer.values(),
            color=PALETTE, edgecolor="#222")
    ax6.set_title("Avg Correct Option Length by Label\n(Position Bias Check)",
                  color="#7c3aed", fontweight="bold", fontsize=11)
    ax6.set_xlabel("Correct Option"); ax6.set_ylabel("Avg Chars")

    # ── 7. Missing values ─────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    miss = df.isnull().sum()
    miss = miss[miss > 0] if miss.any() else pd.Series({"No Missing": 0})
    ax7.barh(miss.index, miss.values, color="#ef4444", edgecolor="#222")
    ax7.set_title("Missing Values", color="#ef4444", fontweight="bold", fontsize=11)
    ax7.set_xlabel("Count")

    # ── 8. Duplicate prompts ──────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    dup_counts = df["prompt"].value_counts()
    dup_gt1 = (dup_counts > 1).sum()
    ax8.pie([len(df) - dup_gt1, dup_gt1],
            labels=["Unique", "Duplicated"],
            colors=["#10b981", "#ef4444"],
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "#e0e0e0"})
    ax8.set_title("Prompt Duplicates", color="#06b6d4", fontweight="bold", fontsize=11)

    # ── 9. Top question prefixes ───────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    prefixes = df["prompt"].str.extract(r"^([^:?]+)")[0].str.strip()
    top_pre = prefixes.value_counts().head(8)
    ax9.barh(top_pre.index, top_pre.values, color=PALETTE * 2, edgecolor="#222")
    ax9.set_title("Top Question Prefixes", color="#f59e0b", fontweight="bold", fontsize=11)
    ax9.invert_yaxis()

    # ── 10. Answer label vs prompt length ─────────────────────────────────────
    ax10 = fig.add_subplot(gs[3, :2])
    for i, opt in enumerate(OPTION_COLS):
        subset = df[df["answer"] == opt]["prompt"].str.len()
        ax10.scatter([i]*len(subset), subset, alpha=0.25, color=PALETTE[i], s=15)
        ax10.scatter([i], [subset.mean()], color="white", s=80, zorder=5, marker="D")
    ax10.set_xticks(range(5)); ax10.set_xticklabels(OPTION_COLS)
    ax10.set_title("Prompt Length vs Answer Label", color="#10b981", fontweight="bold", fontsize=11)
    ax10.set_xlabel("Answer Label"); ax10.set_ylabel("Prompt Length (chars)")

    # ── 11. Pairwise option-answer correlation ────────────────────────────────
    ax11 = fig.add_subplot(gs[3, 2])
    corr_data = np.zeros((5, 5))
    for i, opt_i in enumerate(OPTION_COLS):
        for j, opt_j in enumerate(OPTION_COLS):
            overlap = (df[opt_i].str[:50] == df[opt_j].str[:50]).sum()
            corr_data[i, j] = overlap / len(df)
    im = ax11.imshow(corr_data, cmap="YlOrRd", aspect="auto")
    ax11.set_xticks(range(5)); ax11.set_yticks(range(5))
    ax11.set_xticklabels(OPTION_COLS); ax11.set_yticklabels(OPTION_COLS)
    ax11.set_title("Option Prefix Overlap Heatmap", color="#7c3aed", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax11)

    fig.suptitle("📊 MCQ Competition — Exploratory Data Analysis",
                 fontsize=16, color="white", fontweight="bold", y=1.01)
    plt.savefig(OUT_DIR / "eda_analysis.png", bbox_inches="tight",
                facecolor="#0f0f1a", dpi=150)
    plt.show()
    print(f"💾 EDA saved → {OUT_DIR / 'eda_analysis.png'}")

    # ── Print lexical stats ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LEXICAL STATISTICS")
    print("="*60)
    print(f"{'Metric':<35} {'Value':>10}")
    print("-"*47)
    print(f"{'Train samples':<35} {len(df):>10}")
    print(f"{'Unique prompts':<35} {df['prompt'].nunique():>10}")
    print(f"{'Duplicate prompts':<35} {df['prompt'].duplicated().sum():>10}")
    print(f"{'Avg prompt length (chars)':<35} {df['prompt'].str.len().mean():>10.1f}")
    print(f"{'Avg prompt length (words)':<35} {df['prompt'].str.split().str.len().mean():>10.1f}")
    for opt in OPTION_COLS:
        print(f"{'Avg ' + opt + ' length (chars)':<35} {df[opt].str.len().mean():>10.1f}")
    print(f"{'Missing values total':<35} {df.isnull().sum().sum():>10}")

    print("\n" + "="*60)
    print("ANSWER DISTRIBUTION")
    print("="*60)
    for opt in OPTION_COLS:
        n = (df["answer"] == opt).sum()
        pct = n / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {opt}: {bar:<25} {n:>4} ({pct:>5.1f}%)")

run_eda(train_df)


# ─────────────────────────── CELL 5: FEATURE ENGINEERING ─────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("⚠️  rank_bm25 not found — BM25 features disabled")

try:
    import textstat
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("⚠️  textstat not found — readability features disabled")

try:
    import nltk
    nltk.download("punkt",     quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
    HAS_NLTK = True
except ImportError:
    STOPWORDS = set(); HAS_NLTK = False
    print("⚠️  NLTK not found — NLP features limited")


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer with lowercasing."""
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split()


def ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """Compute n-gram Jaccard overlap between two texts."""
    def ngrams(tokens, n):
        return set(zip(*[tokens[i:] for i in range(n)]))
    t1, t2 = tokenize(text1), tokenize(text2)
    ng1, ng2 = ngrams(t1, n), ngrams(t2, n)
    if not ng1 and not ng2:
        return 0.0
    return len(ng1 & ng2) / (len(ng1 | ng2) + 1e-9)


def keyword_overlap(text1: str, text2: str) -> float:
    """Keyword (non-stopword) Jaccard overlap."""
    t1 = set(tokenize(text1)) - STOPWORDS
    t2 = set(tokenize(text2)) - STOPWORDS
    if not t1 and not t2:
        return 0.0
    return len(t1 & t2) / (len(t1 | t2) + 1e-9)


def length_features(prompt: str, option: str) -> Dict[str, float]:
    """Length-based features for a prompt-option pair."""
    p_chars, o_chars = len(prompt), len(option)
    p_words = len(prompt.split())
    o_words = len(option.split())
    return {
        "prompt_len_chars":  p_chars,
        "option_len_chars":  o_chars,
        "option_len_words":  o_words,
        "char_ratio":        o_chars / (p_chars + 1),
        "word_ratio":        o_words / (p_words + 1),
        "option_is_long":    int(o_chars > 200),
        "option_is_short":   int(o_chars < 30),
    }


def readability_features(text: str) -> Dict[str, float]:
    """Flesch reading ease and other readability metrics."""
    if not HAS_TEXTSTAT or not text.strip():
        return {"flesch": 0.0, "flesch_kincaid": 0.0, "smog": 0.0}
    return {
        "flesch":          textstat.flesch_reading_ease(text),
        "flesch_kincaid":  textstat.flesch_kincaid_grade(text),
        "smog":            textstat.smog_index(text),
    }


def bm25_score(prompt_tokens: List[str], option_tokens: List[str],
               bm25: "BM25Okapi") -> float:
    """BM25 score for an option given a query."""
    return float(bm25.get_scores(prompt_tokens)[0]) if HAS_BM25 else 0.0


def build_feature_row(prompt: str, option: str, option_idx: int,
                      all_options: List[str]) -> Dict[str, float]:
    """Build a feature vector for one (prompt, option) pair."""
    feats: Dict[str, float] = {}

    # ── Length features ────────────────────────────────────────────────────────
    feats.update(length_features(prompt, option))

    # ── Overlap features ───────────────────────────────────────────────────────
    feats["unigram_overlap"]   = keyword_overlap(prompt, option)
    feats["bigram_overlap"]    = ngram_overlap(prompt, option, n=2)
    feats["trigram_overlap"]   = ngram_overlap(prompt, option, n=3)

    # ── Option position (positional bias) ─────────────────────────────────────
    feats["option_position"]   = option_idx  # 0=A, 1=B, ..., 4=E

    # ── Option relative length rank (is this the longest option?) ─────────────
    option_lens = [len(o) for o in all_options]
    sorted_lens = sorted(option_lens, reverse=True)
    feats["length_rank"]      = sorted_lens.index(len(option))  # 0=longest

    # ── Starts-with capital / number ──────────────────────────────────────────
    feats["starts_with_num"]  = int(bool(re.match(r"^\d", option)))
    feats["starts_with_cap"]  = int(option[0].isupper()) if option else 0

    # ── Readability ────────────────────────────────────────────────────────────
    feats.update(readability_features(option))

    # ── Prompt contains option prefix (≥4 words) ──────────────────────────────
    opt_words = option.split()[:4]
    feats["option_in_prompt"] = int(" ".join(opt_words).lower() in prompt.lower())

    # ── Relative length vs mean option ───────────────────────────────────────-
    mean_len = np.mean(option_lens)
    feats["len_vs_mean"]      = len(option) / (mean_len + 1e-9)

    # ── Number of numbers in option ────────────────────────────────────────────
    feats["num_numbers"]      = len(re.findall(r"\b\d+\.?\d*\b", option))

    # ── Contains negation ─────────────────────────────────────────────────────
    neg_words = {"not", "no", "never", "neither", "nor", "without", "cannot", "isn't", "aren't", "doesn't", "don't", "won't"}
    feats["has_negation"]     = int(bool(neg_words & set(tokenize(option))))

    return feats


def build_features(df: pd.DataFrame,
                   tfidf_vectorizer: Optional[Any] = None,
                   fit: bool = False) -> Tuple[np.ndarray, Any]:
    """
    Build feature matrix for the entire dataframe.
    Returns:
        X : np.ndarray of shape (N*5, n_features)  — one row per (sample, option)
        tfidf_vectorizer : fitted TfidfVectorizer
    """
    print("🔧 Building features ...")
    all_texts = []
    for _, row in df.iterrows():
        for opt in OPTION_COLS:
            all_texts.append(str(row["prompt"]) + " " + str(row[opt]))

    # ── TF-IDF vectorizer (fit on train, transform on test) ───────────────────
    if fit or tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=50_000,
            sublinear_tf=True, strip_accents="unicode",
            analyzer="word", min_df=2, max_df=0.95
        )
        tfidf_vectorizer.fit(all_texts)

    # Vectorized TF-IDF Transform
    prompts = df["prompt"].astype(str).tolist()
    prompt_vecs = tfidf_vectorizer.transform(prompts)
    
    option_vecs = {}
    for opt in OPTION_COLS:
        options = df[opt].astype(str).tolist()
        option_vecs[opt] = tfidf_vectorizer.transform(options)

    # ── Per-sample feature vectors ─────────────────────────────────────────────
    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        prompt      = str(row["prompt"])
        all_options = [str(row[c]) for c in OPTION_COLS]

        # BM25 model for this sample (each option is a "document")
        if HAS_BM25:
            bm25_model = BM25Okapi([tokenize(o) for o in all_options])
            prompt_toks = tokenize(prompt)
            bm25_scores = bm25_model.get_scores(prompt_toks)
        else:
            bm25_scores = np.zeros(5)

        p_vec = prompt_vecs[idx]
        for i, opt in enumerate(OPTION_COLS):
            option       = str(row[opt])
            o_vec        = option_vecs[opt][idx]
            tfidf_cos    = float(p_vec.dot(o_vec.T).toarray()[0, 0])

            feat = build_feature_row(prompt, option, i, all_options)
            feat["tfidf_cosine"] = tfidf_cos
            feat["bm25_score"]   = float(bm25_scores[i])

            # BM25 rank among options
            bm25_rank = 5 - int(np.sum(bm25_scores < bm25_scores[i]))
            feat["bm25_rank"] = bm25_rank

            rows.append(feat)

    X = pd.DataFrame(rows).fillna(0).values
    print(f"   Feature matrix: {X.shape}")
    return X, tfidf_vectorizer


# Build train features
X_train_raw, tfidf_vec = build_features(train_df, fit=True)

# Build test features (using train's TF-IDF)
X_test_raw, _ = build_features(test_df, tfidf_vectorizer=tfidf_vec, fit=False)

# Create labels array (one per option per sample)
y_train = np.array([
    1.0 if OPTION_COLS[i % 5] == train_df.iloc[i // 5]["answer"] else 0.0
    for i in range(len(train_df) * 5)
])

print(f"X_train: {X_train_raw.shape}, X_test: {X_test_raw.shape}")
print(f"y_train: {y_train.shape}, positive ratio: {y_train.mean():.3f}")


# ─────────────────────────── CELL 6: CLASSICAL ML MODELS ─────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from joblib import dump, load


def reshape_for_ranking(X: np.ndarray, n_options: int = 5) -> np.ndarray:
    """Reshape (N*5, F) to (N, 5*F) — one row per question."""
    n_samples = X.shape[0] // n_options
    return X.reshape(n_samples, n_options * X.shape[1])


def option_probs_from_flat(model, X_flat: np.ndarray, n_options: int = 5) -> np.ndarray:
    """
    Predict probability of 'correct' for each option,
    returning (N, 5) array.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_flat)[:, 1]
    else:
        probs = model.decision_function(X_flat)
        probs = (probs - probs.min()) / (probs.ptp() + 1e-9)
    n_samples = X_flat.shape[0] // n_options
    return probs.reshape(n_samples, n_options)


class ClassicalEnsemble:
    """Runs multiple classical ML models in 5-fold CV and stores OOF predictions."""

    def __init__(self, n_splits: int = 5):
        self.n_splits    = n_splits
        self.models      = {}
        self.oof_scores  = {}
        self.test_preds  = {}

    def _make_models(self) -> Dict:
        return {
            "lr": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(C=1.0, max_iter=1000,
                                               solver="lbfgs", random_state=SEED)),
            ]),
            "svm": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    CalibratedClassifierCV(
                               LinearSVC(C=0.5, max_iter=2000, random_state=SEED),
                               cv=3, method="isotonic")),
            ]),
            "lgbm": lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.02,
                max_depth=4, num_leaves=15,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.5, reg_lambda=1.5,
                min_child_samples=20, random_state=SEED,
                verbose=-1, n_jobs=-1,
            ),
            "xgb": xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.02,
                max_depth=4, subsample=0.7,
                colsample_bytree=0.7, reg_alpha=1.5, reg_lambda=1.5,
                eval_metric="logloss", random_state=SEED,
                verbosity=0, n_jobs=-1,
                early_stopping_rounds=30,  # new XGB API: in constructor
            ),
            "cat": cb.CatBoostClassifier(
                iterations=500, learning_rate=0.02,
                depth=5, l2_leaf_reg=8.0,
                subsample=0.7, random_seed=SEED,
                verbose=0,
            ),
        }

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_test: np.ndarray, n_samples_train: int) -> None:
        """
        5-fold stratified CV. X shape: (N*5, F).
        y shape: (N*5,) — binary label per option.
        """
        # Stratify by question index so all 5 options of a question
        # stay in the same fold (prevents leakage)
        question_idx = np.arange(n_samples_train).repeat(5)
        label_per_q  = y.reshape(n_samples_train, 5).argmax(axis=1)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=SEED)

        for model_name, model in self._make_models().items():
            print(f"\n{'─'*50}")
            print(f"  Training: {model_name.upper()}")
            print(f"{'─'*50}")
            oof_prob = np.zeros((n_samples_train, 5))
            test_prob_folds = np.zeros((X_test.shape[0] // 5, 5, self.n_splits))

            fold_scores = []
            for fold, (train_q_idx, val_q_idx) in enumerate(
                    skf.split(np.zeros(n_samples_train), label_per_q)):

                # Expand to option indices
                train_opt_idx = np.concatenate([
                    np.arange(qi*5, qi*5+5) for qi in train_q_idx])
                val_opt_idx   = np.concatenate([
                    np.arange(qi*5, qi*5+5) for qi in val_q_idx])

                X_tr, y_tr = X[train_opt_idx], y[train_opt_idx]
                X_vl, y_vl = X[val_opt_idx],   y[val_opt_idx]

                # Early stopping for tree models
                if model_name == "lgbm":
                    model.fit(X_tr, y_tr,
                              eval_set=[(X_vl, y_vl)],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(-1)])
                elif model_name == "xgb":
                    model.fit(X_tr, y_tr,
                              eval_set=[(X_vl, y_vl)],
                              verbose=False)
                elif model_name == "cat":
                    model.fit(X_tr, y_tr,
                              eval_set=(X_vl, y_vl),
                              early_stopping_rounds=50,
                              verbose=False)
                else:
                    model.fit(X_tr, y_tr)

                # OOF predictions
                val_probs = option_probs_from_flat(model, X_vl)
                oof_prob[val_q_idx] = val_probs

                # Test predictions for this fold
                test_probs = option_probs_from_flat(model, X_test)
                test_prob_folds[:, :, fold] = test_probs

                # Fold MAP@3
                val_actuals = label_per_q[val_q_idx].tolist()
                val_top3    = scores_to_top3(val_probs)
                fold_map3   = mapk(val_actuals, val_top3)
                fold_scores.append(fold_map3)
                print(f"    Fold {fold+1}/{self.n_splits} MAP@3 = {fold_map3:.4f}")

            cv_map3 = np.mean(fold_scores)
            print(f"  ► {model_name.upper()} CV MAP@3 = {cv_map3:.4f} ± {np.std(fold_scores):.4f}")

            self.models[model_name]     = deepcopy(model)
            self.oof_scores[model_name] = oof_prob
            self.test_preds[model_name] = test_prob_folds.mean(axis=2)

            # Save model
            dump(model, MODEL_DIR / f"{model_name}_model.joblib")

        # ── Fit full models on all data (for final test predictions) ──────────
        print("\n🔄 Fitting full models on all training data ...")
        for model_name, model in self._make_models().items():
            if model_name == "lgbm":
                model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])
            elif model_name == "xgb":
                model.set_params(early_stopping_rounds=None)
                model.fit(X, y, verbose=False)
            elif model_name == "cat":
                model.fit(X, y, verbose=False)
            else:
                model.fit(X, y)
            dump(model, MODEL_DIR / f"{model_name}_full_model.joblib")
            probs = option_probs_from_flat(model, X_test)
            self.test_preds[f"{model_name}_full"] = probs
            print(f"  ✓ {model_name} full model saved")


# Run classical ensemble
classical = ClassicalEnsemble(n_splits=5)
classical.fit(
    X_train_raw, y_train, X_test_raw,
    n_samples_train=len(train_df)
)


# ─────────────────────────── CELL 7: FEATURE IMPORTANCE ─────────────────────
def plot_feature_importance(model, feature_names: List[str],
                             model_name: str, top_k: int = 20):
    """Plot top-k feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    else:
        return

    df_imp = pd.DataFrame({"feature": feature_names, "importance": imp})
    df_imp = df_imp.nlargest(top_k, "importance")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    bars = ax.barh(df_imp["feature"], df_imp["importance"],
                   color="#7c3aed", edgecolor="#222", alpha=0.85)
    ax.set_title(f"Feature Importance — {model_name.upper()}",
                 color="#7c3aed", fontweight="bold", fontsize=12)
    ax.set_xlabel("Importance", color="#e0e0e0")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"feat_importance_{model_name}.png",
                bbox_inches="tight", facecolor="#0f0f1a", dpi=120)
    plt.show()


feat_row_sample = list(build_feature_row(
    "sample prompt", "sample option A", 0,
    ["sample option A", "sample option B", "sample option C", "sample option D", "sample option E"]
).keys()) + ["tfidf_cosine", "bm25_score", "bm25_rank"]

for mn in ["lgbm", "xgb", "cat"]:
    if mn in classical.models:
        m = classical.models[mn]
        if mn == "lgbm":
            plot_feature_importance(m, feat_row_sample, mn)
        elif mn == "xgb":
            plot_feature_importance(m, feat_row_sample, mn)
        elif mn == "cat":
            plot_feature_importance(m, feat_row_sample, mn)


# ─────────────────────────── CELL 8: SENTENCE TRANSFORMERS ───────────────────
from tqdm.auto import tqdm

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("⚠️  sentence-transformers not available")


def compute_st_scores(df: pd.DataFrame,
                      model_name: str,
                      batch_size: int = 64) -> np.ndarray:
    """
    For each question, compute cosine similarity between prompt embedding
    and each option embedding. Returns (N, 5) array.
    """
    if not HAS_ST:
        print(f"  Skipping {model_name} — sentence-transformers unavailable")
        return np.zeros((len(df), 5))

    print(f"\n📡 Computing embeddings: {model_name}")
    st_model = SentenceTransformer(model_name, device=str(DEVICE))

    # Gather all texts
    prompts = df["prompt"].tolist()
    options_all = [[str(row[c]) for c in OPTION_COLS]
                   for _, row in df.iterrows()]

    # Embed prompts
    prompt_embs = st_model.encode(
        prompts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Embed all options (flatten → embed → reshape)
    flat_opts = [opt for opts in options_all for opt in opts]
    opt_embs  = st_model.encode(
        flat_opts, batch_size=batch_size,
        show_progress_bar=True, normalize_embeddings=True,
        convert_to_numpy=True
    )  # shape: (N*5, D)
    opt_embs_3d = opt_embs.reshape(len(df), 5, -1)  # (N, 5, D)

    # Cosine similarity (vectors already normalized → dot product)
    scores = np.einsum("nd,nkd->nk", prompt_embs, opt_embs_3d)

    del st_model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"   ✓ {model_name}: scores shape {scores.shape}")
    return scores


# ── Models to run ────────────────────────────────────────────────────────────
ST_MODELS = {
    "mpnet":    "sentence-transformers/all-mpnet-base-v2",
    "bge":      "BAAI/bge-large-en-v1.5",
    "e5":       "intfloat/e5-large-v2",
    "minilm":   "sentence-transformers/all-MiniLM-L12-v2",
}

st_train_scores: Dict[str, np.ndarray] = {}
st_test_scores:  Dict[str, np.ndarray] = {}

for model_key, model_id in ST_MODELS.items():
    try:
        train_s = compute_st_scores(train_df, model_id)
        test_s  = compute_st_scores(test_df,  model_id)
        st_train_scores[model_key] = train_s
        st_test_scores[model_key]  = test_s
        np.save(OUT_DIR / f"st_train_{model_key}.npy", train_s)
        np.save(OUT_DIR / f"st_test_{model_key}.npy",  test_s)
    except Exception as e:
        print(f"  ⚠️  {model_key} failed: {e}")
        st_train_scores[model_key] = np.zeros((len(train_df), 5))
        st_test_scores[model_key]  = np.zeros((len(test_df), 5))

# ── Cross-Encoder scoring ────────────────────────────────────────────────────
try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except ImportError:
    HAS_CE = False

CE_MODELS = {
    "ce_minilm":   "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge_rerank":  "BAAI/bge-reranker-base",
}

ce_train_scores: Dict[str, np.ndarray] = {}
ce_test_scores:  Dict[str, np.ndarray] = {}

def compute_ce_scores(df: pd.DataFrame, ce_model_id: str,
                      batch_size: int = 32) -> np.ndarray:
    """Cross-encoder scores for each (prompt, option) pair. Returns (N, 5)."""
    if not HAS_CE:
        return np.zeros((len(df), 5))
    print(f"\n🔀 Cross-Encoder: {ce_model_id}")
    ce_model = CrossEncoder(ce_model_id, device=str(DEVICE), max_length=512)
    all_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        pairs  = [(str(row["prompt"]), str(row[c])) for c in OPTION_COLS]
        scores = ce_model.predict(pairs, batch_size=batch_size,
                                  show_progress_bar=False)
        all_scores.append(scores)
    del ce_model; gc.collect()
    return np.array(all_scores)

for ce_key, ce_id in CE_MODELS.items():
    try:
        train_s = compute_ce_scores(train_df, ce_id)
        test_s  = compute_ce_scores(test_df,  ce_id)
        ce_train_scores[ce_key] = train_s
        ce_test_scores[ce_key]  = test_s
        np.save(OUT_DIR / f"ce_train_{ce_key}.npy", train_s)
        np.save(OUT_DIR / f"ce_test_{ce_key}.npy",  test_s)
    except Exception as e:
        print(f"  ⚠️  {ce_key} failed: {e}")
        ce_train_scores[ce_key] = np.zeros((len(train_df), 5))
        ce_test_scores[ce_key]  = np.zeros((len(test_df), 5))


# ─────────────────────────── CELL 9: DeBERTa FINE-TUNING ─────────────────────
"""
Fine-tune DeBERTa-v3-large as a 5-way MCQ classifier using the
Multiple-Choice Transformer head (MultipleChoiceModelOutput).

Architecture:
  Input: [CLS] prompt [SEP] option_X [SEP]  for each of 5 options
  Each pair encoded independently → pooled → linear → 5-way softmax
"""

try:
    from transformers import (AutoTokenizer, AutoModelForMultipleChoice,
                               get_cosine_schedule_with_warmup, DataCollatorWithPadding)
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  transformers / torch not found — DeBERTa fine-tuning skipped")


DEBERTA_MODEL_ID    = "microsoft/deberta-v3-base"   # use large if GPU >16GB
DEBERTA_MAX_LEN     = 192
DEBERTA_BATCH_SIZE  = 1
DEBERTA_ACCUM_STEPS = 16   # effective batch = 1*16 = 16
DEBERTA_EPOCHS      = 1
DEBERTA_LR          = 1.5e-5
DEBERTA_WARMUP_RATIO= 0.1
DEBERTA_WEIGHT_DECAY= 0.01
LABEL_SMOOTHING     = 0.1


class MCQDataset(Dataset):
    """PyTorch Dataset for 5-choice MCQ."""
    def __init__(self, df: pd.DataFrame, tokenizer,
                 max_length: int = 256, has_labels: bool = True):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row    = self.df.iloc[idx]
        prompt = str(row["prompt"])

        encodings = []
        for opt in OPTION_COLS:
            enc = self.tokenizer(
                prompt, str(row[opt]),
                max_length=self.max_length,
                truncation=True, padding="max_length",
                return_tensors="pt"
            )
            encodings.append({k: v.squeeze(0) for k, v in enc.items()})

        item = {
            "input_ids":      torch.stack([e["input_ids"]      for e in encodings]),
            "attention_mask": torch.stack([e["attention_mask"] for e in encodings]),
        }
        if "token_type_ids" in encodings[0]:
            item["token_type_ids"] = torch.stack(
                [e["token_type_ids"] for e in encodings])

        if self.has_labels:
            item["labels"] = torch.tensor(LABEL_TO_IDX[str(row["answer"])],
                                           dtype=torch.long)
        return item


def label_smoothed_ce(logits: "torch.Tensor", labels: "torch.Tensor",
                       eps: float = 0.1) -> "torch.Tensor":
    """Cross-entropy with label smoothing."""
    import torch.nn.functional as F
    n_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll_loss  = F.nll_loss(log_probs, labels, reduction="mean")
    smooth_loss = -log_probs.mean(dim=-1).mean()
    return (1 - eps) * nll_loss + eps * smooth_loss


def train_deberta(train_df: pd.DataFrame,
                  n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fine-tune DeBERTa with 5-fold CV.
    Returns:
        oof_probs  : (N_train, 5) OOF probability array
        test_probs : (N_test, 5) averaged test probability array
    """
    if not HAS_TF:
        print("⚠️  Skipping DeBERTa — transformers/torch not available")
        return np.zeros((len(train_df), 5)), np.zeros((len(test_df), 5))

    import torch
    import torch.nn as nn
    from torch.optim import AdamW

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {DEBERTA_MODEL_ID}")
    print(f"  Device: {DEVICE} | AMP: {USE_AMP}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_ID)
    label_arr = train_df["label"].values
    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_probs        = np.zeros((len(train_df), 5))
    test_probs_folds = np.zeros((len(test_df), 5, n_splits))

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(train_df, label_arr)):
        print(f"\n── Fold {fold+1}/{n_splits} ─────────────────────────────")
        tr_ds = MCQDataset(train_df.iloc[tr_idx], tokenizer, DEBERTA_MAX_LEN)
        vl_ds = MCQDataset(train_df.iloc[vl_idx], tokenizer, DEBERTA_MAX_LEN)
        te_ds = MCQDataset(test_df, tokenizer, DEBERTA_MAX_LEN, has_labels=False)

        tr_loader = DataLoader(tr_ds, batch_size=DEBERTA_BATCH_SIZE,
                               shuffle=True,  num_workers=0, pin_memory=True)
        vl_loader = DataLoader(vl_ds, batch_size=DEBERTA_BATCH_SIZE*2,
                               shuffle=False, num_workers=0)
        te_loader = DataLoader(te_ds, batch_size=DEBERTA_BATCH_SIZE*2,
                               shuffle=False, num_workers=0)

        model = AutoModelForMultipleChoice.from_pretrained(DEBERTA_MODEL_ID)
        model = model.to(DEVICE)
        try:
            model.gradient_checkpointing_enable()
            print("  ✓ Gradient checkpointing enabled for DeBERTa")
        except Exception as e:
            print(f"  ⚠️ Could not enable gradient checkpointing: {e}")

        optimizer = AdamW(model.parameters(), lr=DEBERTA_LR,
                          weight_decay=DEBERTA_WEIGHT_DECAY)
        total_steps   = len(tr_loader) // DEBERTA_ACCUM_STEPS * DEBERTA_EPOCHS
        warmup_steps  = int(total_steps * DEBERTA_WARMUP_RATIO)
        scheduler     = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)
        scaler = GradScaler() if USE_AMP else None

        best_val_map3 = 0.0
        best_state    = None

        for epoch in range(DEBERTA_EPOCHS):
            # ── Training ──────────────────────────────────────────────────────
            model.train()
            optimizer.zero_grad()
            train_loss = 0.0

            for step, batch in enumerate(tqdm(tr_loader,
                                              desc=f"Epoch {epoch+1} Train",
                                              leave=False)):
                batch = {k: v.to(DEVICE) for k, v in batch.items()
                         if k != "labels"}
                labels = batch.pop("labels", None)
                # Reconstruct labels from batch
                labels_batch = [b for b in (
                    [s["labels"] for s in tr_ds
                     if True])]  # simplified — get from loader
                # Correct approach: pass through DataLoader
                pass

            # ── Simplified epoch loop ─────────────────────────────────────────
            model.train()
            total_loss_ep = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(tr_loader,
                                              desc=f"Ep{epoch+1} Train",
                                              leave=False)):
                labels_b = batch.pop("labels").to(DEVICE)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}

                if USE_AMP:
                    with autocast():
                        outputs = model(**batch, labels=labels_b)
                        loss    = label_smoothed_ce(outputs.logits, labels_b,
                                                    LABEL_SMOOTHING)
                        loss    = loss / DEBERTA_ACCUM_STEPS
                    scaler.scale(loss).backward()
                    if (step + 1) % DEBERTA_ACCUM_STEPS == 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    outputs = model(**batch, labels=labels_b)
                    loss    = label_smoothed_ce(outputs.logits, labels_b,
                                                LABEL_SMOOTHING)
                    loss    = loss / DEBERTA_ACCUM_STEPS
                    loss.backward()
                    if (step + 1) % DEBERTA_ACCUM_STEPS == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                total_loss_ep += loss.item() * DEBERTA_ACCUM_STEPS

            # ── Validation ────────────────────────────────────────────────────
            model.eval()
            val_logits_list = []
            val_labels_list = []
            with torch.no_grad():
                for batch in tqdm(vl_loader, desc="Val", leave=False):
                    labels_b = batch.pop("labels").to(DEVICE)
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    if USE_AMP:
                        with autocast():
                            outputs = model(**batch)
                    else:
                        outputs = model(**batch)
                    val_logits_list.append(outputs.logits.float().cpu())
                    val_labels_list.append(labels_b.cpu())

            val_logits = torch.cat(val_logits_list).numpy()
            val_labels = torch.cat(val_labels_list).numpy()
            import torch.nn.functional as F
            val_probs  = torch.softmax(torch.tensor(val_logits), dim=-1).numpy()
            val_top3   = scores_to_top3(val_probs)
            val_map3   = mapk(val_labels.tolist(), val_top3)

            avg_loss = total_loss_ep / len(tr_loader)
            print(f"   Ep{epoch+1}: loss={avg_loss:.4f}  val_MAP@3={val_map3:.4f}")

            if val_map3 > best_val_map3:
                best_val_map3 = val_map3
                best_state    = deepcopy(model.state_dict())

        # ── Load best checkpoint ───────────────────────────────────────────────
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        # ── OOF predictions ────────────────────────────────────────────────────
        oof_logits_list = []
        with torch.no_grad():
            for batch in tqdm(vl_loader, desc="OOF preds", leave=False):
                batch.pop("labels", None)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                if USE_AMP:
                    with autocast():
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                oof_logits_list.append(outputs.logits.float().cpu())
        oof_logits = torch.cat(oof_logits_list).numpy()
        oof_probs_fold = torch.softmax(torch.tensor(oof_logits), dim=-1).numpy()
        oof_probs[vl_idx] = oof_probs_fold

        fold_map3 = mapk(label_arr[vl_idx].tolist(),
                         scores_to_top3(oof_probs_fold))
        print(f"  ► Fold {fold+1} MAP@3 = {fold_map3:.4f}")

        # ── Test predictions ───────────────────────────────────────────────────
        te_logits_list = []
        with torch.no_grad():
            for batch in tqdm(te_loader, desc="Test preds", leave=False):
                batch.pop("labels", None)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                if USE_AMP:
                    with autocast():
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                te_logits_list.append(outputs.logits.float().cpu())
        te_logits = torch.cat(te_logits_list).numpy()
        test_probs_folds[:, :, fold] = torch.softmax(
            torch.tensor(te_logits), dim=-1).numpy()

        # ── Save fold model ────────────────────────────────────────────────────
        model.save_pretrained(MODEL_DIR / f"deberta_fold{fold+1}")
        tokenizer.save_pretrained(MODEL_DIR / f"deberta_fold{fold+1}")

        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_probs = test_probs_folds.mean(axis=2)
    cv_map3 = mapk(label_arr.tolist(), scores_to_top3(oof_probs))
    print(f"\n🏆 DeBERTa CV MAP@3 = {cv_map3:.4f}")

    np.save(OUT_DIR / "deberta_oof_probs.npy", oof_probs)
    np.save(OUT_DIR / "deberta_test_probs.npy", test_probs)

    return oof_probs, test_probs


# Run DeBERTa fine-tuning (comment out if no GPU/time)
print("⚡ Starting DeBERTa fine-tuning ...")
deberta_oof, deberta_test = train_deberta(train_df, n_splits=5)


# ─────────────────────────── CELL 10: COLLECT ALL PREDICTIONS ────────────────
label_arr = train_df["label"].values

# ── Normalize all score matrices to [0, 1] per question ──────────────────────
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize each row (question) independently."""
    mn = scores.min(axis=1, keepdims=True)
    mx = scores.max(axis=1, keepdims=True)
    return (scores - mn) / (mx - mn + 1e-9)


# ── Collect all (train OOF, test) score arrays ────────────────────────────────
train_predictions: Dict[str, np.ndarray] = {}
test_predictions:  Dict[str, np.ndarray] = {}

# Classical ML OOF scores
for mn in ["lr", "svm", "lgbm", "xgb", "cat"]:
    if mn in classical.oof_scores:
        train_predictions[mn] = normalize_scores(classical.oof_scores[mn])
        test_predictions[mn]  = normalize_scores(classical.test_preds.get(
            f"{mn}_full", classical.test_preds[mn]))

# Sentence transformer scores
for key, arr in st_train_scores.items():
    train_predictions[f"st_{key}"] = normalize_scores(arr)
for key, arr in st_test_scores.items():
    test_predictions[f"st_{key}"] = normalize_scores(arr)

# Cross-encoder scores
for key, arr in ce_train_scores.items():
    train_predictions[f"ce_{key}"] = normalize_scores(arr)
for key, arr in ce_test_scores.items():
    test_predictions[f"ce_{key}"] = normalize_scores(arr)

# DeBERTa
if deberta_oof is not None and deberta_oof.max() > 0:
    train_predictions["deberta"] = normalize_scores(deberta_oof)
    test_predictions["deberta"]  = normalize_scores(deberta_test)

print(f"\n📦 Models collected: {list(train_predictions.keys())}")

# Print individual model CV scores
print("\n" + "="*55)
print("  INDIVIDUAL MODEL CV MAP@3")
print("="*55)
for name, preds in train_predictions.items():
    top3   = scores_to_top3(preds)
    score  = mapk(label_arr.tolist(), top3)
    print(f"  {name:<25} MAP@3 = {score:.4f}")
print("="*55)


# ─────────────────────────── CELL 11: ENSEMBLE OPTIMIZATION ──────────────────
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def weighted_ensemble(weights: np.ndarray,
                      preds_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute weighted average of normalized score arrays."""
    names = list(preds_dict.keys())
    w     = np.array([weights[i] for i in range(len(names))])
    w     = w / (w.sum() + 1e-9)
    out   = np.zeros_like(list(preds_dict.values())[0])
    for i, name in enumerate(names):
        out += w[i] * preds_dict[name]
    return out


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: maximize CV MAP@3 with weighted ensemble."""
    names = list(train_predictions.keys())
    weights = np.array([
        trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in names
    ])
    ens_preds = weighted_ensemble(weights, train_predictions)
    top3      = scores_to_top3(ens_preds)
    score     = mapk(label_arr.tolist(), top3)
    return score


print("\n🔍 Running Optuna ensemble weight optimization ...")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner()
)
N_TRIALS = 200
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_trial  = study.best_trial
best_params = best_trial.params
best_map3   = best_trial.value

print(f"\n🏆 Best ensemble MAP@3 = {best_map3:.4f}")
print(f"{'Model':<30} {'Weight':>8}")
print("-"*40)
names = list(train_predictions.keys())
weights_arr = np.array([best_params[f"w_{n}"] for n in names])
weights_arr = weights_arr / (weights_arr.sum() + 1e-9)
for name, w in sorted(zip(names, weights_arr), key=lambda x: -x[1]):
    print(f"  {name:<28} {w:>8.4f}")

# Save Optuna study plot
try:
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor="#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "optuna_study.png", facecolor="#0f0f1a", dpi=120)
    plt.close()
except Exception:
    pass


# ─────────────────────────── CELL 12: FINAL ENSEMBLE PREDICTIONS ─────────────
# ── Apply best weights to get final train and test predictions ─────────────────
final_train_scores = weighted_ensemble(
    np.array([best_params[f"w_{n}"] for n in names]),
    train_predictions
)
final_test_scores = weighted_ensemble(
    np.array([best_params[f"w_{n}"] for n in names]),
    test_predictions
)

# ── OOF MAP@3 (using best ensemble) ───────────────────────────────────────────
oof_top3  = scores_to_top3(final_train_scores)
oof_map3  = mapk(label_arr.tolist(), oof_top3)
print(f"\n✅ Final Ensemble OOF MAP@3 = {oof_map3:.4f}")


# ─────────────────────────── CELL 13: ERROR ANALYSIS ─────────────────────────
def error_analysis(df: pd.DataFrame, probs: np.ndarray,
                   labels: np.ndarray, top_k: int = 20):
    """Detailed error analysis on OOF predictions."""
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)

    top1_pred = np.argmax(probs, axis=1)
    top3_preds = scores_to_top3(probs)

    # Accuracy
    top1_acc = (top1_pred == labels).mean()
    top3_acc = np.mean([labels[i] in top3_preds[i] for i in range(len(labels))])
    print(f"  Top-1 Accuracy : {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Top-3 Accuracy : {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    print(f"  OOF MAP@3      : {oof_map3:.4f}")

    # Confidence of correct predictions
    correct_conf = [probs[i, labels[i]] for i in range(len(labels))]
    wrong_conf   = [probs[i, top1_pred[i]]
                    for i in range(len(labels)) if top1_pred[i] != labels[i]]
    print(f"\n  Avg confidence (correct preds)   : {np.mean(correct_conf):.4f}")
    if wrong_conf:
        print(f"  Avg confidence (wrong preds)     : {np.mean(wrong_conf):.4f}")

    # Most commonly wrong predictions
    errors = [(i, labels[i], top1_pred[i])
              for i in range(len(labels)) if top1_pred[i] != labels[i]]
    print(f"\n  Total errors: {len(errors)} / {len(labels)}")

    # Confusion: which label predicted as which
    conf_mat = np.zeros((5, 5), dtype=int)
    for _, true, pred in errors:
        conf_mat[true, pred] += 1

    print("\n  Confusion Matrix (true vs predicted):")
    print("  " + " ".join([f"  {c}" for c in OPTION_COLS]))
    for i, opt in enumerate(OPTION_COLS):
        row_str = "  ".join([str(conf_mat[i, j]).rjust(3) for j in range(5)])
        print(f"  {opt} | {row_str}")

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    im = ax.imshow(conf_mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(OPTION_COLS, color="#e0e0e0")
    ax.set_yticklabels(OPTION_COLS, color="#e0e0e0")
    ax.set_xlabel("Predicted", color="#e0e0e0")
    ax.set_ylabel("True",      color="#e0e0e0")
    ax.set_title("OOF Prediction Confusion Matrix",
                 color="#7c3aed", fontweight="bold")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(conf_mat[i, j]), ha="center", va="center",
                    color="white" if conf_mat[i, j] > conf_mat.max()*0.5 else "#333",
                    fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png",
                bbox_inches="tight", facecolor="#0f0f1a", dpi=120)
    plt.show()

    # Worst predictions (lowest confidence on correct answer)
    print(f"\n  Top-{top_k} Hardest Questions:")
    worst_idx = np.argsort(correct_conf)[:top_k]
    for rank, idx in enumerate(worst_idx, 1):
        prompt_trunc = str(df.iloc[idx]["prompt"])[:60]
        true_opt  = IDX_TO_LABEL[labels[idx]]
        pred_opt  = IDX_TO_LABEL[top1_pred[idx]]
        conf      = probs[idx, labels[idx]]
        print(f"  {rank:>3}. [{true_opt}→{pred_opt}] conf={conf:.3f} | {prompt_trunc}...")

error_analysis(train_df, final_train_scores, label_arr)


# ─────────────────────────── CELL 14: THRESHOLD OPTIMIZATION (MAP@3) ─────────
"""
Additional weight-search specifically for MAP@3:
For each question, find the best linear combination of model scores
that maximizes the MAP@3 metric using Optuna.
"""

def compute_mapk_from_weights(weights_dict: Dict[str, float],
                               preds_dict: Dict[str, np.ndarray],
                               labels: np.ndarray) -> float:
    names = list(preds_dict.keys())
    w     = np.array([weights_dict.get(n, 0.0) for n in names])
    w     = w / (w.sum() + 1e-9)
    scores = sum(w[i] * preds_dict[names[i]] for i in range(len(names)))
    return mapk(labels.tolist(), scores_to_top3(scores))


# ── Grid-search over a simplified alpha parameter (DeBERTa weight) ────────────
print("\n🎚️  Threshold/Alpha Optimization ...")
if "deberta" in train_predictions:
    alphas = np.linspace(0, 1, 21)
    alpha_scores = []
    for alpha in alphas:
        # Blend DeBERTa with rest
        deberta_scores = train_predictions["deberta"]
        rest_keys      = [k for k in train_predictions if k != "deberta"]
        if rest_keys:
            rest_scores = np.mean([train_predictions[k] for k in rest_keys], axis=0)
        else:
            rest_scores = deberta_scores
        blended = alpha * deberta_scores + (1 - alpha) * rest_scores
        s = mapk(label_arr.tolist(), scores_to_top3(blended))
        alpha_scores.append(s)
    best_alpha = alphas[np.argmax(alpha_scores)]
    print(f"  Best DeBERTa alpha: {best_alpha:.2f}  MAP@3={max(alpha_scores):.4f}")

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.plot(alphas, alpha_scores, "o-", color="#7c3aed", linewidth=2, markersize=5)
    ax.axvline(best_alpha, color="#f59e0b", linestyle="--", linewidth=1.5,
               label=f"Best α={best_alpha:.2f}")
    ax.set_xlabel("DeBERTa Weight (alpha)", color="#e0e0e0")
    ax.set_ylabel("MAP@3", color="#e0e0e0")
    ax.set_title("DeBERTa Alpha Sweep", color="#7c3aed", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "alpha_sweep.png", facecolor="#0f0f1a", dpi=120)
    plt.show()


# ─────────────────────────── CELL 15: GENERATE SUBMISSION ────────────────────
def generate_submission(test_df: pd.DataFrame,
                         test_scores: np.ndarray,
                         out_path: Path = OUT_DIR / "submission.csv") -> pd.DataFrame:
    """Generate final submission.csv from test score matrix."""
    predictions = labels_to_top3(test_scores)
    sub = pd.DataFrame({
        "ID":         test_df["id"].values,
        "Prediction": predictions,
    })
    sub.to_csv(out_path, index=False)
    print(f"\n✅ Submission saved → {out_path}")
    print(f"   Shape: {sub.shape}")
    print(f"\n   Preview:")
    print(sub.head(10).to_string(index=False))
    return sub

submission = generate_submission(test_df, final_test_scores,
                                  OUT_DIR / "submission.csv")

# Also save to workspace root for easy access
submission.to_csv(BASE_DIR / "submission.csv", index=False)
print(f"\n🏁 submission.csv also copied to {BASE_DIR / 'submission.csv'}")


# ─────────────────────────── CELL 16: FINAL REPORT ───────────────────────────
print("\n" + "="*70)
print("  FINAL PIPELINE REPORT")
print("="*70)

print(f"\n{'Parameter':<40} {'Value':>12}")
print("-"*54)
print(f"{'Training samples':<40} {len(train_df):>12}")
print(f"{'Test samples':<40} {len(test_df):>12}")
print(f"{'Number of options':<40} {5:>12}")
print(f"{'CV folds':<40} {5:>12}")
print(f"{'Random seed':<40} {SEED:>12}")
print(f"{'Optuna trials':<40} {N_TRIALS:>12}")

print(f"\n{'Model':<40} {'CV MAP@3':>12}")
print("-"*54)
for name, preds in sorted(train_predictions.items(),
                            key=lambda x: mapk(label_arr.tolist(),
                                               scores_to_top3(x[1])),
                            reverse=True):
    sc = mapk(label_arr.tolist(), scores_to_top3(preds))
    print(f"  {name:<38} {sc:>12.4f}")

print(f"\n  {'FINAL ENSEMBLE MAP@3':<38} {oof_map3:>12.4f}")
print("="*70)

# ── Confidence distribution ────────────────────────────────────────────────────
max_probs = final_train_scores.max(axis=1)
fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f0f1a")
ax.set_facecolor("#1a1a2e")
ax.hist(max_probs, bins=40, color="#7c3aed", edgecolor="#0f0f1a", alpha=0.85)
ax.axvline(max_probs.mean(), color="#f59e0b", linestyle="--",
           label=f"Mean={max_probs.mean():.3f}")
ax.set_title("Prediction Confidence Distribution", color="#7c3aed", fontweight="bold")
ax.set_xlabel("Max Probability Score"); ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_distribution.png", facecolor="#0f0f1a", dpi=120)
plt.show()

print("\n🎉 Pipeline complete! All outputs saved to:", OUT_DIR)
print("📄 Submission → submission.csv")
print("🔍 EDA        → eda_analysis.png")
print("📊 Confusion  → confusion_matrix.png")
print("⚡ Models     → models/")

# TODO: Support dynamic TF-IDF feature sizes based on input dataset scale
# Note: Min DF helps filter out non-informative keywords from vocabulary