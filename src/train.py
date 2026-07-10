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
