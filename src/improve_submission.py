#!/usr/bin/env python3
"""
================================================================================
IMPROVED MCQ PIPELINE — Target: MAP@3 >= 0.77
================================================================================
Strategy:
  1. Cross-Encoder scoring (joint encoding, most powerful)
  2. E5-Large with proper instruction prefixes
  3. BGE-Large with proper instruction prefixes  
  4. Weighted ensemble optimized on deduplication-aware CV
================================================================================
"""
import os, gc, re, warnings, random, json
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Weights & Biases ──────────────────────────────────────────────────────────
try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False

SEED = 42
random.seed(SEED); np.random.seed(SEED)

OPTION_COLS  = ["A","B","C","D","E"]
LABEL_TO_IDX = {k:i for i,k in enumerate(OPTION_COLS)}
IDX_TO_LABEL = {i:k for k,i in LABEL_TO_IDX.items()}

BASE_DIR  = Path(__file__).resolve().parent.parent
OUT_DIR   = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# ── Load data ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv(BASE_DIR / "train (1).csv")
test_df  = pd.read_csv(BASE_DIR / "test (1).csv")
train_df.columns = [c.strip() for c in train_df.columns]
test_df.columns  = [c.strip() for c in test_df.columns]
label_arr = train_df["answer"].map(LABEL_TO_IDX).values

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── Metric ────────────────────────────────────────────────────────────────────
def apk(a, p, k=3):
    p = p[:k]; s, h = 0.0, 0
    for i, x in enumerate(p):
        if x == a and x not in p[:i]: h += 1; s += h/(i+1.0)
    return s

def mapk(a, p, k=3):
    return float(np.mean([apk(x,y,k) for x,y in zip(a,p)]))

def scores_to_top3(s):
    return [np.argsort(-r)[:3].tolist() for r in s]

def to_pred_str(s):
    return [" ".join(IDX_TO_LABEL[i] for i in np.argsort(-r)[:3]) for r in s]

def normalize(s):
    mn = s.min(axis=1, keepdims=True)
    mx = s.max(axis=1, keepdims=True)
    return (s - mn) / (mx - mn + 1e-9)

# ── Device ────────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# =============================================================================
# STEP 1: CROSS-ENCODER SCORING
# Cross-encoders jointly encode (prompt, option) → much better than bi-encoders
# =============================================================================
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

def compute_cross_encoder(df, model_id, cache_prefix, batch_size=64):
    """Score all (prompt, option) pairs with a cross-encoder."""
    cache = OUT_DIR / f"{cache_prefix}.npy"
    if cache.exists():
        arr = np.load(cache)
        print(f"  ✓ Cached: {cache_prefix}  shape={arr.shape}")
        return arr

    print(f"  Cross-encoding: {model_id}")
    ce = CrossEncoder(model_id, device=str(DEVICE), max_length=512)
    all_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=cache_prefix):
        pairs  = [(str(row["prompt"]), str(row[c])) for c in OPTION_COLS]
        scores = ce.predict(pairs, batch_size=batch_size,
                            apply_softmax=False, show_progress_bar=False)
        all_scores.append(scores)
    result = np.array(all_scores)
    np.save(cache, result)
    del ce; gc.collect()
    print(f"  ✓ Done: {cache_prefix}  shape={result.shape}")
    return result

# =============================================================================
# STEP 2: BI-ENCODER WITH PROPER INSTRUCTION PREFIXES
# E5 / BGE need specific prefixes to work correctly for retrieval
# =============================================================================
from sentence_transformers import SentenceTransformer

def compute_e5(df, cache_prefix, batch_size=32):
    """E5-Large-v2: MUST use 'query: ' and 'passage: ' prefixes."""
    cache = OUT_DIR / f"{cache_prefix}.npy"
    if cache.exists():
        arr = np.load(cache)
        print(f"  ✓ Cached: {cache_prefix}  shape={arr.shape}")
        return arr

    print(f"  E5-Large embedding with proper prefixes ...")
    model = SentenceTransformer("intfloat/e5-large-v2", device=str(DEVICE))

    # Add required prefixes for E5
    queries   = ["query: " + str(r["prompt"]) for _, r in df.iterrows()]
    passages  = ["passage: " + str(r[c]) for _, r in df.iterrows() for c in OPTION_COLS]

    q_embs = model.encode(queries,  batch_size=batch_size,
                           show_progress_bar=True, normalize_embeddings=True)
    p_embs = model.encode(passages, batch_size=batch_size,
                           show_progress_bar=True, normalize_embeddings=True)
    p_3d   = p_embs.reshape(len(df), 5, -1)
    scores = np.einsum("nd,nkd->nk", q_embs, p_3d)
    np.save(cache, scores)
    del model; gc.collect()
    print(f"  ✓ Done: {cache_prefix}  shape={scores.shape}")
    return scores


def compute_bge(df, cache_prefix, batch_size=32):
    """BGE-Large: MUST add instruction prefix to queries."""
    cache = OUT_DIR / f"{cache_prefix}.npy"
    if cache.exists():
        arr = np.load(cache)
        print(f"  ✓ Cached: {cache_prefix}  shape={arr.shape}")
        return arr

    print(f"  BGE-Large embedding with instruction prefix ...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=str(DEVICE))

    # BGE instruction prefix for retrieval
    instruction = "Represent this sentence for searching relevant passages: "
    queries  = [instruction + str(r["prompt"]) for _, r in df.iterrows()]
    passages = [str(r[c]) for _, r in df.iterrows() for c in OPTION_COLS]

    q_embs = model.encode(queries,  batch_size=batch_size,
                           show_progress_bar=True, normalize_embeddings=True)
    p_embs = model.encode(passages, batch_size=batch_size,
                           show_progress_bar=True, normalize_embeddings=True)
    p_3d   = p_embs.reshape(len(df), 5, -1)
    scores = np.einsum("nd,nkd->nk", q_embs, p_3d)
    np.save(cache, scores)
    del model; gc.collect()
    print(f"  ✓ Done: {cache_prefix}  shape={scores.shape}")
    return scores


# =============================================================================
# STEP 3: OPTION-AWARE FEATURE — Longest option wins
# Since correct answers are 13% longer, use this as calibration
# =============================================================================
def compute_length_signal(df):
    """
    Per-question: score each option by its length relative to the mean.
    Correct answers are empirically longer → useful ranking signal.
    """
    scores = np.zeros((len(df), 5))
    for i, (_, row) in enumerate(df.iterrows()):
        lens = np.array([len(str(row[c])) for c in OPTION_COLS], dtype=float)
        # Softmax-style normalization
        lens = lens / (lens.sum() + 1e-9)
        scores[i] = lens
    return scores


# =============================================================================
# RUN ALL MODELS
# =============================================================================
if USE_WANDB:
    wandb.init(
        project="mcq-competition-ensemble",
        config={
            "seed": SEED,
            "train_size": len(train_df),
            "test_size": len(test_df),
        }
    )

all_oof  = {}
all_test = {}

# ── Load existing classical ML OOF (already computed) ─────────────────────────
print("\n[1] Loading existing classical ML predictions ...")
for mn in ["lgbm", "xgb", "cat"]:
    oo = OUT_DIR / f"oof_{mn}.npy"
    te = OUT_DIR / f"test_{mn}.npy"
    if oo.exists() and te.exists():
        all_oof[mn]  = normalize(np.load(oo))
        all_test[mn] = normalize(np.load(te))
        sc = mapk(label_arr.tolist(), scores_to_top3(all_oof[mn]))
        print(f"  {mn}: OOF MAP@3 = {sc:.4f}  (inflated due to train memorization)")
        if USE_WANDB:
            wandb.log({f"base_model/{mn}_map3_inflated": sc})

# ── Length signal ─────────────────────────────────────────────────────────────
print("\n[2] Computing length signal ...")
len_train = compute_length_signal(train_df)
len_test  = compute_length_signal(test_df)
all_oof["length"]  = normalize(len_train)
all_test["length"] = normalize(len_test)
sc = mapk(label_arr.tolist(), scores_to_top3(len_train))
print(f"  Length signal MAP@3 = {sc:.4f}")
if USE_WANDB:
    wandb.log({"base_model/length_signal_map3": sc})

# ── Cross-Encoders ────────────────────────────────────────────────────────────
print("\n[3] Cross-Encoder models ...")
CE_MODELS = [
    ("ce_minilm",   "cross-encoder/ms-marco-MiniLM-L-12-v2"),
    ("bge_rerank",  "BAAI/bge-reranker-base"),
]
for key, model_id in CE_MODELS:
    try:
        s_tr = compute_cross_encoder(train_df, model_id, f"ce_train_{key}")
        s_te = compute_cross_encoder(test_df,  model_id, f"ce_test_{key}")
        all_oof[key]  = normalize(s_tr)
        all_test[key] = normalize(s_te)
        sc = mapk(label_arr.tolist(), scores_to_top3(all_oof[key]))
        print(f"  {key} MAP@3 = {sc:.4f}")
        if USE_WANDB:
            wandb.log({f"base_model/{key}_map3": sc})
    except Exception as e:
        print(f"  ⚠️  {key}: {e}")

# ── E5-Large with proper prefixes ─────────────────────────────────────────────
print("\n[4] E5-Large with proper 'query:'/'passage:' prefixes ...")
try:
    s_tr = compute_e5(train_df, "e5_train")
    s_te = compute_e5(test_df,  "e5_test")
    all_oof["e5"]  = normalize(s_tr)
    all_test["e5"] = normalize(s_te)
    sc = mapk(label_arr.tolist(), scores_to_top3(all_oof["e5"]))
    print(f"  E5-Large MAP@3 = {sc:.4f}")
    if USE_WANDB:
        wandb.log({"base_model/e5_map3": sc})
except Exception as e:
    print(f"  ⚠️  E5: {e}")

# ── BGE-Large with instruction prefix ─────────────────────────────────────────
print("\n[5] BGE-Large with instruction prefix ...")
try:
    s_tr = compute_bge(train_df, "bge_train")
    s_te = compute_bge(test_df,  "bge_test")
    all_oof["bge"]  = normalize(s_tr)
    all_test["bge"] = normalize(s_te)
    sc = mapk(label_arr.tolist(), scores_to_top3(all_oof["bge"]))
    print(f"  BGE-Large MAP@3 = {sc:.4f}")
    if USE_WANDB:
        wandb.log({"base_model/bge_map3": sc})
except Exception as e:
    print(f"  ⚠️  BGE: {e}")

# =============================================================================
# STEP 4: SMART ENSEMBLE
# Use deduplication-aware evaluation: only score on non-duplicate prompts
# to get a more honest signal for weighting
# =============================================================================
print("\n[6] Ensemble optimization ...")

# Filter to non-duplicate prompts for honest evaluation
mask_nodup = ~train_df["prompt"].duplicated(keep=False)
labels_nd  = label_arr[mask_nodup]
all_oof_nd = {k: v[mask_nodup] for k, v in all_oof.items()}

print(f"Non-duplicate samples: {mask_nodup.sum()} / {len(train_df)}")
print("\nHonest MAP@3 scores (no duplicate leakage):")
print(f"{'Model':<25} {'All':>8} {'No-dup':>8}")
print("-"*44)
for n, preds in all_oof.items():
    sc_all = mapk(label_arr.tolist(), scores_to_top3(preds))
    sc_nd  = mapk(labels_nd.tolist(), scores_to_top3(all_oof_nd[n]))
    print(f"  {n:<23} {sc_all:>8.4f} {sc_nd:>8.4f}")

MODEL_NAMES = list(all_oof.keys())

def objective_nd(trial):
    """Optimize on non-duplicate samples for honest estimate."""
    w = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in MODEL_NAMES])
    w = w / (w.sum() + 1e-9)
    ens = sum(w[i] * all_oof_nd[MODEL_NAMES[i]] for i in range(len(MODEL_NAMES)))
    val = mapk(labels_nd.tolist(), scores_to_top3(ens))
    if USE_WANDB:
        wandb.log({
            "trial/map3_no_dup": val,
            "trial/number": trial.number,
            **{f"trial_weight/{n}": w[idx] for idx, n in enumerate(MODEL_NAMES)}
        })
    return val

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective_nd, n_trials=200, show_progress_bar=True)

bp     = study.best_trial.params
raw_w  = np.array([bp[f"w_{n}"] for n in MODEL_NAMES])
norm_w = raw_w / (raw_w.sum() + 1e-9)
best   = study.best_trial.value

final_oof  = sum(norm_w[i] * all_oof[MODEL_NAMES[i]]  for i in range(len(MODEL_NAMES)))
final_test = sum(norm_w[i] * all_test[MODEL_NAMES[i]] for i in range(len(MODEL_NAMES)))

print(f"\n🏆 Optuna Ensemble MAP@3 (no-dup) = {best:.4f}")
print(f"\n{'Model':<25} {'Weight':>8}   {'MAP@3 (nd)':>10}")
print("-"*47)
for i, n in enumerate(MODEL_NAMES):
    sc_nd = mapk(labels_nd.tolist(), scores_to_top3(all_oof_nd[n]))
    print(f"  {n:<23} {norm_w[i]:>8.4f}   {sc_nd:>10.4f}")

# =============================================================================
# STEP 5: GENERATE SUBMISSION
# =============================================================================
predictions = to_pred_str(final_test)
sub = pd.DataFrame({"ID": test_df["id"].values, "Prediction": predictions})
sub.to_csv(BASE_DIR / "submission.csv", index=False)
sub.to_csv(OUT_DIR  / "submission_improved.csv", index=False)

# Save weights
weight_dict = {n: float(norm_w[i]) for i, n in enumerate(MODEL_NAMES)}
with open(OUT_DIR / "best_weights_v2.json", "w") as f:
    json.dump({"weights": weight_dict, "cv_map3_no_dup": best}, f, indent=2)

print(f"\n{'='*60}")
print("  FINAL SUBMISSION")
print(f"{'='*60}")
print(f"  Rows: {len(sub)}")
print(sub.head(15).to_string(index=False))

# Validate
assert len(sub) == len(test_df)
assert all(sub["Prediction"].str.split().str.len() == 3)
print("\n✅ submission.csv validated and saved!")
print(f"   → {BASE_DIR}/submission.csv")

if USE_WANDB:
    # Log final results
    wandb.log({
        "final/ensemble_map3_no_dup": best,
        **{f"final_weight/{n}": norm_w[idx] for idx, n in enumerate(MODEL_NAMES)}
    })
    # Save best weights as artifact
    try:
        artifact = wandb.Artifact("ensemble-weights", type="model")
        artifact.add_file(str(OUT_DIR / "best_weights_v2.json"))
        wandb.log_artifact(artifact)
    except Exception as e:
        print(f"  ⚠️ Could not upload wandb artifact: {e}")
    wandb.finish()

# Formatting: Added visual separator lines to optimization logs