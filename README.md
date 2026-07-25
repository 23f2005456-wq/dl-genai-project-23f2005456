# 🏆 MCQ Competition — Winning Pipeline

> **Task**: 5-choice Multiple-Choice Question Answering  
> **Metric**: Mean Average Precision @ 3 (MAP@3)  
> **Dataset**: 2001 train / 501 test rows, Science/Physics topics
> **Status**: Pipeline completed, inference finalized.

---

## 📁 File Structure

```
DL GEN AI/
├── notebooks/
│   └── mcq_pipeline.ipynb       # Jupyter notebook version
├── src/
│   ├── train.py                 # Main training pipeline (run this first)
│   ├── inference.py             # Standalone inference script
│   └── improve_submission.py    # Optuna-based ensembling weight optimization
├── models/                      # Trained model checkpoints
├── outputs/                     # EDA plots, score matrices, submissions
├── reports/                     # Reports folder
├── requirements.txt             # All dependencies
└── README.md                    # This file
```
    ├── lr_full_model.joblib
    ├── svm_full_model.joblib
    ├── lgbm_full_model.joblib
    ├── xgb_full_model.joblib
    ├── cat_full_model.joblib
    ├── deberta_fold1/       # DeBERTa fine-tuned fold checkpoints
    ├── deberta_fold2/
    ├── deberta_fold3/
    ├── deberta_fold4/
    └── deberta_fold5/
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Download spacy model (optional, for NLP features)
python -m spacy download en_core_web_sm
# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Run the full training pipeline

```bash
python src/train.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook notebooks/mcq_pipeline.ipynb
```

### 3. Run inference only (after training)

```bash
python src/inference.py \
    --test_path "test (1).csv" \
    --model_dir models \
    --output submission.csv
```

---

## 🏗️ Architecture

```
Raw Data
   │
   ├── EDA & Visualization
   │
   ├── Feature Engineering
   │   ├── TF-IDF cosine similarity (prompt vs each option)
   │   ├── BM25 retrieval scores
   │   ├── N-gram (unigram / bigram / trigram) Jaccard overlap
   │   ├── Keyword overlap (non-stopwords)
   │   ├── Length features (chars, words, ratios)
   │   ├── Structural features (position, starts-with-number)
   │   └── Readability features (Flesch, Kincaid, SMOG)
   │
   ├── Classical Models (5-Fold CV)
   │   ├── TF-IDF + Logistic Regression
   │   ├── TF-IDF + Linear SVM (calibrated)
   │   ├── LightGBM
   │   ├── XGBoost
   │   └── CatBoost
   │
   ├── Bi-Encoder Embeddings
   │   ├── all-mpnet-base-v2
   │   ├── BAAI/bge-large-en-v1.5        ← Best for retrieval
   │   ├── intfloat/e5-large-v2          ← Strong general embedder
   │   └── all-MiniLM-L12-v2             ← Fast baseline
   │
   ├── Cross-Encoders
   │   ├── cross-encoder/ms-marco-MiniLM-L-12-v2
   │   └── BAAI/bge-reranker-base
   │
   └── DeBERTa-v3 Fine-tuned (MCQ head, 5-fold CV)
           ├── Mixed precision (FP16)
           ├── Gradient accumulation (effective batch=32)
           ├── Cosine LR scheduler + warmup
           ├── Label smoothing (ε=0.1)
           └── Early stopping on MAP@3
   │
   ├── Ensemble Optimization (Optuna, 200 trials)
   │   └── Best weighted average of all model scores
   │
   └── Submission (submission.csv)
```

---

## 📊 Expected Results

| Model | Expected MAP@3 |
|-------|---------------|
| TF-IDF + LR (baseline) | ~0.55 |
| LightGBM on features | ~0.60 |
| MPNet embeddings | ~0.70 |
| BGE-Large embeddings | ~0.75 |
| DeBERTa fine-tuned | ~0.85+ |
| **Full Ensemble** | **~0.87–0.92** |

> Results depend on hardware (GPU strongly recommended for DeBERTa) and training time.

---

## ⚙️ Configuration

Key hyperparameters in `mcq_pipeline.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEED` | 42 | Random seed for reproducibility |
| `DEBERTA_MODEL_ID` | `deberta-v3-base` | Use `deberta-v3-large` for best results |
| `DEBERTA_MAX_LEN` | 256 | Max token length per pair |
| `DEBERTA_BATCH_SIZE` | 4 | Per-device batch size |
| `DEBERTA_ACCUM_STEPS` | 8 | Gradient accumulation (effective batch=32) |
| `DEBERTA_EPOCHS` | 3 | Training epochs |
| `DEBERTA_LR` | 2e-5 | Learning rate |
| `LABEL_SMOOTHING` | 0.1 | Label smoothing epsilon |
| `N_TRIALS` | 200 | Optuna ensemble optimization trials |

---

## 🔬 MAP@3 Explained

```
If correct answer = A:

Prediction "A B C" → score = 1/1 = 1.000  (correct at rank 1)
Prediction "B A C" → score = 1/2 = 0.500  (correct at rank 2)
Prediction "C D A" → score = 1/3 = 0.333  (correct at rank 3)
Prediction "B C D" → score = 0.000        (not in top 3)

MAP@3 = mean of all AP@3 scores
```

---

## 💡 Key Tricks for High MAP@3

1. **DeBERTa with Multiple Choice Head** — Best single model
2. **BGE-Large embeddings** — Strong pre-trained retrieval signal  
3. **Optuna ensemble optimization** — Finds optimal weights automatically
4. **Label smoothing** — Prevents overconfident predictions
5. **5-Fold stratified CV** — Robust OOF estimates for ensembling
6. **Gradient accumulation** — Effective large batch without OOM
7. **Cosine LR scheduler** — Better convergence than constant LR
8. **Score normalization per question** — Ensures fair model combination
9. **BM25 retrieval scores** — Strong lexical baseline
10. **Position bias feature** — Captures dataset-level biases

---

## 📦 Hardware Requirements

| Configuration | Estimated Runtime |
|--------------|------------------|
| CPU only (classical models only) | ~30 min |
| CPU + sentence transformers | ~2 hours |
| GPU 8GB (all models, deberta-v3-base) | ~4-6 hours |
| GPU 16GB+ (deberta-v3-large) | ~8-12 hours |

---

## 📄 Output Files

- `submission.csv` — Final Kaggle submission (ID + Prediction)
- `outputs/eda_analysis.png` — 11-panel EDA dashboard
- `outputs/confusion_matrix.png` — OOF confusion matrix
- `outputs/confidence_distribution.png` — Prediction confidence histogram
- `outputs/feat_importance_*.png` — Feature importance plots
- `models/` — All trained model checkpoints

---

## ── Reproducing Results

```bash
# Set seed (already default=42 in config)
python src/train.py

# The script prints per-fold MAP@3, CV MAP@3, and final ensemble MAP@3
```
