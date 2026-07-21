import os
import numpy as np
import pandas as pd

BASE = Path = "/Users/shobhitagnihotri/Downloads/DL GEN AI"
OUT  = os.path.join(BASE, "outputs")

# Let's inspect the files that give 0.748/0.746 and blend them conservatively with the embeddings to push the score slightly.
# This prevents major distribution shifts while enhancing the rank signal.
sub_improved = pd.read_csv(os.path.join(OUT, "submission_improved.csv")) # scored 0.74812
sub_final = pd.read_csv(os.path.join(OUT, "submission_final.csv"))       # scored 0.74646

# Load normalized scores from models
st_te_bg = np.load(os.path.join(OUT, "st_test_bge.npy"))
st_te_mp = np.load(os.path.join(OUT, "st_test_mpnet.npy"))
ce_te_ml = np.load(os.path.join(OUT, "ce_test_ce_minilm.npy"))
ce_te_bg = np.load(os.path.join(OUT, "ce_test_bge_rerank.npy"))

def norm(a):
    mn = a.min(1, keepdims=True)
    mx = a.max(1, keepdims=True)
    return (a - mn) / (mx - mn + 1e-9)

# Construct probability matrices from the successful submissions
# Each choice (A-E) gets a score based on its rank (3 points for 1st choice, 2 for 2nd, 1 for 3rd)
OPTION_COLS = ["A", "B", "C", "D", "E"]
L2I = {c: i for i, c in enumerate(OPTION_COLS)}
I2L = {i: c for c, i in L2I.items()}

def sub_to_scores(df):
    scores = np.zeros((len(df), 5))
    for idx, row in df.iterrows():
        preds = str(row["Prediction"]).split()
        for rank, p in enumerate(preds[:3]):
            if p in L2I:
                scores[idx, L2I[p]] = 3 - rank
    return norm(scores)

sc_improved = sub_to_scores(sub_improved)
sc_final = sub_to_scores(sub_final)

# Blend them: 60% of the successful submissions (anchor) + 40% of robust embedding/cross-encoder signals
embed_sig = norm(ce_te_ml) * 0.4 + norm(ce_te_bg) * 0.4 + norm(st_te_bg) * 0.1 + norm(st_te_mp) * 0.1
blend_scores = (sc_improved * 0.5 + sc_final * 0.5) * 0.6 + norm(embed_sig) * 0.4

# Convert back to string predictions
preds = []
for r in blend_scores:
    top3_idx = np.argsort(-r)[:3]
    preds.append(" ".join([I2L[i] for i in top3_idx]))

test_df = pd.read_csv(os.path.join(BASE, "test (1).csv"))
sub = pd.DataFrame({"ID": test_df["id"].values, "Prediction": preds})
sub.to_csv(os.path.join(BASE, "submission.csv"), index=False)
sub.to_csv(os.path.join(OUT, "submission_conservative_blend.csv"), index=False)
print("🎉 Conservative Blend Submission generated!")

# Blending: Standardizes scaling before performing weighted sums