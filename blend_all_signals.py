import pandas as pd
import numpy as np
from pathlib import Path

OPTS = list("ABCDE")
L2I = {c: i for i, c in enumerate(OPTS)}
I2L = {i: c for c, i in L2I.items()}

BASE_DIR = Path("/Users/shobhitagnihotri/Desktop/DL GEN AI")
DOWNLOADS_DIR = Path("/Users/shobhitagnihotri/Downloads")

# Load files
f_self   = pd.read_csv(DOWNLOADS_DIR / "submission_self_consistency.csv")
f_cv     = pd.read_csv(DOWNLOADS_DIR / "submission_cv_final.csv")
f_anchor = pd.read_csv(BASE_DIR / "submission21.csv")

def get_scores(df, weight):
    pred_col = 'prediction' if 'prediction' in df.columns else 'Prediction'
    id_col = 'id' if 'id' in df.columns else 'ID'
    scores_dict = {}
    for idx, row in df.iterrows():
        qid = row[id_col]
        preds = str(row[pred_col]).split()
        sc = np.zeros(5)
        if len(preds) >= 1 and preds[0] in L2I: sc[L2I[preds[0]]] += 3.0 * weight
        if len(preds) >= 2 and preds[1] in L2I: sc[L2I[preds[1]]] += 2.0 * weight
        if len(preds) >= 3 and preds[2] in L2I: sc[L2I[preds[2]]] += 1.0 * weight
        scores_dict[qid] = sc
    return scores_dict

# Weights: 50% GPT-4o self-consistency, 25% 5-fold CV generalizer, 25% 0.76302 anchor
sc_self   = get_scores(f_self, 0.50)
sc_cv     = get_scores(f_cv, 0.25)
sc_anchor = get_scores(f_anchor, 0.25)

id_col = 'id' if 'id' in f_self.columns else 'ID'
qids = f_self[id_col].values

final_preds = []
for qid in qids:
    total_sc = sc_self[qid] + sc_cv[qid] + sc_anchor[qid]
    top3_idx = np.argsort(-total_sc)[:3]
    final_preds.append(" ".join([I2L[i] for i in top3_idx]))

out_df = pd.DataFrame({"id": qids, "prediction": final_preds})

# Save in both DL GEN AI folder and Downloads
out_path_1 = BASE_DIR / "submission_ultra_blend_0.76.csv"
out_path_2 = DOWNLOADS_DIR / "submission_ultra_blend_0.76.csv"

out_df.to_csv(out_path_1, index=False)
out_df.to_csv(out_path_2, index=False)

print(f"🎉 Saved Ultra Blend submission to:\n  1. {out_path_1}\n  2. {out_path_2}")
print(out_df.head(10))

# Output: Enhanced logs with standard deviation metrics