import json
import re
from pathlib import Path

notebook_path = Path("/Users/shobhitagnihotri/Downloads/DL GEN AI/notebooks/kaggle_submission_pipeline.ipynb")
script_path = Path("/Users/shobhitagnihotri/Downloads/DL GEN AI/notebooks/kaggle_submission_pipeline.py")

with open(notebook_path, 'r') as f:
    nb = json.load(f)

code_lines = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        code_lines.append('# --- CELL --- \n')
        for line in cell['source']:
            if not line.strip().startswith('!') and not line.strip().startswith('%'):
                code_lines.append(line)
        code_lines.append('\n\n')

# Join code into a single string
code = "".join(code_lines)

# Apply modifications for a FAST local test run on CPU:
# 1. Force CPU device
code = re.sub(
    r"DEVICE\s*=\s*torch\.device\(['\"].*?['\"]\s*if\s*torch\.cuda\.is_available\(\)\s*else\s*.*?['\"]\s*if\s*torch\.backends\.mps\.is_available\(\)\s*else\s*['\"]cpu['\"]\)",
    "DEVICE = torch.device('cpu')",
    code
)
code = re.sub(
    r"DEVICE\s*=\s*torch\.device\(['\"].*?['\"]\s*if\s*torch\.cuda\.is_available\(\)\s*else\s*.*?['\"]mps['\"]\s*if\s*torch\.backends\.mps\.is_available\(\)\s*else\s*['\"]cpu['\"]\)",
    "DEVICE = torch.device('cpu')",
    code
)
code = code.replace("DEVICE = torch.device('cuda' if torch.cuda.is_available() else \n                      'mps'  if torch.backends.mps.is_available() else 'cpu')", "DEVICE = torch.device('cpu')")

# 2. Limit DeBERTa epochs to 1 and make it extremely small for local testing
code = code.replace("DEBERTA_EPOCHS = 3", "DEBERTA_EPOCHS = 1")
code = code.replace("DEBERTA_MAX_LEN = 256", "DEBERTA_MAX_LEN = 64") # shorter sequence is much faster on CPU

# 3. Limit training/testing dataset size to 50 samples for speed
dataset_limit_code = (
    "\n# Local execution speed-up modifications\n"
    "train_df = train_df.head(50).copy()\n"
    "test_df = test_df.head(10).copy()\n"
    "label_arr = train_df['label'].values\n"
)
code = code.replace("print(f\"Train set: {train_df.shape} | Test set: {test_df.shape}\")", 
                    dataset_limit_code + "print(f\"Train set: {train_df.shape} | Test set: {test_df.shape}\")")

# 4. Reduce cross-validation folds to 2 to minimize loop overhead
code = code.replace("n_splits=5", "n_splits=2")

with open(script_path, 'w') as f:
    f.write(code)

print("Fast test script written successfully to notebooks/kaggle_submission_pipeline.py!")
