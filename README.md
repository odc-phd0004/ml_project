# Network Flow Classification under Zero-Metadata Constraints

Application classification of encrypted mobile network flows (Facebook, Google, Instagram, Mozilla, X, YouTube)
from header-derived flow statistics only: no IP addresses, DNS names, TLS SNI, URLs or payload.
We compare nine tuned classical learners, two class-imbalance remedies and two pretrained tabular
foundation models (TabPFN, TabFM) on the ITU *Application Inference from Packet Flows* dataset (Task 1, full flow).

Authors: Masabattula Teja Nikhil, Hrusikesh Nishank, Nishu Kumar (CSE, IIT Jodhpur)

---

## 1. Repository layout

The notebooks load data with the relative path `../dataset/task1_static/...`, so they must sit one level
below the repository root, next to `dataset/`:

```
.
├── README.md
├── environment.yaml                 # main environment (classical models + TabFM)
├── verify_slides.py                 # checks every reported number against the notebook outputs
├── dataset/
│   └── task1_static/
│       ├── train_features.csv       # 247 flows
│       ├── train_labels.csv
│       ├── validation_features.csv  # 83 flows
│       └── validation_labels.csv
├── notebooks/
│   ├── 01_logistic_regression.ipynb
│   ├── 02_knn.ipynb
│   ├── 03_svm.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_bagging.ipynb
│   ├── 06_boosting.ipynb            # HistGradientBoosting
│   ├── 07_adaboost.ipynb
│   ├── 08_xgboost.ipynb
│   ├── 08_tabfm.ipynb               # needs a CUDA GPU + Hugging Face login
│   ├── 09_random_forest.ipynb
│   ├── 09_tabpfn.ipynb              # needs the separate TabPFN environment
│   └── 10_google_youtube_confusion.ipynb
```

### Getting the data

The data come from the public challenge repository
<https://github.com/FG-AI4H/application-inference-packet-flows-public>. Copy its Task 1 files
(`dataset/task1_static/`) into `dataset/task1_static/` here. Test-split labels are held by the challenge
organisers, so **all results in this project are reported on the validation split.**

---

## 2. Environments

Two environments are required because TabPFN 2.0.9 pins `scikit-learn<1.7` and `pandas<3`, while the
classical results were produced with scikit-learn 1.9.1.

| Environment | Used for | Key versions |
|---|---|---|
| `ml_project` (`environment.yaml`) | notebooks 01–08_xgboost, 09_random_forest, 10, and 08_tabfm | Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, XGBoost 3.4.1, imbalanced-learn 0.14.2, TabFM 1.0.1 |
| `tabpfn` | 09_tabpfn only | Python 3.13, **tabpfn 2.0.9**, scikit-learn 1.6.1, pandas 2.3.3, torch 2.15.0 |

### 2.1 Main environment

```bash
conda env create -f environment.yaml
conda activate ml_project
python -m ipykernel install --user --name ml_project
```

### 2.2 TabPFN environment

```bash
conda create -n tabpfn python=3.13 pip -y
conda activate tabpfn
# CPU build of torch is enough; TabPFN runs on CPU here
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install "tabpfn==2.0.9" "scikit-learn==1.6.1" "pandas==2.3.3" "numpy==2.5.5" \
            "matplotlib==3.11.2" "seaborn==0.13.2" ipykernel nbconvert
python -m ipykernel install --user --name tabpfn
```

### 2.3 TabFM requirements

* A **CUDA GPU**: the notebook loads the model with `device="cuda"`. The v1.0.0 checkpoint has about 1.6B
  parameters and runs in bfloat16. It ran out of memory on a 7 GB CPU-only machine.
* A **Hugging Face account**: cell 3 calls `huggingface_hub.login()`. Weights are downloaded from
  `google/tabfm-1.0.0-pytorch`.

---

## 3. Running the experiments

Every notebook is self-contained: it reads the CSVs, runs its experiments and writes its result files into
`notebooks/`. The notebooks do not depend on each other, so the order is free. All randomness is seeded
(`random_state = 42`; stratified 5-fold CV with `shuffle=True, random_state=42`), so repeated runs give
identical numbers.

### 3.1 Interactively

Open JupyterLab, select the kernel listed below for each notebook, and use **Kernel → Restart Kernel and Run All Cells**.

### 3.2 Headless (classical models and TabPFN)

```bash
cd notebooks

# main environment
conda activate ml_project
for nb in 01_logistic_regression 02_knn 03_svm 04_decision_tree 05_bagging \
          06_boosting 07_adaboost 08_xgboost 09_random_forest 10_google_youtube_confusion; do
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 \
      --ExecutePreprocessor.kernel_name=ml_project "$nb.ipynb"
done

# TabPFN environment
conda activate tabpfn
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=tabpfn 09_tabpfn.ipynb
```

### 3.3 TabFM (interactive, on the GPU machine)

`08_tabfm.ipynb` prompts for a Hugging Face login, so run it interactively in JupyterLab with the
`ml_project` kernel. **Run all cells including the last one ("Full metrics for the slides")**,
which produces the balanced accuracy, weighted F1 and AUC reported for TabFM.