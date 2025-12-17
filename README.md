# Privacy-Preserving Credit Risk Modeling with Synthetic Data

## Project Overview

This project develops a three-stage pipeline for credit risk assessment in India's digital lending market while ensuring complete privacy compliance with DPDPA 2023 and RBI guidelines.

**Stages:**
1. **Synthetic Data Generation** - Generate high-fidelity credit datasets using Generative AI
2. **Unsupervised Risk Profiling** - Discover natural borrower segments via clustering
3. **Risk-Adjusted Deep RL** - Learn optimal lending policies through reinforcement learning

## Quick Start - Hybrid Approach (Recommended)

The hybrid approach combines Home-Credit-Risk dataset structure with RBI/Indian market priors:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Home-Credit-Risk dataset (optional but recommended)
python src/download_kaggle.py

# 3. Update RBI priors (edit config/priors_template.yaml)

# 4. Generate synthetic data
python run_hybrid_approach.py

# OR use the notebook
jupyter notebook notebooks/01_hybrid_synthetic_generation.ipynb
```

See `HYBRID_APPROACH_GUIDE.md` for detailed steps.

## Data Download Options

### ✅ Easy Downloads (Do This First)

**Option 1: Home_credit_default_risk** (Recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Download Home Credit Default Risk (main dataset)
kaggle competitions download -c home-credit-default-risk

# Extract
unzip home-credit-default-risk.zip -d data/
```

**Option 2: UCI Repository** (Quick baseline)
```bash
# Run download script
python download_datasets.py
```

**Option 3: Generate from Scratch** (Most privacy-preserving)
- Use statistical priors from RBI publications
- Generate synthetic data directly using SDV
- No real data download needed!

### ⚠️ Realistic Expectations

**Directly Downloadable:**
- ✅ Kaggle datasets (requires free account)
- ✅ UCI ML Repository datasets
- ✅ Some data.gov.in datasets (varies)

**Requires Manual Work:**
- ⚠️ RBI DBIE - Interactive database, CSV export per query
- ⚠️ NPCI Statistics - Dashboard only, no raw downloads
- ⚠️ Census data - Excel files with multiple sheets

**Reference Only:**
- 📄 RBI Publications (PDFs) - Extract statistics manually
- 📄 MFIN Reports (PDFs) - For microfinance priors

## Project Structure

```
ProjectSem5/
├── data/                    # Downloaded datasets
├── notebooks/               # Jupyter notebooks for each stage
│   ├── 01_synthetic_gen.ipynb
│   ├── 02_clustering.ipynb
│   └── 03_rl_policy.ipynb
├── src/                     # Source code modules
├── config/                  # Configuration files
├── download_datasets.py     # Dataset download script
├── DATASET_DOWNLOAD_GUIDE.md  # Detailed download guide
└── README.md               # This file
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Review-1 Deliverables

- [ ] Synthetic dataset (v0.1) with data card
- [ ] Unsupervised clustering model (k-prototypes/GMM)
- [ ] Basic RL policy (contextual bandit)
- [ ] Privacy compliance memo
- [ ] Three notebooks (one per stage)

## Synthetic outputs and results (not in git)

To keep the repository light and avoid committing large files, all **synthetic datasets** and **visualisation outputs** are ignored via `.gitignore`:

- `data/*.parquet`, `data/*.csv` – generated synthetic credit datasets (RBI priors-only, Home-Credit-Risk-only, Hybrid, etc.)
- `results/*.png`, `results/*.csv`, `results/*.parquet`, `results/*.pptx` – clustering plots, evaluation reports, and the auto-generated `cluster_analysis.pptx`.

You can regenerate these locally by running:

```bash
# Hybrid CTGAN synthetic data (~200k rows)
python run_hybrid_ctgan_pipeline.py

# RBI priors-only and Home-Credit-Risk-only synthetic datasets
python run_priors_only_rbi_synthetic.py
python run_kaggle_only_ctgan_300k.py

# Risk-based clustering on all synthetic datasets
python run_credit_risk_clustering_all.py

# Visualisations (cluster profiles, heatmaps, PCA, etc.)
python run_visualize_risk_clusters_all.py
python run_advanced_cluster_visualizations.py

# Optional: build PowerPoint summary under results/cluster_analysis.pptx
python build_cluster_analysis_ppt.py
```

## Contact

For questions about data sources or project structure, refer to `DATASET_DOWNLOAD_GUIDE.md`.

