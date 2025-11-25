# Hybrid Approach - Step-by-Step Guide

## Overview

The hybrid approach combines:
1. **Kaggle dataset structure** → Learn realistic patterns
2. **RBI/Indian priors** → Calibrate for Indian market
3. **Synthetic generation** → Create privacy-preserving data

---

## Step-by-Step Execution

### **Phase 1: Setup (Day 1)**

#### 1.1 Install Dependencies
```bash
cd /Users/ashishasharma/Desktop/ProjectSem5
pip install -r requirements.txt
```

#### 1.2 Download Kaggle Dataset
```bash
# Option A: Use download script
python src/download_kaggle.py

# Option B: Manual download
# 1. Go to https://www.kaggle.com/c/home-credit-default-risk/data
# 2. Create free account
# 3. Accept competition rules
# 4. Click "Download All"
# 5. Extract to data/kaggle_home_credit/
```

**Expected time:** 10-15 minutes

#### 1.3 Extract RBI Priors (Manual Work)

1. **Download RBI Publications:**
   - Latest "Financial Stability Report"
   - "Trends and Progress in Banking"
   - Available at: https://www.rbi.org.in/Scripts/PublicationsView.aspx

2. **Extract Key Statistics:**
   - Default rates (overall, by product type)
   - Interest rate ranges
   - NPA ratios
   - Loan-to-income ratios

3. **Update Config File:**
   ```bash
   # Edit config/priors_template.yaml
   # Replace placeholder values with actual RBI statistics
   ```

**Expected time:** 2-3 hours

---

### **Phase 2: Generate Synthetic Data (Day 2-3)**

#### 2.1 Run Notebook
```bash
# Open Jupyter
jupyter notebook notebooks/01_hybrid_synthetic_generation.ipynb

# OR use VS Code/Cursor to open and run cells
```

#### 2.2 Follow Notebook Steps

The notebook will:
1. ✅ Load Kaggle dataset (if downloaded)
2. ✅ Load RBI priors
3. ✅ Initialize hybrid generator
4. ✅ Generate synthetic data
5. ✅ Evaluate quality
6. ✅ Save dataset

#### 2.3 Expected Output

- `data/synthetic_credit_data_v0.1.parquet` - Main dataset
- `data/synthetic_credit_data_v0.1.csv` - CSV version (for inspection)
- `data/data_card.yaml` - Documentation

#### 2.4 Verify Quality

Check:
- ✅ Default rate matches RBI priors (~2.5%)
- ✅ Income distributions match Indian market
- ✅ Age distributions are realistic
- ✅ No missing critical columns

---

### **Phase 3: Refinement (Optional, Day 4)**

#### 3.1 Improve Quality (If Needed)

**Option A: Use CTGAN (Better Quality)**
```python
# In notebook, change:
synthetic_data = generator.generate_synthetic(
    n_samples=10000,
    method='ctgan',  # Better quality, slower
    ...
)
```

**Option B: Adjust Priors**
- Review `config/priors_template.yaml`
- Update with more accurate RBI statistics
- Regenerate synthetic data

**Option C: Increase Sample Size**
```python
synthetic_data = generator.generate_synthetic(
    n_samples=50000,  # More samples
    ...
)
```

---

## Quick Start Commands

### One-Line Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download Kaggle dataset (if you have Kaggle credentials)
python src/download_kaggle.py

# Generate synthetic data
python src/hybrid_synthetic_generator.py
```

### Manual Workflow
```bash
# 1. Download Kaggle manually (if script doesn't work)
#    → https://www.kaggle.com/c/home-credit-default-risk/data

# 2. Extract RBI priors manually
#    → Download RBI reports
#    → Update config/priors_template.yaml

# 3. Run notebook
jupyter notebook notebooks/01_hybrid_synthetic_generation.ipynb
```

---

## File Structure

```
ProjectSem5/
├── data/
│   ├── kaggle_home_credit/          # Kaggle dataset (if downloaded)
│   │   └── application_train.csv
│   ├── synthetic_credit_data_v0.1.parquet  # Generated synthetic data
│   └── data_card.yaml              # Dataset documentation
│
├── config/
│   └── priors_template.yaml        # RBI priors (UPDATE THIS!)
│
├── src/
│   ├── download_kaggle.py          # Kaggle download helper
│   └── hybrid_synthetic_generator.py  # Main generator
│
└── notebooks/
    └── 01_hybrid_synthetic_generation.ipynb  # Main workflow
```

---

## Troubleshooting

### Issue: Kaggle download fails
**Solution:**
- Check Kaggle credentials: `~/.kaggle/kaggle.json`
- Accept competition rules manually
- Download manually from website

### Issue: No Kaggle dataset available
**Solution:**
- Not a problem! Generator works with priors only
- Set `use_kaggle_structure=False` in notebook
- Still generates valid synthetic data

### Issue: Synthetic data quality poor
**Solution:**
1. Update RBI priors with accurate values
2. Try CTGAN instead of Gaussian Copula
3. Increase sample size
4. Check reference data quality

### Issue: Missing dependencies
**Solution:**
```bash
pip install sdv pandas numpy scikit-learn
```

---

## Next Steps After Synthetic Data Generation

Once you have synthetic data:

1. **Stage 2: Clustering**
   - Use `notebooks/02_clustering.ipynb` (to be created)
   - Cluster synthetic borrowers into risk profiles

2. **Stage 3: RL Policy**
   - Use `notebooks/03_rl_policy.ipynb` (to be created)
   - Train RL agent on synthetic environment

---

## Tips for Review-1

1. **Start Simple:**
   - Use Gaussian Copula (faster)
   - Generate 10K samples (sufficient for demo)
   - Focus on core features

2. **Document Everything:**
   - Update `config/priors_template.yaml` with sources
   - Fill out `data_card.yaml` completely
   - Save notebook outputs

3. **Show Both Approaches:**
   - Show synthetic data WITH Kaggle structure
   - Show synthetic data FROM PRIORS ONLY
   - Compare quality

4. **Privacy Emphasis:**
   - Highlight: No real data used
   - Mention: DPDPA 2023 compliance
   - Show: Statistical properties preserved, privacy maintained

---

## Success Criteria

✅ **Review-1 Ready When:**
- [ ] Synthetic dataset generated (10K+ rows)
- [ ] Data card completed
- [ ] Quality metrics computed
- [ ] RBI priors documented
- [ ] Notebook runs end-to-end
- [ ] Privacy compliance statement ready

---

## Questions?

Refer to:
- `QUICK_START.md` - Dataset download guide
- `PROJECT_ROADMAP.md` - Detailed project steps
- `DATASET_DOWNLOAD_GUIDE.md` - Data source details

