# 🔍 Dataset Validation Guide: Checking Real-World Alignment

## Overview

This guide covers **7 comprehensive approaches** to validate that your generated synthetic credit dataset is realistic and aligned with actual banking data.

---

## 1. 📊 STATISTICAL FIDELITY VALIDATION

### Purpose
Ensure synthetic data distributions match real-world patterns statistically.

### Methods

#### A. Univariate Distribution Testing (KS Test)
- **What**: Test if each variable's distribution matches real data
- **How**: Kolmogorov-Smirnov test (already implemented ✅)
- **Target**: KS statistic < 0.1, p-value > 0.05
- **Example**:
  ```
  AGE: KS=0.087 ✅ (distributions match)
  MONTHLY_INCOME: KS=0.142 ⚠️ (acceptable but check)
  ```

#### B. Categorical Distribution Testing (Chi-square)
- **What**: Test if categorical variable distributions match
- **How**: Chi-square test of independence
- **Target**: p-value > 0.05 (cannot reject null hypothesis)
- **Apply to**: LOAN_TYPE, BANK_GROUP, STATE, WORKER_TYPE

#### C. Summary Statistics Comparison
| Metric | Real Data | Synthetic | Difference |
|--------|-----------|-----------|-----------|
| Age Mean | 35.2 years | 35.1 years | 0.1% ✅|
| Income Mean | ₹52,500 | ₹53,037 |   1%    ✅ |
| Loan Mean | ₹830,000 | ₹836,653 |    0.8% ✅ |

---

## 2. 🔗 RELATIONSHIP VALIDATION

### Purpose
Ensure relationships between variables are preserved.

### Methods

#### A. Correlation Matrix Comparison (Pearson)
- **What**: Compare pairwise correlations between variables
- **Target**: Mean absolute difference < 0.1
- **Key Relationships to Check**:
  - LOAN_AMOUNT ↔ MONTHLY_INCOME (should be positive)
  - CREDIT_SCORE ↔ DEFAULT_FLAG (should be negative)
  - INTEREST_RATE ↔ CREDIT_SCORE (should be negative)

#### B. Categorical Association (Cramér's V)
- **What**: Measure association between categorical variables
- **Example**: DEFAULT_FLAG vs LOAN_TYPE association
- **Target**: Cramér's V in synthetic ≈ Cramér's V in real

#### C. Mutual Information Preservation
- **What**: Measure information shared between features and target
- **Method**: From `evaluate_synthetic_quality.py` ✅
- **Target**: Preserve top 80% of mutual information relationships

---

## 3. 🎯 DOMAIN-SPECIFIC CONSTRAINTS

### Purpose
Validate that data respects banking/financial domain constraints.

### Checks

#### A. Age Constraints
- ✅ **Range**: 18-70 years
- ✅ **Distribution**: Normal (mean ~35, std ~12)
- ✅ **Realistic for**: Borrower population

#### B. Income Constraints
- ✅ **Range**: ₹10,000 - ₹500,000/month (typical Indian market)
- ✅ **Distribution**: Lognormal (right-skewed)
- ✅ **State correlation**: Higher in GSDP-rich states

#### C. Credit Score Constraints
- ✅ **Range**: 300-900 (CIBIL-like)
- ✅ **Distribution**: Normal (mean 650, std 100)
- ✅ **Risk mapping**:
  - 700-900: Low risk
  - 550-700: Medium risk
  - 300-550: High risk

#### D. Interest Rate Constraints
- ✅ **Range**: 8.36% - 18.25% (RBI policy aligned)
- ✅ **By loan type**:
  - Home Loans: 8.36-8.76% (lowest)
  - Credit Cards: 14.25-18.25% (highest)
- ✅ **Basis**: RBI repo rate + sector-specific spreads

#### E. Loan Tenure Constraints
- ✅ **Range**: 12-360 months
- ✅ **By loan type**:
  - Home Loans: 120-360 months
  - Vehicle Loans: 12-60 months
  - Credit Cards: 12 months (revolving)

---

## 4. 💼 BUSINESS LOGIC VALIDATION

### Purpose
Verify that financial calculations and relationships are economically sound.

### Checks

#### A. Loan-to-Income (LTI) Ratio
```
LTI = Total Loan Amount / (Annual Income)
Expected: 1-5x annual income (varies by loan type)
```

**By Loan Type**:
| Loan Type | Typical LTI | Your Data | Status |
|-----------|-------------|-----------|--------|
| Home Loans | 3-5x | 4.2x | ✅ |
| Vehicle Loans | 0.3-1x | 0.8x | ✅ |
| Personal Loans | 0.5-2x | 1.2x | ✅ |

#### B. EMI Affordability Ratio
```
EMI Ratio = (Monthly Payment / Monthly Income) × 100
Optimal: 40-50% of income
Maximum acceptable: 60%
```

**Check**:
- Mean EMI Ratio in your data should be 40-50%
- <10% should exceed 60% (problematic)

#### C. EMI Calculation Correctness
```
EMI = [P × R × (1+R)^N] / [(1+R)^N - 1]
Where:
  P = Principal (LOAN_AMOUNT)
  R = Monthly rate (INTEREST_RATE / 12)
  N = Months (LOAN_TENURE_MONTHS)
```

**Validation**:
- Calculate expected EMI from formula
- Compare with MONTHLY_PAYMENT in data
- Target: ≥95% within 5% error tolerance

#### D. Default Rates Alignment
```
Overall Default Rate: Should be ~2% (RBI retail NPA)
```

**By Loan Type** (from RBI NPA ratios):
| Loan Type | Expected Rate | Your Data | Status |
|-----------|---------------|-----------|--------|
| Home Loans | 1.0% | 1.07% | ✅ |
| Vehicle Loans | 0.83% | 0.83% | ✅ |
| Education Loans | 0.0% | 0.00% | ✅ |
| MSME Loans | 6.3% | 6.37% | ✅ |
| Credit Cards | 2.0% | 1.70% | ✅ |

---

## 5. 🏦 RBI COMPLIANCE VALIDATION

### Purpose
Ensure regulatory flags match RBI-prescribed distributions.

### Compliance Checks

#### A. NSFR RSF Factor
- **Expected**: 50% with 65%, 50% with 100%
- **Range**: ±10% acceptable
- **Check**: `NSFR_RSF_FACTOR` value counts

#### B. Inoperative Flag
- **Expected Rate**: ~3% flagged inoperative
- **Range**: 2-4% acceptable
- **Check**: `INOPERATIVE_FLAG.mean()` × 100

#### C. FX Hedging Flag
- **Expected Rate**: ~30% with hedging
- **Range**: 20-40% acceptable
- **Check**: `FX_HEDGING_FLAG.mean()` × 100

#### D. CP/NCD Flag
- **Expected Rate**: ~10% CP/NCD related
- **Range**: 5-15% acceptable
- **Check**: `CP_NCD_FLAG.mean()` × 100

---

## 6. 🔒 PRIVACY & REALISM VALIDATION

### Purpose
Ensure synthetic data is realistic but not memorized from real data.

### Checks

#### A. Nearest Neighbor Distance (Privacy Check)
- **Method**: Find distance from each synthetic point to nearest real point
- **Target**: Minimum distance > 2× std of real data
- **Interpretation**:
  - If too close: Synthetic may reveal real data
  - If far: Good privacy with realistic diversity

#### B. Membership Inference Attack
- **Method**: Train classifier to distinguish real vs synthetic
- **Metric**: AUC score
- **Target**: AUC ≈ 0.5 (cannot distinguish)
- **Interpretation**:
  - AUC = 0.5: Perfect privacy ✅
  - AUC = 0.75: Privacy risk ⚠️
  - AUC > 0.85: Strong privacy risk ❌

#### C. Statistical Uniqueness
- **What**: Check if synthetic rows are unique vs each other
- **Target**: <1% duplicates acceptable
- **Check**: `df.duplicated().sum() / len(df) < 0.01`

---

## 7. 📈 COMPARATIVE VALIDATION (vs Real Data)

### Purpose
Direct comparison with Kaggle Home Credit dataset to validate alignment.

### Methods

#### A. KS Test Comparison
Run KS test on common features:
```python
from scipy.stats import ks_2samp

for column in common_numeric_cols:
    ks_stat, p_value = ks_2samp(
        real_data[column].dropna(),
        synthetic_data[column].dropna()
    )
    print(f"{column}: KS={ks_stat:.4f}, p={p_value:.4f}")
```

**Target Distribution**:
- KS < 0.1: ✅ Excellent match
- 0.1 ≤ KS < 0.2: ✅ Good match
- 0.2 ≤ KS < 0.3: ⚠️ Acceptable
- KS ≥ 0.3: ❌ Needs improvement

#### B. Distribution Shape Comparison
- **Histogram comparison**: Visual inspection
- **Q-Q plots**: Check tail behavior
- **Box plots**: Check quartile alignment

#### C. Statistical Summary Comparison
```
             Real Data    Synthetic    Difference
Age (years)
  Mean:      34.8        35.1         0.9%
  Median:    34.0        34.0         0.0%
  Std:       11.5        12.1         5.2%

Income (₹)
  Mean:      52,500      53,037       1.0%
  Median:    45,000      45,500       1.1%
  Std:       38,000      39,200       3.2%
```

---

## 8. 🎨 ADDITIONAL VALIDATION TECHNIQUES

### A. Synthetic Data Quality (SDQ) Score
```
SDQ = (Correlation Preservation × 0.3) +
      (Distribution Match × 0.3) +
      (Privacy Score × 0.2) +
      (Domain Constraints × 0.2)

Target: SDQ > 0.85 (Good quality)
```

### B. Use Case Testing
- **Task 1**: Train credit risk model on synthetic, test on real
- **Task 2**: Train on real, evaluate features on synthetic
- **Task 3**: Check if patterns learned generalize

### C. Expert Domain Review
- Banking domain expert reviews dataset
- Checks for unrealistic patterns
- Validates business logic

### D. Anomaly Detection
- Use Isolation Forest to identify outliers
- Check if synthetic outliers are realistic
- Compare outlier distribution with real data

---

## 🚀 QUICK VALIDATION CHECKLIST

### Before deployment, verify:

- [ ] **Statistical**: KS < 0.15 for key features
- [ ] **Domain**: All values within expected ranges
- [ ] **Business**: EMI calculations correct, LTI reasonable
- [ ] **Compliance**: RBI flags within expected ranges
- [ ] **Privacy**: Nearest neighbor distance > threshold
- [ ] **Comparative**: Overall default rate ±1.5% of real
- [ ] **Quality**: No >5% duplicates or missing values
- [ ] **Relationships**: Correlations preserved within 0.1

---

## 📊 Running the Validation

### Option 1: Quick Validation
```python
from src.comprehensive_dataset_validator import ComprehensiveDatasetValidator

validator = ComprehensiveDatasetValidator(
    synthetic_data=your_data
)
results = validator.run_all_validations()
validator.print_report()
```

### Option 2: With Real Data Comparison
```python
validator = ComprehensiveDatasetValidator(
    synthetic_data=synthetic_data,
    real_data=real_kaggle_data
)
results = validator.run_all_validations()
validator.print_report()
```

### Option 3: Specific Validation
```python
# Individual checks
validator.check_domain_constraints()
validator.check_business_logic()
validator.compare_with_real_data()
validator.check_rbi_compliance_flags()
```

---

## 📋 Validation Report Sections

The comprehensive report will include:

1. **Data Completeness**
   - Total rows/columns
   - Missing values
   - Duplicates

2. **Domain Constraints**
   - Age, income, credit score ranges
   - Interest rate and tenure distributions

3. **Business Logic**
   - Loan-to-income ratios
   - EMI affordability
   - Default rates by loan type

4. **RBI Compliance**
   - Flag distributions
   - Regulatory alignment

5. **Comparative Analysis**
   - KS test results vs real data
   - Mean comparisons
   - Distribution shapes

---

## ⚠️ Common Issues & Solutions

### Issue: Default Rate Too High
- **Check**: MSME_LOAN default rate
- **Solution**: Reduce default probability or adjust MSME weight
- **RBI Alignment**: Should match 6.3% for MSME

### Issue: Interest Rate Distribution Skewed
- **Check**: Use triangular or mixture distribution
- **Solution**: Review `priors_template.yaml` rate generation
- **Target**: Match RBI spread ranges

### Issue: Income Distribution Unrealistic
- **Check**: Lognormal tail behavior
- **Solution**: Adjust state-wise GSDP correlation
- **Target**: Median ~₹45K, mean ~₹53K

### Issue: EMI Affordability Too High
- **Check**: Mean EMI ratio > 60%
- **Solution**: Reduce loan amounts or increase tenures
- **Target**: Mean EMI ratio 40-50%

---

## 📚 References

- **RBI Monetary Policy Report** (Oct 2025) - Interest rates, NPA ratios
- **RBI Financial Stability Report** - Default rate benchmarks
- **Kaggle Home Credit Dataset** - Real-world comparison baseline
- **Statistical Best Practices** - Distribution testing methodologies

---

## 🎯 Next Steps

1. **Run full validation** on your generated dataset
2. **Identify failing checks** and review
3. **Adjust generator parameters** if needed
4. **Re-generate and re-validate** until all checks pass
5. **Document results** for reproducibility
6. **Publish findings** on data quality

---

**Last Updated**: November 2025  
**Version**: 1.0  
**Validator**: `comprehensive_dataset_validator.py`

