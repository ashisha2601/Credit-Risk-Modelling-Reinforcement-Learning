# Column Descriptions - Synthetic Credit Data v0.1

## Numeric Features (12)

### 1. **AGE**
- **Description**: Borrower's age in years
- **Data Type**: Integer (int64)
- **Range**: 18 to 70 years
- **Mean**: ~35 years
- **Distribution**: Normal distribution (mean: 35, std: 12)
- **Source**: Demographics priors from Census 2011
- **Use Case**: Age is a key risk factor; younger borrowers may have less credit history, older borrowers may have stable income but approaching retirement
- **Privacy Note**: Synthetic age values, no real personal data

---

### 2. **MONTHLY_INCOME**
- **Description**: Borrower's monthly household income in Indian Rupees (INR)
- **Data Type**: Float (float64)
- **Range**: ₹10,000 to ₹374,930
- **Mean**: ~₹53,037
- **Distribution**: Lognormal distribution (adjusted by state GSDP)
- **Source**: RBI priors - Urban mean: ₹50,000, Rural mean: ₹25,000
- **Correlation**: Slightly correlated with STATE (higher GSDP states tend to have higher incomes)
- **Use Case**: Primary indicator of repayment capacity; used to calculate loan-to-income ratios
- **Privacy Note**: Synthetic income values, no real financial data

---

### 3. **CREDIT_SCORE**
- **Description**: Credit score (similar to CIBIL score) indicating creditworthiness
- **Data Type**: Integer (int64)
- **Range**: 300 to 900
- **Mean**: 650
- **Distribution**: Normal distribution (mean: 650, std: 100)
- **Source**: Estimated CIBIL-like score distribution
- **Risk Ranges**:
  - **Low Risk**: 700-900 (low default probability)
  - **Medium Risk**: 550-700 (medium default probability)
  - **High Risk**: 300-550 (high default probability)
- **Use Case**: Key predictor of default probability; higher scores indicate better credit history
- **Privacy Note**: Synthetic credit scores, no real credit bureau data

---

### 4. **LOAN_AMOUNT**
- **Description**: Total loan amount sanctioned/disbursed in Indian Rupees (INR)
- **Data Type**: Float (float64)
- **Range**: ₹50,000 to ₹10,000,000+
- **Mean**: ~₹836,653
- **Distribution**: Varies by loan type (home loans: higher, credit cards: lower)
- **Source**: Calculated using loan-to-income ratios from RBI priors
- **Loan Type Specific**:
  - **Credit Cards**: ₹10,000 - ₹500,000 (revolving credit limit)
  - **Home Loans**: ₹500,000 - ₹10,000,000+ (secured, long-term)
  - **Personal/Other**: ₹50,000 - ₹5,000,000 (unsecured/semi-secured)
- **Use Case**: Principal amount for EMI calculation; larger loans may have higher default risk
- **Privacy Note**: Synthetic loan amounts, no real loan data

---

### 5. **INTEREST_RATE**
- **Description**: Annual Percentage Rate (APR) for the loan, expressed as a decimal (e.g., 0.10 = 10%)
- **Data Type**: Float (float64)
- **Range**: 0.0836 (8.36%) to 0.1825 (18.25%)
- **Mean**: ~0.0996 (9.96%)
- **Distribution**: Triangular distribution (skewed towards mean) by loan type
- **Source**: RBI Monetary Policy Report Oct 2025 - Based on repo rate (6.25%) + sector-specific spreads
- **Loan Type Specific Rates**:
  - **Home Loans**: 8.36% - 8.76% (lowest, secured loans)
  - **Vehicle Loans**: 8.94% - 10.51%
  - **Education Loans**: 9.67% - 11.80%
  - **MSME Loans**: 9.58% - 9.73%
  - **Personal Loans**: 9.55% - 11.56%
  - **Credit Cards**: 14.25% - 18.25% (highest, unsecured revolving credit)
- **Regulatory Basis**: Aligned with RBI policy rates (SDF: 6.25%, Repo: 6.50%, MSF: 6.75%) + spreads from Table IV.12
- **Use Case**: Determines monthly payment (EMI); higher rates compensate for higher risk
- **Privacy Note**: Synthetic interest rates based on RBI published spreads

---

### 6. **LOAN_TENURE_MONTHS**
- **Description**: Loan repayment period in months
- **Data Type**: Integer (int64)
- **Range**: 12 to 360 months (1 to 30 years)
- **Mean**: ~71 months (~6 years)
- **Distribution**: Varies significantly by loan type
- **Source**: RBI priors with loan-type-specific distributions
- **Loan Type Specific Tenures**:
  - **Home Loans**: 120-360 months (10-30 years) - Long-term secured loans
  - **Vehicle Loans**: 12-60 months (1-5 years) - Medium-term secured loans
  - **Education Loans**: 60-120 months (5-10 years) - Medium-to-long-term
  - **MSME Loans**: 12-60 months (1-5 years) - Short-to-medium-term
  - **Personal Loans**: 12-60 months (1-5 years) - Short-term unsecured
  - **Credit Cards**: 12 months (revolving credit, no fixed tenure)
- **Use Case**: Used in EMI calculation; longer tenures reduce monthly payment but increase total interest
- **Privacy Note**: Synthetic tenure values

---

### 7. **DEFAULT_FLAG**
- **Description**: Binary indicator of loan default (target variable)
- **Data Type**: Integer (int64)
- **Values**: 
  - **0**: No default (loan performing/paid)
  - **1**: Default (loan non-performing/delinquent)
- **Distribution**: 
  - Overall default rate: ~2.67% (aligned with RBI retail NPA ratio of 2.0%)
  - Varies by loan type (see below)
- **Source**: RBI NPA ratios from Monetary Policy Report Oct 2025
- **Loan Type Specific Default Rates**:
  - **Home Loans**: ~1.07% (lowest, secured)
  - **Vehicle Loans**: ~0.83% (low, secured)
  - **Personal Loans**: ~2.84% (moderate, unsecured)
  - **Credit Cards**: ~1.70% (moderate, unsecured)
  - **MSME Loans**: ~6.37% (highest, aligned with agriculture NPA ratio)
  - **Other Loans**: ~1.99% (moderate)
- **Use Case**: Primary target variable for credit risk modeling; used in clustering and RL policy training
- **Privacy Note**: Synthetic default flags based on statistical distributions, no real default data

---

### 8. **MONTHLY_PAYMENT**
- **Description**: Monthly Equated Monthly Installment (EMI) or minimum payment in Indian Rupees (INR)
- **Data Type**: Float (float64)
- **Range**: Varies by loan amount, interest rate, and tenure
- **Calculation**: 
  - **Regular Loans**: EMI = [P × R × (1+R)^N] / [(1+R)^N - 1]
    - P = Principal (LOAN_AMOUNT)
    - R = Monthly interest rate (INTEREST_RATE / 12)
    - N = Tenure in months (LOAN_TENURE_MONTHS)
  - **Credit Cards**: Minimum payment = 5% of outstanding balance (revolving credit)
- **Source**: Calculated from LOAN_AMOUNT, INTEREST_RATE, and LOAN_TENURE_MONTHS
- **Use Case**: Key metric for affordability assessment; debt-to-income ratio calculations
- **Privacy Note**: Synthetic calculated values

---

### 9. **NSFR_RSF_FACTOR**
- **Description**: Net Stable Funding Ratio (NSFR) Required Stable Funding (RSF) factor
- **Data Type**: Integer (int64)
- **Values**: 
  - **65**: 65% RSF factor (for certain loan categories)
  - **100**: 100% RSF factor (standard)
- **Distribution**: Approximately 50% each (5013 records with 65, 4987 records with 100)
- **Source**: RBI Regulation - Dec 29, 2023 - NSFR for Non-Deposit Taking NBFCs
- **Regulatory Context**: 
  - NSFR ensures banks/NBFCs maintain stable funding sources
  - RSF factors determine how much stable funding is required for assets
  - 65% factor applies to certain secured loans; 100% is standard
- **Use Case**: Regulatory compliance tracking; liquidity risk management
- **Privacy Note**: Synthetic regulatory flags, compliant with RBI guidelines

---

### 10. **INOPERATIVE_FLAG**
- **Description**: Flag indicating whether the account is inoperative
- **Data Type**: Integer (int64)
- **Values**: 
  - **0**: Active account
  - **1**: Inoperative account
- **Distribution**: ~2.84% flagged as inoperative (284 out of 10,000)
- **Source**: RBI Regulation - Jan 01, 2024 - Inoperative Accounts
- **Regulatory Context**: 
  - Accounts with no customer-initiated transactions for 2+ years are marked inoperative
  - Inoperative accounts have restrictions on transactions
  - Helps prevent fraud and comply with KYC norms
- **Use Case**: Account management, fraud prevention, regulatory compliance
- **Privacy Note**: Synthetic flags based on RBI regulation rate (~3%)

---

### 11. **FX_HEDGING_FLAG**
- **Description**: Flag indicating whether the loan/bank has Foreign Exchange (FX) risk hedging
- **Data Type**: Integer (int64)
- **Values**: 
  - **0**: No FX hedging
  - **1**: FX hedging present
- **Distribution**: ~29.68% flagged as hedged (2,968 out of 10,000)
- **Source**: RBI Regulation - Jan 05, 2024 - FX Risk Hedging
- **Regulatory Context**: 
  - Banks/NBFCs with foreign currency exposure must hedge FX risk
  - Helps manage currency fluctuation risks
  - Required for certain types of loans with foreign currency components
- **Use Case**: FX risk management, regulatory compliance, portfolio risk assessment
- **Privacy Note**: Synthetic flags based on estimated hedging prevalence (~30%)

---

### 12. **CP_NCD_FLAG**
- **Description**: Flag indicating Commercial Paper (CP) or Non-Convertible Debenture (NCD) related transactions
- **Data Type**: Integer (int64)
- **Values**: 
  - **0**: Not CP/NCD related
  - **1**: CP/NCD related
- **Distribution**: ~10.06% flagged as CP/NCD (1,006 out of 10,000)
- **Source**: RBI Regulation - Jan 03, 2024 - CP/NCD Consistency
- **Regulatory Context**: 
  - Commercial Paper: Short-term unsecured promissory notes
  - Non-Convertible Debentures: Long-term debt instruments
  - Required for consistency in reporting and risk management
- **Use Case**: Debt instrument tracking, regulatory reporting, risk classification
- **Privacy Note**: Synthetic flags based on estimated CP/NCD prevalence (~10%)

---

## Summary

All 12 numeric features are:
- ✅ **Synthetic**: Generated using statistical priors, no real customer data
- ✅ **RBI Compliant**: Aligned with RBI regulations and monetary policy data
- ✅ **Realistic**: Distributions match Indian market characteristics
- ✅ **Complete**: No missing values, ready for analysis
- ✅ **Privacy-Preserving**: Compliant with DPDPA 2023

---

## Use Cases by Feature Type

### **Demographic Features** (AGE, MONTHLY_INCOME, CREDIT_SCORE)
- Borrower profiling
- Risk segmentation
- Creditworthiness assessment

### **Loan Features** (LOAN_AMOUNT, INTEREST_RATE, LOAN_TENURE_MONTHS, MONTHLY_PAYMENT)
- Loan pricing decisions
- Affordability analysis
- Portfolio risk assessment

### **Target Variable** (DEFAULT_FLAG)
- Credit risk modeling
- Default prediction
- Loss estimation

### **Regulatory Features** (NSFR_RSF_FACTOR, INOPERATIVE_FLAG, FX_HEDGING_FLAG, CP_NCD_FLAG)
- Regulatory compliance
- Risk management
- Audit and reporting

---

**Last Updated**: 2025  
**Data Version**: v0.1  
**Generator**: Hybrid Synthetic Data Generator

