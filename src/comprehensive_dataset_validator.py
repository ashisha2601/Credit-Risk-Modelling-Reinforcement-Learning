"""
Comprehensive Dataset Validation Framework
Validates synthetic data against multiple dimensions:
1. Statistical fidelity
2. Business logic
3. Domain-specific constraints
4. Privacy preservation
5. Real-world alignment with RBI/Kaggle data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class ComprehensiveDatasetValidator:
    """Multi-dimensional validation of synthetic banking data"""
    
    def __init__(self, synthetic_data, real_data=None):
        self.synthetic = synthetic_data
        self.real = real_data
        self.validation_results = {}
    
    # ============================================================
    # 1. STATISTICAL FIDELITY CHECKS
    # ============================================================
    
    def check_statistical_properties(self):
        """Validate basic statistical properties"""
        results = {}
        
        numeric_cols = self.synthetic.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            synth_stats = self.synthetic[col].describe()
            results[col] = {
                'count': synth_stats['count'],
                'mean': synth_stats['mean'],
                'std': synth_stats['std'],
                'min': synth_stats['min'],
                'max': synth_stats['max'],
                '25%': synth_stats['25%'],
                '50%': synth_stats['50%'],
                '75%': synth_stats['75%'],
                'missing': self.synthetic[col].isna().sum(),
                'status': '✅ OK' if self.synthetic[col].isna().sum() == 0 else '⚠️ Missing values'
            }
        
        self.validation_results['statistical_properties'] = results
        return results
    
    def check_data_types(self):
        """Ensure data types are appropriate"""
        issues = []
        
        # Age should be integer
        if 'AGE' in self.synthetic.columns:
            if self.synthetic['AGE'].dtype not in [np.int32, np.int64, int]:
                issues.append("⚠️ AGE should be integer type")
        
        # Amounts should be numeric
        amount_cols = ['LOAN_AMOUNT', 'MONTHLY_INCOME', 'MONTHLY_PAYMENT']
        for col in amount_cols:
            if col in self.synthetic.columns:
                if not pd.api.types.is_numeric_dtype(self.synthetic[col]):
                    issues.append(f"⚠️ {col} should be numeric")
        
        # Flags should be 0/1
        flag_cols = ['DEFAULT_FLAG', 'INOPERATIVE_FLAG', 'FX_HEDGING_FLAG', 'CP_NCD_FLAG']
        for col in flag_cols:
            if col in self.synthetic.columns:
                unique_vals = self.synthetic[col].unique()
                if not all(val in [0, 1, 0.0, 1.0] for val in unique_vals):
                    issues.append(f"⚠️ {col} should only have values 0 or 1")
        
        status = "✅ All data types OK" if not issues else "⚠️ Data type issues found"
        self.validation_results['data_types'] = {'issues': issues, 'status': status}
        return {'issues': issues, 'status': status}
    
    # ============================================================
    # 2. DOMAIN CONSTRAINTS VALIDATION
    # ============================================================
    
    def check_age_constraints(self):
        """Age should be 18-70 for borrowers"""
        results = {}
        
        if 'AGE' not in self.synthetic.columns:
            return {'status': 'Column not found'}
        
        age_col = self.synthetic['AGE']
        results['min'] = age_col.min()
        results['max'] = age_col.max()
        results['mean'] = age_col.mean()
        results['violations'] = len(age_col[(age_col < 18) | (age_col > 70)])
        
        expected_range = (18, 70)
        in_range = len(age_col[(age_col >= 18) & (age_col <= 70)]) / len(age_col) * 100
        
        results['percentage_in_range'] = in_range
        results['status'] = '✅ OK' if results['violations'] == 0 else f"⚠️ {results['violations']} violations"
        results['expected_range'] = expected_range
        
        self.validation_results['age_constraints'] = results
        return results
    
    def check_income_constraints(self):
        """Monthly income should be positive and reasonable for Indian market"""
        results = {}
        
        if 'MONTHLY_INCOME' not in self.synthetic.columns:
            return {'status': 'Column not found'}
        
        income_col = self.synthetic['MONTHLY_INCOME']
        results['min'] = income_col.min()
        results['max'] = income_col.max()
        results['mean'] = income_col.mean()
        results['median'] = income_col.median()
        
        # Reasonable range for Indian market: ₹10,000 - ₹500,000
        reasonable_min = 10000
        reasonable_max = 500000
        
        violations = len(income_col[(income_col < reasonable_min) | (income_col > reasonable_max)])
        in_range = len(income_col[(income_col >= reasonable_min) & (income_col <= reasonable_max)]) / len(income_col) * 100
        
        results['reasonable_range'] = (reasonable_min, reasonable_max)
        results['percentage_in_range'] = in_range
        results['violations'] = violations
        results['status'] = '✅ OK' if violations == 0 else f"⚠️ {violations} violations"
        
        self.validation_results['income_constraints'] = results
        return results
    
    def check_credit_score_constraints(self):
        """Credit score should be 300-900 (CIBIL-like)"""
        results = {}
        
        if 'CREDIT_SCORE' not in self.synthetic.columns:
            return {'status': 'Column not found'}
        
        score_col = self.synthetic['CREDIT_SCORE']
        results['min'] = score_col.min()
        results['max'] = score_col.max()
        results['mean'] = score_col.mean()
        
        violations = len(score_col[(score_col < 300) | (score_col > 900)])
        in_range = len(score_col[(score_col >= 300) & (score_col <= 900)]) / len(score_col) * 100
        
        results['expected_range'] = (300, 900)
        results['percentage_in_range'] = in_range
        results['violations'] = violations
        results['status'] = '✅ OK' if violations == 0 else f"⚠️ {violations} violations"
        
        self.validation_results['credit_score_constraints'] = results
        return results
    
    def check_interest_rate_constraints(self):
        """Interest rates should align with RBI policy"""
        results = {}
        
        if 'INTEREST_RATE' not in self.synthetic.columns:
            return {'status': 'Column not found'}
        
        rate_col = self.synthetic['INTEREST_RATE']
        results['min'] = rate_col.min()
        results['max'] = rate_col.max()
        results['mean'] = rate_col.mean()
        
        # RBI rates: 8.36% - 18.25%
        violations = len(rate_col[(rate_col < 0.0836) | (rate_col > 0.1825)])
        in_range = len(rate_col[(rate_col >= 0.0836) & (rate_col <= 0.1825)]) / len(rate_col) * 100
        
        results['expected_range'] = (0.0836, 0.1825)
        results['percentage_in_range'] = in_range
        results['violations'] = violations
        results['status'] = '✅ OK' if violations == 0 else f"⚠️ {violations} violations"
        
        # By loan type check
        if 'LOAN_TYPE' in self.synthetic.columns:
            by_type = self.synthetic.groupby('LOAN_TYPE')['INTEREST_RATE'].agg(['min', 'max', 'mean'])
            results['by_loan_type'] = by_type.to_dict()
        
        self.validation_results['interest_rate_constraints'] = results
        return results
    
    def check_loan_tenure_constraints(self):
        """Loan tenure should be reasonable for each loan type"""
        results = {}
        
        if 'LOAN_TENURE_MONTHS' not in self.synthetic.columns:
            return {'status': 'Column not found'}
        
        tenure_col = self.synthetic['LOAN_TENURE_MONTHS']
        results['min'] = int(tenure_col.min())
        results['max'] = int(tenure_col.max())
        results['mean'] = int(tenure_col.mean())
        
        violations = len(tenure_col[(tenure_col < 12) | (tenure_col > 360)])
        in_range = len(tenure_col[(tenure_col >= 12) & (tenure_col <= 360)]) / len(tenure_col) * 100
        
        results['expected_range'] = (12, 360)
        results['percentage_in_range'] = in_range
        results['violations'] = violations
        results['status'] = '✅ OK' if violations == 0 else f"⚠️ {violations} violations"
        
        self.validation_results['loan_tenure_constraints'] = results
        return results
    
    # ============================================================
    # 3. BUSINESS LOGIC VALIDATION
    # ============================================================
    
    def check_loan_to_income_ratio(self):
        """Loan-to-income ratios should be realistic"""
        results = {}
        
        required_cols = ['LOAN_AMOUNT', 'MONTHLY_INCOME']
        if not all(col in self.synthetic.columns for col in required_cols):
            return {'status': 'Required columns not found'}
        
        # Calculate loan-to-income (in months of income)
        lti = self.synthetic['LOAN_AMOUNT'] / (self.synthetic['MONTHLY_INCOME'] * 12)
        
        results['mean_lti'] = lti.mean()
        results['median_lti'] = lti.median()
        results['min_lti'] = lti.min()
        results['max_lti'] = lti.max()
        results['std_lti'] = lti.std()
        
        # Typical ranges by loan type (if available)
        if 'LOAN_TYPE' in self.synthetic.columns:
            by_type = {}
            for loan_type in self.synthetic['LOAN_TYPE'].unique():
                mask = self.synthetic['LOAN_TYPE'] == loan_type
                type_lti = lti[mask]
                by_type[loan_type] = {
                    'mean': type_lti.mean(),
                    'median': type_lti.median(),
                    'min': type_lti.min(),
                    'max': type_lti.max()
                }
            results['by_loan_type'] = by_type
        
        # Check for unrealistic ratios (> 10x annual income is very unusual)
        unrealistic = len(lti[lti > 10]) / len(lti) * 100
        results['percentage_unrealistic_high'] = unrealistic
        results['status'] = '✅ OK' if unrealistic < 1 else f"⚠️ {unrealistic:.2f}% unrealistic ratios"
        
        self.validation_results['loan_to_income'] = results
        return results
    
    def check_emi_affordability(self):
        """EMI should be reasonable (typically 40-50% of income)"""
        results = {}
        
        required_cols = ['MONTHLY_PAYMENT', 'MONTHLY_INCOME']
        if not all(col in self.synthetic.columns for col in required_cols):
            return {'status': 'Required columns not found'}
        
        # EMI as percentage of income
        emi_ratio = (self.synthetic['MONTHLY_PAYMENT'] / self.synthetic['MONTHLY_INCOME']) * 100
        
        results['mean_emi_ratio'] = emi_ratio.mean()
        results['median_emi_ratio'] = emi_ratio.median()
        results['min_emi_ratio'] = emi_ratio.min()
        results['max_emi_ratio'] = emi_ratio.max()
        
        # Flag high EMIs (> 60% of income is problematic)
        high_emi = len(emi_ratio[emi_ratio > 60]) / len(emi_ratio) * 100
        results['percentage_high_emi'] = high_emi
        
        # Recommended: 40-50%
        optimal = len(emi_ratio[(emi_ratio >= 30) & (emi_ratio <= 60)]) / len(emi_ratio) * 100
        results['percentage_optimal_range'] = optimal
        
        results['status'] = '✅ OK' if high_emi < 10 else f"⚠️ {high_emi:.2f}% high EMI"
        
        self.validation_results['emi_affordability'] = results
        return results
    
    def check_emi_calculation_correctness(self):
        """Verify EMI is calculated correctly: P*R*(1+R)^N / ((1+R)^N - 1)"""
        results = {}
        
        required_cols = ['LOAN_AMOUNT', 'INTEREST_RATE', 'LOAN_TENURE_MONTHS', 'MONTHLY_PAYMENT']
        if not all(col in self.synthetic.columns for col in required_cols):
            return {'status': 'Required columns not found'}
        
        # Calculate expected EMI
        P = self.synthetic['LOAN_AMOUNT']
        r = self.synthetic['INTEREST_RATE'] / 12  # Monthly rate
        n = self.synthetic['LOAN_TENURE_MONTHS']
        
        # Avoid division by zero
        mask = r > 0
        expected_emi = np.zeros(len(P))
        expected_emi[mask] = (P[mask] * r[mask] * (1 + r[mask]) ** n[mask]) / ((1 + r[mask]) ** n[mask] - 1)
        expected_emi[~mask] = P[~mask] / n[~mask]  # Simple division if rate = 0
        
        actual_emi = self.synthetic['MONTHLY_PAYMENT'].values
        
        # Calculate difference
        difference = np.abs(actual_emi - expected_emi)
        relative_error = (difference / expected_emi) * 100
        
        results['mean_absolute_difference'] = difference.mean()
        results['max_absolute_difference'] = difference.max()
        results['mean_relative_error'] = relative_error[np.isfinite(relative_error)].mean()
        results['median_relative_error'] = np.median(relative_error[np.isfinite(relative_error)])
        
        # Within 5% is acceptable
        within_tolerance = len(relative_error[relative_error < 5]) / len(relative_error) * 100
        results['percentage_within_5_percent'] = within_tolerance
        
        results['status'] = '✅ OK' if within_tolerance > 95 else f"⚠️ Only {within_tolerance:.1f}% within tolerance"
        
        self.validation_results['emi_calculation'] = results
        return results
    
    def check_default_rate_distribution(self):
        """Default rates should match RBI statistics"""
        results = {}
        
        if 'DEFAULT_FLAG' not in self.synthetic.columns:
            return {'status': 'DEFAULT_FLAG column not found'}
        
        overall_rate = self.synthetic['DEFAULT_FLAG'].mean() * 100
        results['overall_default_rate'] = overall_rate
        results['expected_rbi_rate'] = 2.0  # RBI retail NPA ~2%
        results['difference'] = abs(overall_rate - 2.0)
        results['status'] = '✅ OK' if abs(overall_rate - 2.0) < 1.5 else f"⚠️ Differs from RBI rate"
        
        # By loan type
        if 'LOAN_TYPE' in self.synthetic.columns:
            by_type = {}
            expected_rates = {
                'HOME_LOAN': 1.0,
                'VEHICLE_LOAN': 0.83,
                'EDUCATION_LOAN': 0.0,
                'CREDIT_CARD': 2.0,
                'PERSONAL_LOAN': 2.0,
                'MSME_LOAN': 6.3,
                'SERVICES_LOAN': 2.0,
                'OTHER_LOAN': 2.0
            }
            
            for loan_type in self.synthetic['LOAN_TYPE'].unique():
                mask = self.synthetic['LOAN_TYPE'] == loan_type
                actual = self.synthetic.loc[mask, 'DEFAULT_FLAG'].mean() * 100
                expected = expected_rates.get(loan_type, 2.0)
                
                by_type[loan_type] = {
                    'actual_rate': actual,
                    'expected_rate': expected,
                    'difference': abs(actual - expected)
                }
            
            results['by_loan_type'] = by_type
        
        self.validation_results['default_rate_distribution'] = results
        return results
    
    def check_rbi_compliance_flags(self):
        """RBI compliance flags should have correct distributions"""
        results = {}
        
        # NSFR RSF Factor: ~50% each
        if 'NSFR_RSF_FACTOR' in self.synthetic.columns:
            nsfr_dist = self.synthetic['NSFR_RSF_FACTOR'].value_counts(normalize=True) * 100
            results['nsfr_rsf'] = {
                'distribution': nsfr_dist.to_dict(),
                'expected': {65: 50, 100: 50},
                'status': '✅ OK' if all(40 < v < 60 for v in nsfr_dist.values) else '⚠️ Check distribution'
            }
        
        # Inoperative Flag: ~3%
        if 'INOPERATIVE_FLAG' in self.synthetic.columns:
            inop_rate = self.synthetic['INOPERATIVE_FLAG'].mean() * 100
            results['inoperative'] = {
                'rate': inop_rate,
                'expected_rate': 3.0,
                'difference': abs(inop_rate - 3.0),
                'status': '✅ OK' if abs(inop_rate - 3.0) < 1.5 else '⚠️ Check rate'
            }
        
        # FX Hedging: ~30%
        if 'FX_HEDGING_FLAG' in self.synthetic.columns:
            fx_rate = self.synthetic['FX_HEDGING_FLAG'].mean() * 100
            results['fx_hedging'] = {
                'rate': fx_rate,
                'expected_rate': 30.0,
                'difference': abs(fx_rate - 30.0),
                'status': '✅ OK' if abs(fx_rate - 30.0) < 10 else '⚠️ Check rate'
            }
        
        # CP/NCD Flag: ~10%
        if 'CP_NCD_FLAG' in self.synthetic.columns:
            cp_rate = self.synthetic['CP_NCD_FLAG'].mean() * 100
            results['cp_ncd'] = {
                'rate': cp_rate,
                'expected_rate': 10.0,
                'difference': abs(cp_rate - 10.0),
                'status': '✅ OK' if abs(cp_rate - 10.0) < 5 else '⚠️ Check rate'
            }
        
        self.validation_results['rbi_compliance_flags'] = results
        return results
    
    def check_interest_rate_by_credit_score(self):
        """Interest rates should increase with risk (inverse to credit score)"""
        results = {}
        
        required_cols = ['INTEREST_RATE', 'CREDIT_SCORE']
        if not all(col in self.synthetic.columns for col in required_cols):
            return {'status': 'Required columns not found'}
        
        correlation = self.synthetic['INTEREST_RATE'].corr(self.synthetic['CREDIT_SCORE'])
        results['correlation'] = correlation
        results['expected_sign'] = 'negative'  # Lower credit score = higher rate
        results['interpretation'] = 'Strong inverse' if correlation < -0.3 else ('Moderate inverse' if correlation < -0.1 else 'Weak inverse')
        results['status'] = '✅ OK' if correlation < -0.1 else '⚠️ Check relationship'
        
        self.validation_results['interest_rate_vs_credit_score'] = results
        return results
    
    def check_default_rate_vs_credit_score(self):
        """Default rate should be higher for lower credit scores"""
        results = {}
        
        required_cols = ['DEFAULT_FLAG', 'CREDIT_SCORE']
        if not all(col in self.synthetic.columns for col in required_cols):
            return {'status': 'Required columns not found'}
        
        correlation = self.synthetic['DEFAULT_FLAG'].corr(self.synthetic['CREDIT_SCORE'])
        results['correlation'] = correlation
        results['expected_sign'] = 'negative'  # Lower credit score = higher default
        results['interpretation'] = 'Strong inverse' if correlation < -0.1 else 'Weak inverse'
        results['status'] = '✅ OK' if correlation < 0 else '⚠️ Check relationship'
        
        self.validation_results['default_vs_credit_score'] = results
        return results
    
    # ============================================================
    # 4. COMPARATIVE VALIDATION (vs Real Data)
    # ============================================================
    
    def compare_with_real_data(self):
        """Compare synthetic vs real data distributions"""
        if self.real is None:
            return {'status': 'Real data not provided'}
        
        results = {}
        
        # Find common numeric columns
        common_cols = [col for col in self.synthetic.columns 
                      if col in self.real.columns 
                      and pd.api.types.is_numeric_dtype(self.synthetic[col])]
        
        from scipy.stats import ks_2samp
        
        for col in common_cols:
            if col in ['SK_ID_CURR']:  # Skip ID columns
                continue
            
            synth_clean = self.synthetic[col].dropna()
            real_clean = self.real[col].dropna()
            
            if len(synth_clean) > 0 and len(real_clean) > 0:
                ks_stat, p_value = ks_2samp(real_clean, synth_clean)
                
                results[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'match_quality': 'Excellent' if ks_stat < 0.1 else ('Good' if ks_stat < 0.2 else 'Needs improvement'),
                    'synth_mean': synth_clean.mean(),
                    'real_mean': real_clean.mean(),
                    'mean_diff': abs(synth_clean.mean() - real_clean.mean())
                }
        
        self.validation_results['comparison_with_real'] = results
        return results
    
    # ============================================================
    # 5. DATA QUALITY CHECKS
    # ============================================================
    
    def check_data_completeness(self):
        """Check for missing values and duplicates"""
        results = {}
        
        # Missing values
        missing = self.synthetic.isnull().sum()
        results['missing_values'] = missing[missing > 0].to_dict() if len(missing[missing > 0]) > 0 else {}
        
        # Duplicate rows
        duplicates = self.synthetic.duplicated().sum()
        results['duplicate_rows'] = duplicates
        
        # Duplicate by ID (if ID column exists)
        if 'SK_ID_CURR' in self.synthetic.columns:
            dup_ids = self.synthetic['SK_ID_CURR'].duplicated().sum()
            results['duplicate_ids'] = dup_ids
        
        results['total_rows'] = len(self.synthetic)
        results['total_cols'] = len(self.synthetic.columns)
        results['status'] = '✅ OK' if duplicates == 0 and len(results['missing_values']) == 0 else '⚠️ Data quality issues'
        
        self.validation_results['data_completeness'] = results
        return results
    
    def check_categorical_distributions(self):
        """Check categorical variables"""
        results = {}
        
        categorical_cols = self.synthetic.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            value_counts = self.synthetic[col].value_counts()
            results[col] = {
                'unique_values': len(value_counts),
                'top_values': value_counts.head(5).to_dict(),
                'distribution': value_counts.to_dict()
            }
        
        self.validation_results['categorical_distributions'] = results
        return results
    
    # ============================================================
    # 6. COMPREHENSIVE VALIDATION REPORT
    # ============================================================
    
    def run_all_validations(self):
        """Run all validation checks"""
        print("🔍 Running Comprehensive Dataset Validation...\n")
        
        # Basic checks
        print("1️⃣  Basic Statistical Properties...")
        self.check_statistical_properties()
        self.check_data_types()
        
        # Domain constraints
        print("2️⃣  Domain Constraints...")
        self.check_age_constraints()
        self.check_income_constraints()
        self.check_credit_score_constraints()
        self.check_interest_rate_constraints()
        self.check_loan_tenure_constraints()
        
        # Business logic
        print("3️⃣  Business Logic Validation...")
        self.check_loan_to_income_ratio()
        self.check_emi_affordability()
        self.check_emi_calculation_correctness()
        self.check_default_rate_distribution()
        self.check_rbi_compliance_flags()
        self.check_interest_rate_by_credit_score()
        self.check_default_rate_vs_credit_score()
        
        # Data quality
        print("4️⃣  Data Quality...")
        self.check_data_completeness()
        self.check_categorical_distributions()
        
        # Comparative
        if self.real is not None:
            print("5️⃣  Comparative Analysis with Real Data...")
            self.compare_with_real_data()
        
        print("\n✅ Validation Complete!\n")
        return self.validation_results
    
    def print_report(self):
        """Print formatted validation report"""
        print("=" * 80)
        print("COMPREHENSIVE DATASET VALIDATION REPORT")
        print("=" * 80)
        
        # Data Completeness
        if 'data_completeness' in self.validation_results:
            print("\n📊 DATA COMPLETENESS")
            print("-" * 80)
            dc = self.validation_results['data_completeness']
            print(f"Total Rows: {dc['total_rows']:,}")
            print(f"Total Columns: {dc['total_cols']}")
            print(f"Missing Values: {len(dc['missing_values'])}")
            print(f"Duplicate Rows: {dc['duplicate_rows']}")
            print(f"Status: {dc['status']}")
        
        # Domain Constraints
        print("\n🎯 DOMAIN CONSTRAINTS")
        print("-" * 80)
        
        constraints = ['age_constraints', 'income_constraints', 'credit_score_constraints', 
                      'interest_rate_constraints', 'loan_tenure_constraints']
        
        for constraint in constraints:
            if constraint in self.validation_results:
                result = self.validation_results[constraint]
                name = constraint.replace('_constraints', '').replace('_', ' ').title()
                print(f"{name}: {result['status']}")
                if 'violations' in result and result['violations'] > 0:
                    print(f"  ⚠️ Violations: {result['violations']}")
        
        # Business Logic
        print("\n💼 BUSINESS LOGIC VALIDATION")
        print("-" * 80)
        
        if 'loan_to_income' in self.validation_results:
            lti = self.validation_results['loan_to_income']
            print(f"Loan-to-Income Ratio: {lti['mean_lti']:.2f}x annual (median: {lti['median_lti']:.2f}x)")
            print(f"  {lti['status']}")
        
        if 'emi_affordability' in self.validation_results:
            emi = self.validation_results['emi_affordability']
            print(f"EMI Affordability: {emi['mean_emi_ratio']:.1f}% of income (median: {emi['median_emi_ratio']:.1f}%)")
            print(f"  {emi['status']}")
        
        if 'default_rate_distribution' in self.validation_results:
            drd = self.validation_results['default_rate_distribution']
            print(f"Default Rate: {drd['overall_default_rate']:.2f}% (Expected: ~2.0%)")
            print(f"  {drd['status']}")
        
        # RBI Compliance
        if 'rbi_compliance_flags' in self.validation_results:
            print("\n🏦 RBI COMPLIANCE FLAGS")
            print("-" * 80)
            rbi = self.validation_results['rbi_compliance_flags']
            for key, value in rbi.items():
                if 'status' in value:
                    print(f"{key.replace('_', ' ').title()}: {value['status']}")
        
        # Comparative Analysis
        if 'comparison_with_real' in self.validation_results:
            print("\n📈 COMPARISON WITH REAL DATA (KS Test)")
            print("-" * 80)
            comp = self.validation_results['comparison_with_real']
            for col, metrics in comp.items():
                if 'ks_statistic' in metrics:
                    print(f"{col}: KS={metrics['ks_statistic']:.4f}, {metrics['match_quality']}")
        
        print("\n" + "=" * 80)


def main():
    """Example usage"""
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    
    # Load synthetic data
    synthetic_path = project_root / "data" / "synthetic_credit_data_v0.3_hybrid_ctgan.csv"
    print(f"Loading synthetic data from: {synthetic_path}")
    synthetic_data = pd.read_csv(synthetic_path)
    
    # Load real data if available
    real_data = None
    kaggle_path = project_root / "home-credit-default-risk" / "application_train.csv"
    if kaggle_path.exists():
        print(f"Loading real data from: {kaggle_path}")
        real_data = pd.read_csv(kaggle_path, nrows=10000)
    
    # Run validation
    validator = ComprehensiveDatasetValidator(synthetic_data=synthetic_data, real_data=real_data)
    results = validator.run_all_validations()
    validator.print_report()
    
    return results, validator


if __name__ == "__main__":
    results, validator = main()

