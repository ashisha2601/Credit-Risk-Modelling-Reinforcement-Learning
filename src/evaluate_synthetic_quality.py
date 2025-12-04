"""
Statistical Fidelity and Quality Metrics for Synthetic Data Validation

This module implements statistical tests to validate synthetic data quality:
1. Univariate Distribution Match (KS Test, Chi-square)
2. Correlation Preservation (Pearson, Cramér's V)
3. Mutual Information Preservation
4. Privacy Metrics (Nearest Neighbor, Membership Inference)
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.stats.contingency import association
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class SyntheticDataEvaluator:
    """
    Evaluates statistical fidelity and privacy of synthetic data
    """
    
    def __init__(self, real_data=None, synthetic_data=None):
        """
        Initialize evaluator
        
        Args:
            real_data: Reference/real data (optional, for comparison)
            synthetic_data: Synthetic data to evaluate
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.metrics = {}
    
    def evaluate_univariate_distributions(self, real_col, synth_col):
        """
        Test if univariate distributions match using Kolmogorov-Smirnov test
        
        Args:
            real_col: Real data column (numeric)
            synth_col: Synthetic data column (numeric)
        
        Returns:
            dict: KS statistic and p-value
        
        Statistical Test: Kolmogorov-Smirnov (KS) Test
        - Null Hypothesis: Two distributions are identical
        - KS Statistic: Maximum difference between CDFs (0-1 scale)
        - p-value: Probability that distributions match
        
        Target: KS < 0.1 (good), p-value > 0.05 (cannot reject null)
        """
        # Remove missing values
        real_clean = real_col.dropna()
        synth_clean = synth_col.dropna()
        
        if len(real_clean) == 0 or len(synth_clean) == 0:
            return {'ks_statistic': np.nan, 'p_value': np.nan, 'interpretation': 'Insufficient data'}
        
        # Perform KS test
        ks_statistic, p_value = ks_2samp(real_clean, synth_clean)
        
        # Interpretation
        if ks_statistic < 0.1 and p_value > 0.05:
            interpretation = "✅ Excellent match (KS < 0.1, p > 0.05)"
        elif ks_statistic < 0.2 and p_value > 0.05:
            interpretation = "✅ Good match (KS < 0.2, p > 0.05)"
        elif ks_statistic < 0.3:
            interpretation = "⚠️ Acceptable match (KS < 0.3)"
        else:
            interpretation = "❌ Poor match (KS >= 0.3)"
        
        return {
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'interpretation': interpretation
        }
    
    def evaluate_categorical_distributions(self, real_col, synth_col):
        """
        Test if categorical distributions match using Chi-square test
        
        Args:
            real_col: Real data column (categorical)
            synth_col: Synthetic data column (categorical)
        
        Returns:
            dict: Chi-square statistic and p-value
        
        Statistical Test: Chi-square Test of Independence
        - Null Hypothesis: Distributions are independent/identical
        - Chi-square Statistic: Measures discrepancy from expected frequencies
        - p-value: Probability that distributions match
        
        Target: p-value > 0.05 (cannot reject null, distributions match)
        """
        # Get value counts
        real_counts = real_col.value_counts()
        synth_counts = synth_col.value_counts()
        
        # Get union of all categories
        all_categories = set(real_counts.index) | set(synth_counts.index)
        
        # Create contingency table
        contingency = pd.DataFrame({
            'real': [real_counts.get(cat, 0) for cat in all_categories],
            'synthetic': [synth_counts.get(cat, 0) for cat in all_categories]
        })
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Interpretation
        if p_value > 0.05:
            interpretation = f"✅ Distributions match (p={p_value:.4f} > 0.05)"
        else:
            interpretation = f"⚠️ Distributions differ (p={p_value:.4f} <= 0.05)"
        
        return {
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'interpretation': interpretation
        }
    
    def evaluate_correlation_preservation(self, numeric_cols=None):
        """
        Compare correlation matrices between real and synthetic data
        
        Args:
            numeric_cols: List of numeric column names to compare
        
        Returns:
            dict: Correlation difference metrics
        
        Statistical Technique: Pearson Correlation Matrix Comparison
        - Calculate correlation matrix for real data
        - Calculate correlation matrix for synthetic data
        - Measure absolute difference
        - Target: Mean difference < 0.1
        
        Also computes: Cramér's V for categorical-categorical associations
        """
        if self.real_data is None or self.synthetic_data is None:
            return {'error': 'Both real and synthetic data required'}
        
        if numeric_cols is None:
            # Auto-detect numeric columns
            numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c in self.synthetic_data.columns]
        
        if len(numeric_cols) == 0:
            return {'error': 'No numeric columns found'}
        
        # Calculate correlation matrices
        real_corr = self.real_data[numeric_cols].corr()
        synth_corr = self.synthetic_data[numeric_cols].corr()
        
        # Mean absolute difference
        corr_diff = np.abs(real_corr - synth_corr)
        mean_diff = corr_diff.mean().mean()
        max_diff = corr_diff.max().max()
        
        # Count how many correlations differ significantly
        significant_diffs = (corr_diff > 0.1).sum().sum()
        total_pairs = len(numeric_cols) * (len(numeric_cols) - 1) // 2
        
        # Interpretation
        if mean_diff < 0.05:
            interpretation = "✅ Excellent correlation preservation (mean diff < 0.05)"
        elif mean_diff < 0.1:
            interpretation = "✅ Good correlation preservation (mean diff < 0.1)"
        elif mean_diff < 0.2:
            interpretation = "⚠️ Acceptable correlation preservation (mean diff < 0.2)"
        else:
            interpretation = "❌ Poor correlation preservation (mean diff >= 0.2)"
        
        return {
            'mean_correlation_difference': mean_diff,
            'max_correlation_difference': max_diff,
            'significant_differences': significant_diffs,
            'total_pairs': total_pairs,
            'percentage_preserved': (1 - significant_diffs / total_pairs) * 100 if total_pairs > 0 else 0,
            'interpretation': interpretation
        }
    
    def evaluate_mutual_information(self, target_col='TARGET'):
        """
        Check if feature-target relationships are preserved
        
        Args:
            target_col: Target variable column name
        
        Returns:
            dict: Mutual Information metrics
        
        Statistical Technique: Mutual Information
        - Measures information shared between features and target
        - Higher MI = stronger relationship
        - Target: Preserve top 80% of MI values
        """
        from sklearn.feature_selection import mutual_info_classif
        
        if target_col not in self.synthetic_data.columns:
            return {'error': f'Target column {target_col} not found'}
        
        # Get numeric features
        feature_cols = self.synthetic_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != target_col]
        
        if len(feature_cols) == 0:
            return {'error': 'No feature columns found'}
        
        # Calculate Mutual Information
        X = self.synthetic_data[feature_cols].fillna(0)
        y = self.synthetic_data[target_col]
        
        mi_scores = mutual_info_classif(X, y, random_state=42)
        
        # Top features by MI
        mi_df = pd.DataFrame({
            'feature': feature_cols,
            'mutual_information': mi_scores
        }).sort_values('mutual_information', ascending=False)
        
        # Identify top 80% of relationships
        threshold = np.percentile(mi_scores, 20)  # Bottom 20% = top 80%
        top_features = mi_df[mi_df['mutual_information'] >= threshold]
        
        return {
            'total_features': len(feature_cols),
            'high_mi_features': len(top_features),
            'mi_threshold': threshold,
            'top_10_features': mi_df.head(10).to_dict('records'),
            'mean_mi': mi_scores.mean(),
            'max_mi': mi_scores.max()
        }
    
    def evaluate_nearest_neighbor_distance(self, n_neighbors=1):
        """
        Ensure synthetic data is not too close to real data (privacy check)
        
        Args:
            n_neighbors: Number of nearest neighbors to check
        
        Returns:
            dict: Distance metrics
        
        Privacy Metric: Nearest Neighbor Distance
        - Compute distance from each synthetic sample to nearest real sample
        - Higher distance = better privacy
        - Target: Minimum distance > threshold (varies by feature scale)
        """
        if self.real_data is None:
            return {'error': 'Real data required for privacy check'}
        
        # Select numeric columns for distance calculation
        common_cols = [c for c in self.real_data.columns 
                      if c in self.synthetic_data.columns 
                      and self.real_data[c].dtype in [np.number]]
        
        if len(common_cols) == 0:
            return {'error': 'No common numeric columns found'}
        
        # Prepare data
        real_subset = self.real_data[common_cols].fillna(0)
        synth_subset = self.synthetic_data[common_cols].fillna(0)
        
        # Fit nearest neighbors on real data
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
        nn.fit(real_subset)
        
        # Find distances from synthetic to real
        distances, indices = nn.kneighbors(synth_subset)
        
        # Statistics
        min_dist = distances.min()
        mean_dist = distances.mean()
        median_dist = np.median(distances)
        max_dist = distances.max()
        
        # Interpretation (threshold heuristic: mean of real data std)
        threshold = real_subset.std().mean() * 2
        
        if min_dist > threshold:
            interpretation = f"✅ Excellent privacy (min distance {min_dist:.2f} > threshold {threshold:.2f})"
        elif min_dist > threshold / 2:
            interpretation = f"✅ Good privacy (min distance {min_dist:.2f} > threshold/2)"
        else:
            interpretation = f"⚠️ Privacy risk (min distance {min_dist:.2f} <= threshold/2)"
        
        return {
            'min_distance': min_dist,
            'mean_distance': mean_dist,
            'median_distance': median_dist,
            'max_distance': max_dist,
            'threshold': threshold,
            'interpretation': interpretation
        }
    
    def evaluate_membership_inference(self):
        """
        Test if real records can be identified from synthetic data
        
        Returns:
            dict: Membership inference attack results
        
        Privacy Metric: Membership Inference Attack
        - Train classifier to distinguish real vs synthetic
        - AUC ≈ 0.5 means cannot distinguish (good privacy)
        - AUC >> 0.5 means can distinguish (privacy risk)
        """
        if self.real_data is None:
            return {'error': 'Real data required for membership inference test'}
        
        # Prepare data
        common_cols = [c for c in self.real_data.columns 
                      if c in self.synthetic_data.columns 
                      and self.real_data[c].dtype in [np.number]]
        
        if len(common_cols) == 0:
            return {'error': 'No common numeric columns found'}
        
        # Sample equal amounts
        n_samples = min(len(self.real_data), len(self.synthetic_data), 5000)
        real_sample = self.real_data[common_cols].fillna(0).sample(n_samples, random_state=42)
        synth_sample = self.synthetic_data[common_cols].fillna(0).sample(n_samples, random_state=42)
        
        # Create labels: 1 = real, 0 = synthetic
        X = pd.concat([real_sample, synth_sample], ignore_index=True)
        y = np.array([1] * len(real_sample) + [0] * len(synth_sample))
        
        # Train classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        rf.fit(X, y)
        
        # Predict probabilities
        y_pred_proba = rf.predict_proba(X)[:, 1]
        
        # Calculate AUC
        auc = roc_auc_score(y, y_pred_proba)
        
        # Interpretation
        if auc < 0.55:
            interpretation = f"✅ Excellent privacy (AUC={auc:.3f} ≈ 0.5, cannot distinguish)"
        elif auc < 0.65:
            interpretation = f"✅ Good privacy (AUC={auc:.3f} < 0.65)"
        elif auc < 0.75:
            interpretation = f"⚠️ Moderate privacy risk (AUC={auc:.3f} < 0.75)"
        else:
            interpretation = f"❌ Privacy risk (AUC={auc:.3f} >= 0.75, can distinguish)"
        
        return {
            'auc_score': auc,
            'interpretation': interpretation,
            'privacy_level': 'Good' if auc < 0.65 else 'Risk'
        }
    
    def evaluate_rbi_compliance(self):
        """
        Check if RBI compliance flags are correctly distributed
        
        Returns:
            dict: RBI compliance metrics
        """
        if self.synthetic_data is None:
            return {'error': 'Synthetic data required'}
        
        metrics = {}
        
        # NSFR RSF Factor
        if 'NSFR_RSF_FACTOR' in self.synthetic_data.columns:
            nsfr_values = self.synthetic_data['NSFR_RSF_FACTOR'].value_counts()
            metrics['nsfr_factor'] = {
                'values': nsfr_values.to_dict(),
                'expected': {65: 0.5, 100: 0.5},
                'status': '✅ Present' if len(nsfr_values) == 2 else '⚠️ Check distribution'
            }
        
        # Inoperative Flag (~3%)
        if 'INOPERATIVE_FLAG' in self.synthetic_data.columns:
            inop_rate = self.synthetic_data['INOPERATIVE_FLAG'].mean()
            metrics['inoperative_flag'] = {
                'rate': inop_rate,
                'expected_rate': 0.03,
                'difference': abs(inop_rate - 0.03),
                'status': '✅ Compliant' if abs(inop_rate - 0.03) < 0.01 else '⚠️ Check rate'
            }
        
        # FX Hedging Flag (~30%)
        if 'FX_HEDGING_FLAG' in self.synthetic_data.columns:
            fx_rate = self.synthetic_data['FX_HEDGING_FLAG'].mean()
            metrics['fx_hedging_flag'] = {
                'rate': fx_rate,
                'expected_rate': 0.30,
                'difference': abs(fx_rate - 0.30),
                'status': '✅ Compliant' if abs(fx_rate - 0.30) < 0.05 else '⚠️ Check rate'
            }
        
        # CP/NCD Flag (~10%)
        if 'CP_NCD_FLAG' in self.synthetic_data.columns:
            cp_rate = self.synthetic_data['CP_NCD_FLAG'].mean()
            metrics['cp_ncd_flag'] = {
                'rate': cp_rate,
                'expected_rate': 0.10,
                'difference': abs(cp_rate - 0.10),
                'status': '✅ Compliant' if abs(cp_rate - 0.10) < 0.03 else '⚠️ Check rate'
            }
        
        return metrics
    
    def evaluate_default_rate_by_loan_type(self, target_col='TARGET'):
        """
        Check if default rates by loan type match RBI NPA ratios
        
        Args:
            target_col: Target/default column name
        
        Returns:
            dict: Default rate comparisons
        """
        if self.synthetic_data is None or 'LOAN_TYPE' not in self.synthetic_data.columns:
            return {'error': 'LOAN_TYPE and target column required'}
        
        if target_col not in self.synthetic_data.columns:
            # Try DEFAULT_FLAG as alternative
            target_col = 'DEFAULT_FLAG' if 'DEFAULT_FLAG' in self.synthetic_data.columns else None
        
        if target_col is None:
            return {'error': 'Target column not found'}
        
        # Calculate default rates by loan type
        loan_defaults = self.synthetic_data.groupby('LOAN_TYPE')[target_col].agg(['mean', 'count'])
        loan_defaults.columns = ['default_rate', 'count']
        loan_defaults['default_rate_pct'] = loan_defaults['default_rate'] * 100
        
        # Expected rates (from RBI priors)
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
        
        # Compare
        comparison = []
        for loan_type, row in loan_defaults.iterrows():
            expected = expected_rates.get(loan_type, 2.0)
            actual = row['default_rate_pct']
            diff = abs(actual - expected)
            
            comparison.append({
                'loan_type': loan_type,
                'actual_rate': actual,
                'expected_rate': expected,
                'difference': diff,
                'status': '✅ Match' if diff < 1.0 else '⚠️ Check'
            })
        
        return {
            'by_loan_type': comparison,
            'overall_default_rate': self.synthetic_data[target_col].mean() * 100,
            'expected_overall': 2.0
        }
    
    def comprehensive_evaluation(self):
        """
        Run all evaluation metrics
        
        Returns:
            dict: Complete evaluation results
        """
        results = {
            'univariate_tests': {},
            'correlation_preservation': {},
            'mutual_information': {},
            'privacy_metrics': {},
            'rbi_compliance': {},
            'default_rate_validation': {}
        }
        
        if self.real_data is not None and self.synthetic_data is not None:
            # Find common numeric columns
            numeric_cols = [c for c in self.real_data.columns 
                          if c in self.synthetic_data.columns 
                          and self.real_data[c].dtype in [np.number]
                          and c not in ['SK_ID_CURR']]
            
            # Univariate tests (sample of key features)
            key_features = ['AGE', 'MONTHLY_INCOME', 'LOAN_AMOUNT', 'INTEREST_RATE']
            for feat in key_features:
                if feat in self.real_data.columns and feat in self.synthetic_data.columns:
                    results['univariate_tests'][feat] = self.evaluate_univariate_distributions(
                        self.real_data[feat], self.synthetic_data[feat]
                    )
            
            # Correlation preservation
            results['correlation_preservation'] = self.evaluate_correlation_preservation()
            
            # Mutual Information
            target_col = 'TARGET' if 'TARGET' in self.synthetic_data.columns else 'DEFAULT_FLAG'
            if target_col in self.synthetic_data.columns:
                results['mutual_information'] = self.evaluate_mutual_information(target_col)
            
            # Privacy metrics
            results['privacy_metrics']['nearest_neighbor'] = self.evaluate_nearest_neighbor_distance()
            results['privacy_metrics']['membership_inference'] = self.evaluate_membership_inference()
        
        # RBI compliance (no real data needed)
        results['rbi_compliance'] = self.evaluate_rbi_compliance()
        
        # Default rate validation
        target_col = 'TARGET' if 'TARGET' in self.synthetic_data.columns else 'DEFAULT_FLAG'
        if 'LOAN_TYPE' in self.synthetic_data.columns:
            results['default_rate_validation'] = self.evaluate_default_rate_by_loan_type(target_col)
        
        return results
    
    def print_evaluation_report(self, results=None):
        """
        Print formatted evaluation report
        """
        if results is None:
            results = self.comprehensive_evaluation()
        
        print("="*80)
        print("SYNTHETIC DATA QUALITY EVALUATION REPORT")
        print("="*80)
        
        # Univariate Tests
        if results.get('univariate_tests'):
            print("\n📊 UNIVARIATE DISTRIBUTION MATCH (Kolmogorov-Smirnov Test)")
            print("-"*80)
            for feature, metrics in results['univariate_tests'].items():
                if 'ks_statistic' in metrics:
                    print(f"{feature}:")
                    print(f"  KS Statistic: {metrics['ks_statistic']:.4f}")
                    print(f"  p-value: {metrics['p_value']:.4f}")
                    print(f"  {metrics['interpretation']}")
        
        # Correlation Preservation
        if results.get('correlation_preservation') and 'mean_correlation_difference' in results['correlation_preservation']:
            print("\n🔗 CORRELATION PRESERVATION")
            print("-"*80)
            cp = results['correlation_preservation']
            print(f"Mean Correlation Difference: {cp['mean_correlation_difference']:.4f}")
            print(f"Max Correlation Difference: {cp['max_correlation_difference']:.4f}")
            print(f"Preserved Correlations: {cp['percentage_preserved']:.1f}%")
            print(f"  {cp['interpretation']}")
        
        # Mutual Information
        if results.get('mutual_information') and 'mean_mi' in results['mutual_information']:
            print("\n📈 MUTUAL INFORMATION PRESERVATION")
            print("-"*80)
            mi = results['mutual_information']
            print(f"Mean Mutual Information: {mi['mean_mi']:.4f}")
            print(f"Max Mutual Information: {mi['max_mi']:.4f}")
            print(f"High MI Features: {mi['high_mi_features']}/{mi['total_features']}")
        
        # Privacy Metrics
        if results.get('privacy_metrics'):
            print("\n🔒 PRIVACY METRICS")
            print("-"*80)
            
            if 'nearest_neighbor' in results['privacy_metrics']:
                nn = results['privacy_metrics']['nearest_neighbor']
                if 'min_distance' in nn:
                    print(f"Nearest Neighbor Distance:")
                    print(f"  Minimum: {nn['min_distance']:.2f}")
                    print(f"  Mean: {nn['mean_distance']:.2f}")
                    print(f"  {nn.get('interpretation', '')}")
            
            if 'membership_inference' in results['privacy_metrics']:
                mi_attack = results['privacy_metrics']['membership_inference']
                if 'auc_score' in mi_attack:
                    print(f"\nMembership Inference Attack:")
                    print(f"  AUC Score: {mi_attack['auc_score']:.4f}")
                    print(f"  {mi_attack.get('interpretation', '')}")
        
        # RBI Compliance
        if results.get('rbi_compliance'):
            print("\n🏦 RBI COMPLIANCE VALIDATION")
            print("-"*80)
            for flag_name, flag_metrics in results['rbi_compliance'].items():
                print(f"{flag_name}:")
                for key, value in flag_metrics.items():
                    if key != 'status':
                        print(f"  {key}: {value}")
                print(f"  {flag_metrics.get('status', '')}")
        
        # Default Rate Validation
        if results.get('default_rate_validation'):
            print("\n💰 DEFAULT RATE VALIDATION")
            print("-"*80)
            drv = results['default_rate_validation']
            print(f"Overall Default Rate: {drv.get('overall_default_rate', 0):.2f}%")
            print(f"Expected Rate: {drv.get('expected_overall', 0):.2f}%")
            
            if 'by_loan_type' in drv:
                print("\nBy Loan Type:")
                for item in drv['by_loan_type']:
                    print(f"  {item['loan_type']}: {item['actual_rate']:.2f}% "
                          f"(Expected: {item['expected_rate']:.2f}%) - {item['status']}")
        
        print("\n" + "="*80)


def main():
    """Example usage"""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    
    # Load datasets
    synthetic_path = project_root / "data" / "synthetic_credit_data_v0.3_hybrid_ctgan.csv"
    kaggle_path = project_root / "home-credit-default-risk" / "application_train.csv"
    
    print("Loading synthetic data...")
    synthetic_data = pd.read_csv(synthetic_path)
    
    real_data = None
    if kaggle_path.exists():
        print("Loading Kaggle reference data...")
        real_data = pd.read_csv(kaggle_path, nrows=10000)  # Sample for speed
    
    # Evaluate
    evaluator = SyntheticDataEvaluator(real_data=real_data, synthetic_data=synthetic_data)
    results = evaluator.comprehensive_evaluation()
    evaluator.print_evaluation_report(results)
    
    return results


if __name__ == "__main__":
    results = main()









