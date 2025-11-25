"""
Hybrid Synthetic Data Generator
Combines Kaggle dataset structure with RBI/Indian market priors
"""

import pandas as pd
import numpy as np
import yaml
import random
from pathlib import Path
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
import warnings
warnings.filterwarnings('ignore')

class HybridSyntheticGenerator:
    """
    Generates synthetic credit data by:
    1. Learning structure from Kaggle dataset
    2. Calibrating distributions using RBI/Indian priors
    3. Generating privacy-preserving synthetic data
    """
    
    def __init__(self, kaggle_data_path=None, priors_path='config/priors_template.yaml'):
        """
        Initialize generator
        
        Args:
            kaggle_data_path: Path to Kaggle application_train.csv (optional)
            priors_path: Path to YAML file with RBI priors
        """
        self.priors_path = priors_path
        self.priors = self._load_priors()
        self.kaggle_data = None
        
        if kaggle_data_path and Path(kaggle_data_path).exists():
            print(f"📊 Loading Kaggle dataset from {kaggle_data_path}...")
            self.kaggle_data = pd.read_csv(kaggle_data_path, nrows=50000)  # Load sample for speed
            print(f"✓ Loaded {len(self.kaggle_data)} rows")
        else:
            print("⚠️  No Kaggle data provided. Will generate from priors only.")
    
    def _load_priors(self):
        """Load RBI priors from YAML file"""
        priors_file = Path(self.priors_path)
        if not priors_file.exists():
            print(f"⚠️  Priors file not found: {self.priors_path}")
            print("Using default priors. Update config/priors_template.yaml with actual values.")
            return self._get_default_priors()
        
        with open(priors_file, 'r') as f:
            priors = yaml.safe_load(f)
        print(f"✓ Loaded priors from {self.priors_path}")
        return priors
    
    def _get_default_priors(self):
        """Return default priors if file not found"""
        return {
            'default_rates': {'overall_retail': 0.025},
            'income_stats': {'urban': {'mean': 50000, 'std': 30000}},
            'demographics': {'age': {'mean': 35, 'std': 12}}
        }
    
    def prepare_kaggle_data(self, max_rows=10000):
        """
        Prepare Kaggle data: select relevant columns and sample
        Focus on columns similar to Indian credit application
        """
        if self.kaggle_data is None:
            return None
        
        # Select key columns (matching Indian credit application structure)
        key_columns = [
            'SK_ID_CURR',  # ID
            'TARGET',      # Default flag
            'AMT_INCOME_TOTAL',  # Income
            'AMT_CREDIT',  # Loan amount
            'AMT_ANNUITY', # Monthly payment
            'AMT_GOODS_PRICE',  # Goods price
            'DAYS_BIRTH',  # Age (negative days)
            'DAYS_EMPLOYED',  # Employment duration
            'DAYS_REGISTRATION',  # Registration date
            'DAYS_ID_PUBLISH',  # ID publication
            'OWN_CAR_AGE',  # Car age
            'FLAG_MOBIL',  # Has mobile
            'FLAG_EMP_PHONE',  # Has work phone
            'FLAG_WORK_PHONE',  # Has work phone
            'FLAG_CONT_MOBILE',  # Has contact mobile
            'FLAG_PHONE',  # Has phone
            'FLAG_EMAIL',  # Has email
            'CNT_CHILDREN',  # Number of children
            'AMT_REQ_CREDIT_BUREAU_HOUR',  # Credit bureau requests
            'AMT_REQ_CREDIT_BUREAU_DAY',
            'AMT_REQ_CREDIT_BUREAU_WEEK',
            'AMT_REQ_CREDIT_BUREAU_MONTH',
            'AMT_REQ_CREDIT_BUREAU_QRT',
            'AMT_REQ_CREDIT_BUREAU_YEAR',
        ]
        
        # Add categorical columns
        categorical_cols = [
            'NAME_CONTRACT_TYPE',
            'CODE_GENDER',
            'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY',
            'NAME_TYPE_SUITE',
            'NAME_INCOME_TYPE',
            'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE',
            'OCCUPATION_TYPE',
            'ORGANIZATION_TYPE',
        ]
        
        # Combine all columns
        all_cols = ['SK_ID_CURR', 'TARGET'] + [c for c in key_columns if c not in ['SK_ID_CURR', 'TARGET']]
        for col in categorical_cols:
            if col in self.kaggle_data.columns:
                all_cols.append(col)
        
        # Select available columns
        available_cols = [c for c in all_cols if c in self.kaggle_data.columns]
        
        # Sample data
        data_sample = self.kaggle_data[available_cols].head(max_rows).copy()
        
        # Transform DAYS_BIRTH to age
        if 'DAYS_BIRTH' in data_sample.columns:
            data_sample['AGE'] = (-data_sample['DAYS_BIRTH'] / 365.25).astype(int)
            data_sample['AGE'] = data_sample['AGE'].clip(18, 70)
            data_sample = data_sample.drop('DAYS_BIRTH', axis=1)
        
        # Transform DAYS_EMPLOYED
        if 'DAYS_EMPLOYED' in data_sample.columns:
            data_sample['YEARS_EMPLOYED'] = (-data_sample['DAYS_EMPLOYED'] / 365.25).clip(0, 50)
            data_sample = data_sample.drop('DAYS_EMPLOYED', axis=1)
        
        print(f"✓ Prepared {len(data_sample)} rows with {len(data_sample.columns)} columns")
        return data_sample
    
    def apply_indian_priors(self, df):
        """
        Adjust Kaggle data to match Indian market characteristics
        Uses RBI priors to transform distributions
        """
        df_adjusted = df.copy()
        n_samples = len(df_adjusted)
        
        # Adjust income to INR (if needed)
        if 'AMT_INCOME_TOTAL' in df_adjusted.columns:
            # Convert to monthly income in INR (adjust scale if needed)
            # Kaggle data might be in different currency/scale
            urban_mean = self.priors.get('income_stats', {}).get('urban', {}).get('mean', 50000)
            current_mean = df_adjusted['AMT_INCOME_TOTAL'].mean()
            if current_mean > 0:
                scale_factor = urban_mean / current_mean
                df_adjusted['MONTHLY_INCOME'] = df_adjusted['AMT_INCOME_TOTAL'] * scale_factor
            else:
                df_adjusted['MONTHLY_INCOME'] = np.random.lognormal(
                    np.log(urban_mean), 0.5, len(df_adjusted)
                )
        
        # Add loan types if not present
        if 'LOAN_TYPE' not in df_adjusted.columns:
            df_adjusted['LOAN_TYPE'] = self._assign_loan_types(n_samples)
        
        # Adjust loan amounts using loan-to-income ratios if present
        if 'AMT_CREDIT' in df_adjusted.columns:
            # Recalculate using RBI priors
            if 'MONTHLY_INCOME' in df_adjusted.columns:
                df_adjusted['LOAN_AMOUNT'] = self._calculate_loan_amounts(
                    df_adjusted['MONTHLY_INCOME'].values,
                    df_adjusted['LOAN_TYPE'].values
                )
            else:
                df_adjusted['LOAN_AMOUNT'] = df_adjusted['AMT_CREDIT']
        elif 'MONTHLY_INCOME' in df_adjusted.columns:
            df_adjusted['LOAN_AMOUNT'] = self._calculate_loan_amounts(
                df_adjusted['MONTHLY_INCOME'].values,
                df_adjusted['LOAN_TYPE'].values
            )
        
        # Add interest rates based on loan type
        if 'INTEREST_RATE' not in df_adjusted.columns:
            df_adjusted['INTEREST_RATE'] = self._assign_interest_rates(df_adjusted['LOAN_TYPE'].values)
        
        # Adjust loan tenures based on loan type
        if 'LOAN_TENURE_MONTHS' not in df_adjusted.columns:
            df_adjusted['LOAN_TENURE_MONTHS'] = self._assign_loan_tenures(df_adjusted['LOAN_TYPE'].values)
        
        # Add bank groups
        if 'BANK_GROUP' not in df_adjusted.columns:
            df_adjusted['BANK_GROUP'] = self._assign_bank_groups(n_samples)
        
        # Adjust default rate to match RBI priors by loan type
        if 'TARGET' in df_adjusted.columns or 'DEFAULT_FLAG' in df_adjusted.columns:
            default_col = 'TARGET' if 'TARGET' in df_adjusted.columns else 'DEFAULT_FLAG'
            default_rates_by_type = self._assign_default_rates_by_loan_type(df_adjusted['LOAN_TYPE'].values)
            df_adjusted[default_col] = np.array([
                np.random.binomial(1, rate) for rate in default_rates_by_type
            ])
        
        # Calculate monthly payment (EMI)
        if 'MONTHLY_PAYMENT' not in df_adjusted.columns:
            monthly_rates = df_adjusted['INTEREST_RATE'].values / 12
            tenure_months = df_adjusted['LOAN_TENURE_MONTHS'].values
            principal = df_adjusted['LOAN_AMOUNT'].values
            loan_types_arr = df_adjusted['LOAN_TYPE'].values
            
            emi = np.zeros(n_samples)
            for i in range(n_samples):
                if loan_types_arr[i] == 'CREDIT_CARD':
                    emi[i] = principal[i] * 0.05
                else:
                    p = principal[i]
                    r = monthly_rates[i]
                    n = tenure_months[i]
                    if r > 0 and n > 0:
                        emi[i] = p * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
                    else:
                        emi[i] = 0
            
            df_adjusted['MONTHLY_PAYMENT'] = emi
        
        # Adjust age distribution
        if 'AGE' in df_adjusted.columns:
            age_mean = self.priors.get('demographics', {}).get('age', {}).get('mean', 35)
            age_std = self.priors.get('demographics', {}).get('age', {}).get('std', 12)
            df_adjusted['AGE'] = np.random.normal(age_mean, age_std, len(df_adjusted)).clip(18, 70).astype(int)
        
        return df_adjusted
    
    def generate_synthetic(self, n_samples=10000, method='gaussian_copula', 
                          use_kaggle_structure=True, use_indian_priors=True):
        """
        Generate synthetic credit data
        
        Args:
            n_samples: Number of synthetic samples to generate
            method: 'gaussian_copula' (fast) or 'ctgan' (better quality)
            use_kaggle_structure: Whether to learn structure from Kaggle data
            use_indian_priors: Whether to apply Indian market priors
        """
        print(f"\n🔧 Generating {n_samples} synthetic samples using {method}...")
        
        # Prepare reference data
        if use_kaggle_structure and self.kaggle_data is not None:
            reference_data = self.prepare_kaggle_data(max_rows=10000)
            
            if use_indian_priors:
                reference_data = self.apply_indian_priors(reference_data)
            
            # Create metadata
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(reference_data)
            
            # Choose synthesizer
            if method == 'gaussian_copula':
                synthesizer = GaussianCopulaSynthesizer(metadata)
            elif method == 'ctgan':
                synthesizer = CTGANSynthesizer(metadata)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Fit on reference data
            print("📚 Fitting synthesizer on reference data...")
            synthesizer.fit(reference_data)
            
            # Generate synthetic data
            print("✨ Generating synthetic data...")
            synthetic_data = synthesizer.sample(num_rows=n_samples)
            
            # Add RBI compliance flags and Indian market features
            synthetic_data = self._add_rbi_compliance_flags(synthetic_data)
            synthetic_data = self._add_indian_market_features(synthetic_data)
            
        else:
            # Generate from scratch using priors only
            print("📊 Generating from priors only...")
            synthetic_data = self._generate_from_priors(n_samples)
        
        print(f"✅ Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data
    
    def _assign_loan_types(self, n_samples):
        """Assign loan types based on RBI credit growth distributions"""
        priors = self.priors
        credit_growth = priors.get('credit_growth_by_sector', {})
        
        # Extract personal loan type distributions
        personal_loans = credit_growth.get('personal_loans', {})
        if isinstance(personal_loans, dict) and 'total' in personal_loans:
            # Normalize shares to probabilities
            total_personal = personal_loans.get('total', 0.351)
            housing_share = personal_loans.get('housing', 0.176) / total_personal
            vehicle_share = personal_loans.get('vehicle', 0.037) / total_personal
            education_share = personal_loans.get('education', 0.008) / total_personal
            credit_card_share = personal_loans.get('credit_cards', 0.016) / total_personal
            other_share = personal_loans.get('other_personal_loans', 0.087) / total_personal
            
            # Remaining probability for other loan types (industry, services, agriculture)
            remaining_prob = 1 - total_personal
            
            # Assign loan types
            loan_types = []
            probs = [
                housing_share * total_personal,
                vehicle_share * total_personal,
                education_share * total_personal,
                credit_card_share * total_personal,
                other_share * total_personal,
                remaining_prob * 0.3,  # Industry/MSME
                remaining_prob * 0.3,  # Services
                remaining_prob * 0.4,  # Other/Agriculture
            ]
            probs = np.array(probs) / sum(probs)  # Normalize
            
            loan_type_map = [
                'HOME_LOAN',
                'VEHICLE_LOAN',
                'EDUCATION_LOAN',
                'CREDIT_CARD',
                'PERSONAL_LOAN',
                'MSME_LOAN',
                'SERVICES_LOAN',
                'OTHER_LOAN'
            ]
            
            return np.random.choice(loan_type_map, n_samples, p=probs)
        else:
            # Fallback: equal distribution
            return np.random.choice(['HOME_LOAN', 'PERSONAL_LOAN', 'VEHICLE_LOAN', 'EDUCATION_LOAN'], n_samples)
    
    def _assign_interest_rates(self, loan_types):
        """Assign interest rates based on loan type using RBI priors"""
        priors = self.priors
        interest_rates = priors.get('interest_rates', {})
        n_samples = len(loan_types)
        
        rates = []
        for loan_type in loan_types:
            if loan_type == 'HOME_LOAN':
                rate_info = interest_rates.get('home_loans', {})
                min_rate = rate_info.get('min', 0.0836)
                max_rate = rate_info.get('max', 0.0876)
                mean_rate = rate_info.get('mean', 0.0862)
                # Sample from triangular distribution (skewed towards mean)
                rate = np.random.triangular(min_rate, mean_rate, max_rate)
            elif loan_type == 'VEHICLE_LOAN':
                rate_info = interest_rates.get('vehicle_loans', {})
                min_rate = rate_info.get('min', 0.0894)
                max_rate = rate_info.get('max', 0.1051)
                mean_rate = rate_info.get('mean', 0.0926)
                rate = np.random.triangular(min_rate, mean_rate, max_rate)
            elif loan_type == 'EDUCATION_LOAN':
                rate_info = interest_rates.get('education_loans', {})
                min_rate = rate_info.get('min', 0.0967)
                max_rate = rate_info.get('max', 0.1180)
                mean_rate = rate_info.get('mean', 0.1080)
                rate = np.random.triangular(min_rate, mean_rate, max_rate)
            elif loan_type == 'CREDIT_CARD':
                rate_info = interest_rates.get('credit_cards', {})
                min_rate = rate_info.get('min', 0.1425)
                max_rate = rate_info.get('max', 0.1825)
                mean_rate = rate_info.get('mean', 0.1625)
                rate = np.random.triangular(min_rate, mean_rate, max_rate)
            elif loan_type == 'MSME_LOAN':
                rate_info = interest_rates.get('msme_loans', {})
                min_rate = rate_info.get('min', 0.0958)
                max_rate = rate_info.get('max', 0.0973)
                mean_rate = rate_info.get('mean', 0.0961)
                rate = np.random.triangular(min_rate, mean_rate, max_rate)
            else:  # PERSONAL_LOAN, SERVICES_LOAN, OTHER_LOAN
                rate_info = interest_rates.get('personal_loans', {})
                min_rate = rate_info.get('min', 0.0955)
                max_rate = rate_info.get('max', 0.1156)
                mean_rate = rate_info.get('mean', 0.0983)
                rate = np.random.triangular(min_rate, mean_rate, max_rate)
            
            rates.append(rate)
        
        return np.array(rates)
    
    def _calculate_loan_amounts(self, monthly_incomes, loan_types):
        """Calculate loan amounts using loan-to-income ratios"""
        priors = self.priors
        loan_stats = priors.get('loan_stats', {})
        lti_info = loan_stats.get('loan_to_income_ratio', {})
        
        min_lti = lti_info.get('min', 0.5)  # 6 months
        max_lti = lti_info.get('max', 24.0)  # 2 years
        mean_lti = lti_info.get('mean', 10.0)  # 10 months
        
        loan_amounts = []
        for income, loan_type in zip(monthly_incomes, loan_types):
            # Different LTI ratios by loan type
            if loan_type == 'HOME_LOAN':
                # Home loans: higher LTI (up to 5 years income)
                lti = np.random.triangular(12, 36, 60)  # 1-5 years
            elif loan_type == 'VEHICLE_LOAN':
                # Vehicle loans: moderate LTI (1-2 years)
                lti = np.random.triangular(6, 12, 24)
            elif loan_type == 'EDUCATION_LOAN':
                # Education loans: moderate LTI
                lti = np.random.triangular(6, 12, 24)
            elif loan_type == 'CREDIT_CARD':
                # Credit cards: low limit (1-3 months)
                lti = np.random.triangular(0.5, 1.5, 3)
            else:
                # Personal/MSME/Other: use priors
                lti = np.random.triangular(min_lti, mean_lti, max_lti)
            
            loan_amount = income * lti
            # Apply reasonable bounds
            if loan_type == 'CREDIT_CARD':
                loan_amount = np.clip(loan_amount, 10000, 500000)  # Credit card limits
            elif loan_type == 'HOME_LOAN':
                loan_amount = np.clip(loan_amount, 500000, 10000000)  # Home loans
            else:
                loan_amount = np.clip(loan_amount, 50000, 5000000)  # Other loans
            
            loan_amounts.append(loan_amount)
        
        return np.array(loan_amounts)
    
    def _assign_loan_tenures(self, loan_types):
        """Assign loan tenures based on loan type"""
        priors = self.priors
        loan_stats = priors.get('loan_stats', {})
        tenure_info = loan_stats.get('tenure', {})
        
        tenures = []
        for loan_type in loan_types:
            if loan_type == 'HOME_LOAN':
                # Home loans: 10-30 years (120-360 months)
                tenure = np.random.choice([120, 180, 240, 300, 360], p=[0.1, 0.2, 0.3, 0.25, 0.15])
            elif loan_type == 'VEHICLE_LOAN':
                # Vehicle loans: 1-5 years (12-60 months)
                tenure = np.random.choice([12, 24, 36, 48, 60], p=[0.1, 0.3, 0.35, 0.15, 0.1])
            elif loan_type == 'EDUCATION_LOAN':
                # Education loans: 5-10 years (60-120 months)
                tenure = np.random.choice([60, 72, 84, 96, 120], p=[0.2, 0.3, 0.25, 0.15, 0.1])
            elif loan_type == 'CREDIT_CARD':
                # Credit cards: revolving, set to 0 or 12 months for tracking
                tenure = 12  # Revolving credit
            elif loan_type == 'MSME_LOAN':
                # MSME loans: 1-5 years
                tenure = np.random.choice([12, 24, 36, 48, 60], p=[0.15, 0.3, 0.3, 0.15, 0.1])
            else:  # PERSONAL_LOAN, SERVICES_LOAN, OTHER_LOAN
                # Personal loans: use prior distribution
                personal_tenures = tenure_info.get('personal_loans', [12, 24, 36, 48, 60])
                default_dist = tenure_info.get('default_distribution', [0.2, 0.3, 0.25, 0.15, 0.1])
                tenure = np.random.choice(personal_tenures, p=default_dist)
            
            tenures.append(tenure)
        
        return np.array(tenures)
    
    def _assign_bank_groups(self, n_samples):
        """Assign bank groups based on RBI credit share data"""
        priors = self.priors
        banking_access = priors.get('banking_access', {})
        credit_share = banking_access.get('credit_share', {})
        
        psb_prob = credit_share.get('public_sector_banks', 0.597)
        pvb_prob = credit_share.get('private_banks', 0.380)
        fb_prob = credit_share.get('foreign_banks', 0.023)
        
        # Normalize probabilities
        total = psb_prob + pvb_prob + fb_prob
        if total > 0:
            psb_prob /= total
            pvb_prob /= total
            fb_prob /= total
        
        bank_groups = np.random.choice(
            ['PSB', 'PVB', 'FB'],
            n_samples,
            p=[psb_prob, pvb_prob, fb_prob]
        )
        
        return bank_groups
    
    def _assign_default_rates_by_loan_type(self, loan_types):
        """Assign default rates based on loan type and RBI NPA data"""
        priors = self.priors
        default_rates = priors.get('default_rates', {})
        
        rates = []
        for loan_type in loan_types:
            if loan_type == 'HOME_LOAN':
                rate = default_rates.get('home_loans', 0.01)
            elif loan_type == 'CREDIT_CARD':
                rate = default_rates.get('credit_cards', 0.02)
            elif loan_type == 'MSME_LOAN':
                rate = default_rates.get('microfinance', 0.063)  # Use agriculture rate as proxy
            else:
                rate = default_rates.get('overall_retail', 0.02)
            
            rates.append(rate)
        
        return np.array(rates)
    
    def _generate_from_priors(self, n_samples):
        """Generate synthetic data purely from statistical priors"""
        priors = self.priors
        
        # Extract parameters
        urban_mean = priors.get('income_stats', {}).get('urban', {}).get('mean', 50000)
        urban_std = priors.get('income_stats', {}).get('urban', {}).get('std', 30000)
        age_mean = priors.get('demographics', {}).get('age', {}).get('mean', 35)
        age_std = priors.get('demographics', {}).get('age', {}).get('std', 12)
        credit_mean = priors.get('credit_score', {}).get('mean', 650)
        credit_std = priors.get('credit_score', {}).get('std', 100)
        
        # Get Census states and GSDP data
        census_states = priors.get('census_states', [])
        gsdp_data = priors.get('gsdp_manufacturing', {})
        worker_types = priors.get('worker_types', [])
        
        # Generate base demographics
        data = pd.DataFrame({
            'AGE': np.random.normal(age_mean, age_std, n_samples).clip(18, 70).astype(int),
            'MONTHLY_INCOME': np.random.lognormal(np.log(urban_mean), 0.5, n_samples).clip(10000, 500000),
            'CREDIT_SCORE': np.random.normal(credit_mean, credit_std, n_samples).clip(300, 900).astype(int),
        })
        
        # Add Indian market features (state, worker type) before loan calculations
        if census_states:
            data['STATE'] = np.random.choice(census_states, n_samples)
            # Add GSDP-based income adjustment
            if gsdp_data:
                state_gsdp = [gsdp_data.get(state, 0) for state in data['STATE']]
                gsdp_normalized = np.array(state_gsdp) / max(gsdp_data.values()) if max(gsdp_data.values()) > 0 else np.ones(n_samples)
                income_multiplier = 1 + 0.2 * (gsdp_normalized - 0.5)
                data['MONTHLY_INCOME'] = (data['MONTHLY_INCOME'] * income_multiplier).clip(10000, 500000)
        
        if worker_types:
            data['WORKER_TYPE'] = np.random.choice(worker_types, n_samples)
        
        # Assign loan types based on RBI distributions
        data['LOAN_TYPE'] = self._assign_loan_types(n_samples)
        
        # Calculate loan amounts using loan-to-income ratios
        data['LOAN_AMOUNT'] = self._calculate_loan_amounts(data['MONTHLY_INCOME'].values, data['LOAN_TYPE'].values)
        
        # Assign interest rates based on loan type
        data['INTEREST_RATE'] = self._assign_interest_rates(data['LOAN_TYPE'].values)
        
        # Assign loan tenures based on loan type
        data['LOAN_TENURE_MONTHS'] = self._assign_loan_tenures(data['LOAN_TYPE'].values)
        
        # Assign bank groups
        data['BANK_GROUP'] = self._assign_bank_groups(n_samples)
        
        # Assign default rates by loan type and generate default flags
        default_rates_by_type = self._assign_default_rates_by_loan_type(data['LOAN_TYPE'].values)
        data['DEFAULT_FLAG'] = np.array([
            np.random.binomial(1, rate) for rate in default_rates_by_type
        ])
        
        # Calculate monthly payment (EMI) using standard formula
        # EMI = [P x R x (1+R)^N] / [(1+R)^N - 1]
        # where P = principal, R = monthly interest rate, N = tenure in months
        monthly_rates = data['INTEREST_RATE'].values / 12
        tenure_months = data['LOAN_TENURE_MONTHS'].values
        principal = data['LOAN_AMOUNT'].values
        loan_types_arr = data['LOAN_TYPE'].values
        
        # Handle credit cards (revolving credit, no fixed EMI)
        emi = np.zeros(n_samples)
        
        for i in range(n_samples):
            if loan_types_arr[i] == 'CREDIT_CARD':
                # Credit card: use minimum payment (typically 5% of outstanding)
                emi[i] = principal[i] * 0.05
            else:
                # Regular loan EMI calculation
                p = principal[i]
                r = monthly_rates[i]
                n = tenure_months[i]
                if r > 0 and n > 0:
                    emi[i] = p * r * ((1 + r) ** n) / (((1 + r) ** n) - 1)
                else:
                    emi[i] = 0
        
        data['MONTHLY_PAYMENT'] = emi
        
        # Add RBI compliance flags
        data = self._add_rbi_compliance_flags(data)
        
        return data
    
    def _add_rbi_compliance_flags(self, df):
        """Add RBI compliance flags to dataframe"""
        priors = self.priors
        rbi_flags = priors.get('rbi_compliance_flags', {})
        
        df = df.copy()
        n_samples = len(df)
        
        # NSFR RSF Factor (Dec 29, 2023)
        if 'nsfr_rsf_factor' in rbi_flags:
            nsfr_values = rbi_flags['nsfr_rsf_factor'].get('values', [65, 100])
            df['NSFR_RSF_FACTOR'] = np.random.choice(nsfr_values, n_samples)
        
        # Inoperative Account Flag (Jan 01, 2024)
        if 'inoperative_flag' in rbi_flags:
            inop_rate = rbi_flags['inoperative_flag'].get('default_rate', 0.03)
            df['INOPERATIVE_FLAG'] = np.random.binomial(1, inop_rate, n_samples)
        
        # FX Hedging Flag (Jan 05, 2024)
        if 'fx_hedging_flag' in rbi_flags:
            fx_values = rbi_flags['fx_hedging_flag'].get('values', [0, 1])
            # Assume ~30% of accounts have FX hedging
            df['FX_HEDGING_FLAG'] = np.random.choice(fx_values, n_samples, p=[0.7, 0.3])
        
        # CP/NCD Flag (Jan 03, 2024)
        if 'cp_ncd_flag' in rbi_flags:
            cp_values = rbi_flags['cp_ncd_flag'].get('values', [0, 1])
            # Assume ~10% are CP/NCD related
            df['CP_NCD_FLAG'] = np.random.choice(cp_values, n_samples, p=[0.9, 0.1])
        
        return df
    
    def _add_indian_market_features(self, df):
        """Add Indian market-specific features"""
        priors = self.priors
        df = df.copy()
        n_samples = len(df)
        
        # Add STATE if not present
        if 'STATE' not in df.columns:
            census_states = priors.get('census_states', [])
            if census_states:
                df['STATE'] = np.random.choice(census_states, n_samples)
        
        # Add worker type if not present
        if 'WORKER_TYPE' not in df.columns:
            worker_types = priors.get('worker_types', [])
            if worker_types:
                df['WORKER_TYPE'] = np.random.choice(worker_types, n_samples)
        
        # Add currency (INR for Indian context)
        df['CURRENCY'] = 'INR'
        
        return df
    
    def save_synthetic_data(self, synthetic_data, output_path='data/synthetic_credit_data_v0.1.parquet'):
        """Save synthetic data to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        synthetic_data.to_parquet(output_file, index=False)
        print(f"💾 Saved synthetic data to: {output_file.absolute()}")
        
        # Also save as CSV for easy inspection
        csv_path = output_file.with_suffix('.csv')
        synthetic_data.to_csv(csv_path, index=False)
        print(f"💾 Also saved as CSV: {csv_path.absolute()}")


def main():
    """Example usage"""
    print("="*70)
    print("HYBRID SYNTHETIC DATA GENERATOR")
    print("="*70)
    
    # Paths
    kaggle_path = "data/kaggle_home_credit/application_train.csv"
    priors_path = "config/priors_template.yaml"
    
    # Initialize generator
    generator = HybridSyntheticGenerator(
        kaggle_data_path=kaggle_path if Path(kaggle_path).exists() else None,
        priors_path=priors_path
    )
    
    # Generate synthetic data
    synthetic_data = generator.generate_synthetic(
        n_samples=10000,
        method='gaussian_copula',  # Use 'ctgan' for better quality (slower)
        use_kaggle_structure=True,
        use_indian_priors=True
    )
    
    # Save
    generator.save_synthetic_data(synthetic_data)
    
    # Display summary
    print("\n" + "="*70)
    print("SYNTHETIC DATA SUMMARY")
    print("="*70)
    print(synthetic_data.describe())
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

