# Detailed Project Roadmap - Credit Risk Modeling

## Project Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│         STAGE 1: SYNTHETIC DATA GENERATION                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Input: Statistical Priors (RBI/NPCI/OGD)             │  │
│  │ Process: Generative AI (CTGAN/TVAE/GaussianCopula)  │  │
│  │ Output: High-fidelity Synthetic Credit Dataset       │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 2: UNSUPERVISED RISK PROFILING                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Input: Synthetic Dataset                              │  │
│  │ Process: Clustering (k-prototypes/HDBSCAN/UMAP+GMM)  │  │
│  │ Output: Natural Risk Segments (no labels needed)     │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 3: RISK-ADJUSTED DEEP RL POLICY              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Input: Clustered Synthetic Data + Simulator          │  │
│  │ Process: Deep RL (PPO/DQN/CQL) with Risk Constraints │  │
│  │ Output: Optimal Lending Policy (approve/price/limit)│  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## STAGE 1: SYNTHETIC DATA GENERATION (Detailed Steps)

### Step 1.1: Data Schema Design

**Objective**: Define the structure of your synthetic credit dataset

**Fields to Include**:

```python
# Borrower Demographics
- age (numeric, 18-70)
- income (numeric, monthly in INR)
- occupation_category (categorical: Salaried, Self-Employed, etc.)
- education_level (categorical: High School, Graduate, Post-Graduate)
- location_tier (categorical: Tier-1 City, Tier-2, Tier-3, Rural)
- years_at_current_job (numeric)
- years_of_experience (numeric)

# Banking History
- existing_accounts_count (numeric, 0-10)
- bank_account_age_months (numeric)
- credit_card_count (numeric, 0-5)
- previous_loans_count (numeric, 0-10)
- avg_account_balance (numeric, INR)

# Credit Application
- loan_amount (numeric, INR)
- loan_tenure_months (numeric, 3-60)
- loan_purpose (categorical: Personal, Business, Education, etc.)
- requested_interest_rate (numeric, %)

# Credit History (Synthetic Bureau-like)
- credit_score (numeric, 300-900)
- past_delinquencies (numeric, 0-10)
- max_dpd_ever (numeric, days past due)
- current_overdue_amount (numeric, INR)
- utilization_ratio (numeric, 0-1)

# Behavioral Features
- monthly_expense_ratio (numeric, expense/income)
- savings_rate (numeric, %)
- transaction_frequency (numeric, per month)

# Target Variables (for downstream modeling)
- default_flag (binary, 0/1)
- days_past_due (numeric, if default)
- loss_given_default (numeric, %)
- risk_segment (derived from clustering, Stage 2)
```

**Action Items**:
- [ ] Create `config/data_schema.yaml` with all field definitions
- [ ] Define distributions for each field (based on RBI/NPCI stats)
- [ ] Set correlation constraints (e.g., income vs loan_amount)

### Step 1.2: Statistical Priors Extraction

**Objective**: Gather real-world statistics to inform synthetic generation

**Sources & Extraction Method**:

1. **RBI Publications** (Manual extraction - 2-3 hours)
   - Download latest "Financial Stability Report" or "Trends and Progress in Banking"
   - Extract:
     - Average default rate: ~2-3% (retail loans)
     - NPA ratio: ~5-7% (varies by sector)
     - Interest rate ranges: Personal loans (10-24%), Credit cards (24-36%)
     - Loan-to-income ratios: Typically 10-15x monthly income
   
2. **Census 2011 + PLFS** (Download and parse)
   - Income distributions by state/urban-rural
   - Education levels across India
   - Occupation distribution
   
3. **NPCI Statistics** (Reference only)
   - UPI transaction volumes (proxy for digital adoption)
   - Card usage patterns

**Action Items**:
- [ ] Download 2-3 RBI reports (PDF)
- [ ] Extract key statistics into `config/priors.yaml`
- [ ] Download Census demographics CSV if available
- [ ] Document source of each prior

### Step 1.3: Implement Synthetic Generator

**Objective**: Generate synthetic data using SDV (Synthetic Data Vault)

**Code Structure**:

```python
# File: src/synthetic_generator.py

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
import pandas as pd
import yaml

class CreditSyntheticGenerator:
    def __init__(self, schema_path, priors_path):
        # Load schema and priors
        self.schema = self.load_schema(schema_path)
        self.priors = self.load_priors(priors_path)
        
    def generate_base_dataset(self, n_samples=10000):
        """
        Generate base synthetic dataset using Gaussian Copula
        (faster, good for initial baseline)
        """
        # Create metadata from schema
        metadata = SingleTableMetadata()
        # Add fields...
        
        # Initialize synthesizer
        synthesizer = GaussianCopulaSynthesizer(metadata)
        
        # Fit on reference data (if available) or generate from priors
        if self.has_reference_data():
            synthesizer.fit(self.reference_data)
        else:
            # Generate from scratch using priors
            synthesizer.fit(self.generate_from_priors())
        
        # Generate synthetic data
        synthetic_data = synthesizer.sample(num_rows=n_samples)
        
        return synthetic_data
    
    def generate_high_fidelity_dataset(self, n_samples=10000):
        """
        Generate high-fidelity synthetic data using CTGAN
        (slower, better quality)
        """
        synthesizer = CTGANSynthesizer(metadata)
        # Similar process...
```

**Action Items**:
- [ ] Install SDV: `pip install sdv`
- [ ] Create `src/synthetic_generator.py`
- [ ] Implement base generator (Gaussian Copula)
- [ ] Test with small sample (1000 rows)
- [ ] Generate full dataset (10,000-50,000 rows)

### Step 1.4: Quality Evaluation

**Objective**: Verify synthetic data preserves statistical properties

**Metrics to Compute**:

1. **Univariate Distribution Match**
   - Kolmogorov-Smirnov (KS) test for numeric features
   - Chi-square test for categorical features
   - Target: KS < 0.1, Chi-square p-value > 0.05

2. **Correlation Preservation**
   - Pearson correlation for numeric-numeric
   - Cramér's V for categorical-categorical
   - Compare correlation matrices (synthetic vs priors)
   - Target: Correlation difference < 0.1

3. **Mutual Information**
   - Check feature-feature and feature-target relationships
   - Target: Preserve top 80% of MI values

4. **Privacy Checks**
   - Nearest neighbor distance (should be far from any real record)
   - Membership inference attack (AUC should be ~0.5)
   - k-map uniqueness (no unique combinations)

**Code**:

```python
# File: src/evaluation.py

from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

def evaluate_synthetic_quality(real_data, synthetic_data):
    metrics = {}
    
    # KS tests for numeric columns
    for col in numeric_columns:
        ks_stat, p_value = ks_2samp(real_data[col], synthetic_data[col])
        metrics[f'ks_{col}'] = ks_stat
    
    # Correlation preservation
    real_corr = real_data[numeric_columns].corr()
    synth_corr = synthetic_data[numeric_columns].corr()
    corr_diff = np.abs(real_corr - synth_corr).mean()
    metrics['correlation_diff'] = corr_diff
    
    return metrics
```

**Action Items**:
- [ ] Create `src/evaluation.py`
- [ ] Implement all quality metrics
- [ ] Generate evaluation report
- [ ] Document findings in `data_card.md`

---

## STAGE 2: UNSUPERVISED RISK PROFILING (Detailed Steps)

### Step 2.1: Feature Engineering for Clustering

**Objective**: Prepare features suitable for mixed-type clustering

**Preprocessing Steps**:

1. **Handle Missing Values**
   - Impute with mode (categorical) or median (numeric)

2. **Encode Categorical Variables**
   - Use target encoding (if target available) or frequency encoding
   - Or use k-prototypes which handles categoricals natively

3. **Scale Numeric Features**
   - RobustScaler (less sensitive to outliers)
   - Or StandardScaler if no outliers

4. **Create Derived Features**
   - Debt-to-income ratio
   - Credit utilization score
   - Payment stability index

**Code**:

```python
# File: src/feature_engineering.py

from sklearn.preprocessing import RobustScaler
import category_encoders as ce

class CreditFeatureEngineer:
    def __init__(self):
        self.scaler = RobustScaler()
        self.encoder = ce.TargetEncoder()
    
    def prepare_for_clustering(self, df):
        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Scale numeric
        df_scaled = df.copy()
        df_scaled[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        # Encode categorical (if not using k-prototypes)
        if not self.use_kprototypes:
            df_scaled = self.encoder.fit_transform(df_scaled, df['default_flag'])
        
        return df_scaled
```

**Action Items**:
- [ ] Create `src/feature_engineering.py`
- [ ] Implement preprocessing pipeline
- [ ] Test on synthetic dataset

### Step 2.2: Clustering Algorithm Selection

**Objective**: Choose and implement clustering algorithm

**Options**:

1. **k-prototypes** (Recommended for mixed data)
   - Handles numeric + categorical natively
   - K-means extension
   - Fast and interpretable

2. **HDBSCAN** (For finding natural clusters)
   - Density-based, finds clusters of varying densities
   - Handles noise (outliers)
   - Doesn't require k

3. **UMAP + GMM** (For complex manifolds)
   - UMAP for dimensionality reduction
   - GMM for clustering in reduced space

**Implementation**:

```python
# File: src/clustering.py

from kmodes.kprototypes import KPrototypes
import hdbscan
from sklearn.mixture import GaussianMixture
import umap

class RiskProfiler:
    def __init__(self, method='kprototypes', n_clusters=5):
        self.method = method
        self.n_clusters = n_clusters
    
    def fit_kprototypes(self, df, categorical_indices):
        """k-prototypes for mixed data"""
        kproto = KPrototypes(n_clusters=self.n_clusters, init='Huang', n_jobs=-1)
        clusters = kproto.fit_predict(df.values, categorical=categorical_indices)
        return clusters, kproto
    
    def fit_hdbscan(self, df):
        """HDBSCAN for natural clusters"""
        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
        clusters = clusterer.fit_predict(df)
        return clusters, clusterer
    
    def profile_clusters(self, df, clusters):
        """Analyze each cluster's characteristics"""
        profiles = {}
        for cluster_id in np.unique(clusters):
            cluster_data = df[clusters == cluster_id]
            profiles[cluster_id] = {
                'size': len(cluster_data),
                'avg_income': cluster_data['income'].mean(),
                'default_rate': cluster_data['default_flag'].mean(),
                'avg_credit_score': cluster_data['credit_score'].mean(),
                # ... more statistics
            }
        return profiles
```

**Action Items**:
- [ ] Install clustering libraries: `pip install kmodes hdbscan umap-learn`
- [ ] Create `src/clustering.py`
- [ ] Implement k-prototypes (start with this)
- [ ] Try different k values (3, 4, 5, 6)
- [ ] Evaluate cluster quality (silhouette score, interpretability)

### Step 2.3: Cluster Validation & Interpretation

**Objective**: Validate clusters and understand risk profiles

**Validation Metrics**:

1. **Silhouette Score** (-1 to 1, higher is better)
2. **Within-cluster variance** (lower is better)
3. **Stability** (clusters consistent across resamples)

**Interpretation**:

For each cluster, create a profile:
- Average income, age, credit score
- Default rate
- Loan characteristics
- Geographic/demographic patterns

**Expected Risk Profiles** (Example):

- **Cluster 0: Low Risk**
  - High income, high credit score, low utilization
  - Default rate: ~1%
  
- **Cluster 1: Medium Risk**
  - Moderate income, moderate credit score
  - Default rate: ~3-5%
  
- **Cluster 2: High Risk**
  - Low income, low credit score, high utilization
  - Default rate: ~10-15%

**Action Items**:
- [ ] Compute validation metrics
- [ ] Generate cluster profiles
- [ ] Visualize clusters (PCA/t-SNE plots)
- [ ] Document findings

---

## STAGE 3: RISK-ADJUSTED DEEP RL POLICY (Detailed Steps)

### Step 3.1: Define MDP (Markov Decision Process)

**Objective**: Frame lending as sequential decision problem

**State Space**:
```python
state = {
    'borrower_features': [income, age, credit_score, ...],
    'cluster_id': cluster_assignment,
    'current_loan_amount': requested_amount,
    'tenure': requested_tenure,
    'macro_indicator': [interest_rate_trend, economic_indicator],
    'portfolio_state': [current_portfolio_risk, utilization]
}
```

**Action Space** (Discrete for Review-1):
```python
actions = [
    'APPROVE_LOW_RATE',    # Approve with 12% APR
    'APPROVE_MEDIUM_RATE', # Approve with 18% APR
    'APPROVE_HIGH_RATE',   # Approve with 24% APR
    'REJECT'               # Decline application
]
```

**Reward Function**:
```python
def calculate_reward(action, state, outcome):
    if action == 'REJECT':
        return 0  # No gain, no loss
    
    # If approved
    interest_income = loan_amount * interest_rate * tenure/12
    funding_cost = loan_amount * cost_of_funds * tenure/12
    
    if outcome == 'default':
        loss = loan_amount * loss_given_default
        reward = interest_income - funding_cost - loss
    else:
        reward = interest_income - funding_cost
    
    # Risk adjustment
    risk_penalty = -lambda_risk * expected_loss
    
    return reward + risk_penalty
```

**Transition Model**:
```python
# Simulate loan outcome based on cluster and loan terms
def simulate_loan_outcome(cluster_id, loan_amount, interest_rate, tenure):
    # Get cluster-specific default probability
    base_pd = cluster_pd[cluster_id]
    
    # Adjust for loan characteristics
    pd_adjusted = base_pd * (1 + interest_rate/100) * (loan_amount/income_ratio)
    
    # Sample outcome
    default = np.random.binomial(1, pd_adjusted)
    
    return default
```

**Action Items**:
- [ ] Define state/action/reward in `src/rl_env.py`
- [ ] Implement simple transition model
- [ ] Test simulator with sample decisions

### Step 3.2: Implement Environment

**Objective**: Create Gym-compatible environment

**Code**:

```python
# File: src/credit_env.py

import gym
from gym import spaces
import numpy as np

class CreditLendingEnv(gym.Env):
    def __init__(self, synthetic_data, clusters):
        super().__init__()
        
        self.data = synthetic_data
        self.clusters = clusters
        self.current_idx = 0
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # 4 actions
        
    def reset(self):
        self.current_idx = np.random.randint(0, len(self.data))
        return self._get_state()
    
    def step(self, action):
        state = self._get_state()
        reward = self._calculate_reward(action, state)
        done = self.current_idx >= len(self.data) - 1
        
        self.current_idx += 1
        next_state = self._get_state() if not done else state
        
        return next_state, reward, done, {}
    
    def _get_state(self):
        row = self.data.iloc[self.current_idx]
        return row[feature_columns].values
    
    def _calculate_reward(self, action, state):
        # Implement reward logic
        ...
```

**Action Items**:
- [ ] Create `src/credit_env.py`
- [ ] Implement environment following Gym API
- [ ] Test with random actions

### Step 3.3: Train RL Agent

**Objective**: Train agent to learn optimal policy

**For Review-1: Start Simple**

**Option 1: Contextual Bandit** (Simplest)
```python
from sklearn.linear_model import LogisticRegression

class ContextualBandit:
    def __init__(self):
        self.models = {}  # One model per action
    
    def train(self, states, actions, rewards):
        for action in [0, 1, 2, 3]:
            mask = actions == action
            if mask.sum() > 0:
                self.models[action] = LogisticRegression()
                self.models[action].fit(states[mask], rewards[mask] > rewards[mask].median())
    
    def predict(self, state):
        # Choose action with highest expected reward
        scores = {}
        for action, model in self.models.items():
            scores[action] = model.predict_proba(state.reshape(1, -1))[0, 1]
        return max(scores, key=scores.get)
```

**Option 2: DQN** (More sophisticated)
```python
from stable_baselines3 import DQN

# Create environment
env = CreditLendingEnv(synthetic_data, clusters)

# Train DQN agent
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)
```

**Action Items**:
- [ ] Implement contextual bandit (faster for Review-1)
- [ ] OR implement DQN using stable-baselines3
- [ ] Train on synthetic dataset
- [ ] Evaluate against baseline (random policy, fixed policy)

### Step 3.4: Policy Evaluation

**Objective**: Compare RL policy vs baselines

**Baselines**:
1. **Random Policy**: Random approve/reject
2. **Fixed Policy**: Approve if credit_score > 650
3. **Rule-based**: Approve based on simple rules

**Metrics**:
- Expected return (total reward)
- Default rate
- Portfolio risk (CVaR)
- Approval rate
- Fairness metrics (across demographics)

**Action Items**:
- [ ] Implement baseline policies
- [ ] Run evaluation
- [ ] Generate comparison report

---

## REVIEW-1 DELIVERABLES CHECKLIST

### Data & Generation
- [ ] `data/synthetic_credit_data_v0.1.parquet` (10K+ rows)
- [ ] `data/data_card.md` (documentation of synthetic data)
- [ ] `config/data_schema.yaml` (schema definition)
- [ ] `config/priors.yaml` (statistical priors used)

### Clustering
- [ ] `models/cluster_model.pkl` (trained clustering model)
- [ ] `results/cluster_profiles.csv` (cluster characteristics)
- [ ] `results/cluster_visualization.png` (cluster plots)

### RL Policy
- [ ] `models/rl_policy.pkl` (trained RL agent)
- [ ] `results/policy_evaluation.csv` (comparison vs baselines)
- [ ] `results/policy_visualization.png` (policy insights)

### Notebooks
- [ ] `notebooks/01_synthetic_gen.ipynb` (complete with outputs)
- [ ] `notebooks/02_clustering.ipynb` (complete with outputs)
- [ ] `notebooks/03_rl_policy.ipynb` (complete with outputs)

### Documentation
- [ ] `REVIEW1_REPORT.md` (summary of findings)
- [ ] `PRIVACY_COMPLIANCE.md` (DPDPA compliance memo)
- [ ] `README.md` (project overview)

---

## TIMELINE FOR REVIEW-1 (2 Weeks)

### Week 1:
- **Day 1-2**: Download/reference datasets, extract priors
- **Day 3-4**: Implement synthetic generator, generate v0.1 dataset
- **Day 5**: Quality evaluation, create data card

### Week 2:
- **Day 1-2**: Implement clustering, generate profiles
- **Day 3-4**: Implement RL environment and agent, train policy
- **Day 5**: Evaluation, documentation, finalize notebooks

---

## NEXT STEPS AFTER REVIEW-1

1. Improve synthetic data fidelity (CTGAN instead of Gaussian Copula)
2. Add differential privacy (DP-SGD)
3. Implement more sophisticated RL (PPO, CQL)
4. Add fairness constraints
5. Stress testing under macro scenarios
6. Deploy model as API/service

