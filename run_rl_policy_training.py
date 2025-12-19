#!/usr/bin/env python3
"""
Stage 3: Risk-Adjusted Reinforcement Learning Policy
----------------------------------------------------
Trains a Deep RL agent (PPO) to make lending decisions based on the 
synthetic credit data and risk clusters generated in Stage 2.

The agent learns to balance:
- Profit maximization (Interest income)
- Risk minimization (Avoiding defaults)
- Fairness/Compliance (via constrained observation space)

Inputs:
- data/synthetic_credit_with_risk_clusters.parquet

Outputs:
- models/ppo_credit_policy.zip
- results/rl_policy_evaluation.csv
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_credit_with_risk_clusters.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

class CreditLendingEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    
    State: [Income, Loan_Amount, Credit_Score, Interest_Rate, Cluster_ID_Encoded]
    Action: 0 (Reject), 1 (Approve)
    Reward: 
      - Reject: 0
      - Approve & Repay: +Interest Amount
      - Approve & Default: -Principal Amount
    """
    
    def __init__(self, df, feature_cols):
        super(CreditLendingEnv, self).__init__()
        
        self.df = df
        self.feature_cols = feature_cols
        
        # Define action and observation space
        # Action: 0=Reject, 1=Approve
        self.action_space = spaces.Discrete(2)
        
        # Observation: Continuous features
        n_features = len(feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
        # Internal state
        self.current_step = 0
        self.max_steps = len(df)
        self.data_matrix = df[feature_cols].values.astype(np.float32)
        
        # Pre-calculate rewards for efficiency
        # If Approved (1):
        #   Repay (Default=0): Reward = Loan * Rate
        #   Default (Default=1): Reward = -Loan
        
        # Note: We need LOAN_AMOUNT and INTEREST_RATE for reward calc
        # We assume they are in the dataframe but maybe scaled in data_matrix
        # So we keep raw values for reward calculation
        self.raw_loans = df['LOAN_AMOUNT'].values
        self.raw_rates = df['INTEREST_RATE'].values
        self.defaults = df['DEFAULT_FLAG'].values if 'DEFAULT_FLAG' in df.columns else df['TARGET'].values
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Optional: Shuffle data at start of episode
        # indices = np.random.permutation(len(self.df))
        # self.data_matrix = self.data_matrix[indices]
        # self.raw_loans = self.raw_loans[indices]
        # self.raw_rates = self.raw_rates[indices]
        # self.defaults = self.defaults[indices]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        return self.data_matrix[self.current_step]
    
    def step(self, action):
        done = False
        truncated = False
        reward = 0.0
        
        if action == 1:  # Approve
            is_default = self.defaults[self.current_step]
            loan_amt = self.raw_loans[self.current_step]
            rate = self.raw_rates[self.current_step]
            
            if is_default:
                # Loss = Principal
                reward = -loan_amt
            else:
                # Gain = Interest (Simple Interest approx for reward)
                reward = loan_amt * rate
        
        else:  # Reject
            # Opportunity cost? Or just 0.
            # Usually 0 is safe. 
            # Could penalize rejecting good customers (Opportunity Loss)
            # if self.defaults[self.current_step] == 0:
            #     reward = -(self.raw_loans[self.current_step] * self.raw_rates[self.current_step]) * 0.1
            reward = 0.0
            
        self.current_step += 1
        if self.current_step >= self.max_steps - 1:
            done = True
            
        return self._get_observation(), reward, done, truncated, {}

    def render(self):
        pass

def preprocess_data(df):
    """
    Preprocess data for RL environment.
    Scales numerical features and encodes categoricals.
    """
    df = df.copy()
    
    # Feature Selection
    numeric_features = ['MONTHLY_INCOME', 'LOAN_AMOUNT', 'CREDIT_SCORE', 'LOAN_TENURE_MONTHS', 'INTEREST_RATE']
    
    # Encode Cluster ID if present
    if 'CLUSTER_ID' in df.columns:
        le = LabelEncoder()
        df['CLUSTER_ID_ENC'] = le.fit_transform(df['CLUSTER_ID'])
        numeric_features.append('CLUSTER_ID_ENC')
        
    # Scale features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_features])
    
    # Create scaled dataframe columns
    df_scaled = pd.DataFrame(scaled_data, columns=numeric_features)
    
    # Add back raw columns needed for reward
    df_scaled['LOAN_AMOUNT'] = df['LOAN_AMOUNT'].values # Overwrite with raw for reward? No, we need both.
    
    # Let's keep a clean matrix for observation
    obs_df = pd.DataFrame(scaled_data, columns=numeric_features)
    
    # But pass the FULL df (with raw values) to the Env, 
    # and tell Env which columns are observations
    
    # Re-attach targets and raw values to obs_df for the Env wrapper
    obs_df['LOAN_AMOUNT_RAW'] = df['LOAN_AMOUNT'].values
    obs_df['INTEREST_RATE_RAW'] = df['INTEREST_RATE'].values
    obs_df['DEFAULT_FLAG'] = df['DEFAULT_FLAG'].values if 'DEFAULT_FLAG' in df.columns else df['TARGET'].values
    
    return obs_df, numeric_features

def main():
    print("="*80)
    print("STAGE 3: RISK-ADJUSTED RL POLICY TRAINING")
    print("="*80)
    
    # 1. Load Data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found at {DATA_PATH}. Please run Stage 2 (Clustering) first.")
        
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded {len(df)} rows.")
    
    # 2. Preprocess
    print("Preprocessing data for RL environment...")
    # We need to distinguish between Observation features (Scaled) and Reward features (Raw)
    # Let's simplify: pass raw DF to env, and let Env handle scaling? 
    # Or pre-scale. Let's pre-scale observations.
    
    # Features to observe
    obs_features = ['MONTHLY_INCOME', 'LOAN_AMOUNT', 'CREDIT_SCORE', 'INTEREST_RATE']
    if 'CLUSTER_ID' in df.columns:
         # Simple numeric encoding for cluster
         df['CLUSTER_ENC'] = df['CLUSTER_ID'].astype(str).astype('category').cat.codes
         obs_features.append('CLUSTER_ENC')
         
    # Scale observations
    scaler = StandardScaler()
    df_obs = df.copy()
    df_obs[obs_features] = scaler.fit_transform(df[obs_features])
    
    # The Env needs: Scaled Obs Features AND Raw Reward Features (Loan, Rate, Default)
    # We will construct a DF that has both
    env_df = df_obs.copy()
    # Ensure raw reward cols are present (they might have been overwritten if in obs_features)
    # Actually, scaler overwrote them. Let's restore raw for reward calculation columns
    # We can rename scaled columns
    
    final_df = pd.DataFrame()
    for col in obs_features:
        final_df[f"obs_{col}"] = df_obs[col]
        
    final_df['LOAN_AMOUNT'] = df['LOAN_AMOUNT']
    final_df['INTEREST_RATE'] = df['INTEREST_RATE']
    final_df['DEFAULT_FLAG'] = df['DEFAULT_FLAG'] if 'DEFAULT_FLAG' in df.columns else df['TARGET']
    
    obs_cols = [f"obs_{col}" for col in obs_features]
    
    # Split Train/Test (Time-based or random)
    train_size = int(len(final_df) * 0.8)
    train_df = final_df.iloc[:train_size].reset_index(drop=True)
    test_df = final_df.iloc[train_size:].reset_index(drop=True)
    
    print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")
    
    # 3. Create Environments
    print("Initializing Environments...")
    train_env = DummyVecEnv([lambda: CreditLendingEnv(train_df, obs_cols)])
    test_env = DummyVecEnv([lambda: CreditLendingEnv(test_df, obs_cols)])
    
    # 4. Train PPO Agent
    print("\n🚀 Training PPO Agent...")
    model = PPO("MlpPolicy", train_env, verbose=1, learning_rate=0.0003)
    model.learn(total_timesteps=20000) # Short training for demo
    
    # Save model
    model_path = MODELS_DIR / "ppo_credit_policy"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # 5. Evaluate
    print("\n📊 Evaluating Policies...")
    
    # Baseline 1: Always Approve
    print("Evaluating Baseline: Always Approve...")
    total_reward_approve = 0
    test_env_raw = CreditLendingEnv(test_df, obs_cols)
    obs, _ = test_env_raw.reset()
    done = False
    while not done:
        obs, reward, done, _, _ = test_env_raw.step(1) # Always 1
        total_reward_approve += reward
        
    # Baseline 2: Random
    print("Evaluating Baseline: Random...")
    total_reward_random = 0
    test_env_raw.reset()
    done = False
    while not done:
        action = test_env_raw.action_space.sample()
        obs, reward, done, _, _ = test_env_raw.step(action)
        total_reward_random += reward
        
    # RL Policy
    print("Evaluating RL Policy...")
    total_reward_rl = 0
    test_env_raw.reset()
    done = False
    obs, _ = test_env_raw.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = test_env_raw.step(action)
        total_reward_rl += reward
        
    # Results
    results = pd.DataFrame({
        'Policy': ['Always Approve', 'Random', 'PPO (RL)'],
        'Total Profit': [total_reward_approve, total_reward_random, total_reward_rl],
        'Avg Profit per Loan': [
            total_reward_approve/len(test_df),
            total_reward_random/len(test_df),
            total_reward_rl/len(test_df)
        ]
    })
    
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(results)
    
    results.to_csv(RESULTS_DIR / "policy_evaluation.csv", index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results, x='Policy', y='Total Profit', palette='viridis')
    plt.title('Total Profit by Policy Strategy')
    plt.ylabel('Net Profit (INR)')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "policy_comparison.png")
    print(f"Saved plot to {RESULTS_DIR}/policy_comparison.png")

if __name__ == "__main__":
    main()
