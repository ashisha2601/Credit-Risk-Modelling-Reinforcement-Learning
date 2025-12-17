#!/usr/bin/env python3
"""
Train a PPO reinforcement learning agent with preference-shaped rewards
to decide whether to grant credit on clustered synthetic data.

Prerequisites:
  1. Run `run_credit_risk_clustering_all.py` to generate:
       - data/kaggle_only_with_risk_clusters.parquet
       - data/rbi_priors_only_with_risk_clusters.parquet
       - data/integrated_hybrid_with_risk_clusters.parquet

High-level flow:
  - Load one clustered dataset (prefer the integrated hybrid).
  - Build a Gymnasium-compatible environment where:
       * Observation  = applicant features (scaled & encoded)
       * Action       = 0 (reject) / 1 (approve)
       * Reward       = combination of:
            - outcome-based signal (approve + non-default is good;
              approve + default is bad, etc.)
            - preference-based shaping that reflects lender preferences
              over LOW / MEDIUM / HIGH risk segments.
  - Train a PPO policy using stable-baselines3.
  - Evaluate on a held-out test split and save per-applicant decisions:
       - columns: [...original columns..., RL_DECISION, RL_DECISION_TEXT]
       - RL_DECISION: 1 = approve, 0 = reject
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def detect_default_col(df: pd.DataFrame) -> str:
    """Detect the default flag column."""
    if "DEFAULT_FLAG" in df.columns:
        return "DEFAULT_FLAG"
    if "TARGET" in df.columns:
        return "TARGET"
    raise ValueError("No default flag column found (expected DEFAULT_FLAG or TARGET).")


def get_clustered_dataset() -> Tuple[str, Path]:
    """
    Pick a clustered dataset in priority order:
      1. integrated_hybrid_with_risk_clusters.parquet
      2. rbi_priors_only_with_risk_clusters.parquet
      3. kaggle_only_with_risk_clusters.parquet
    """
    preferred_order: Dict[str, Path] = {
        "integrated_hybrid": DATA_DIR / "integrated_hybrid_with_risk_clusters.parquet",
        "rbi_priors_only": DATA_DIR / "rbi_priors_only_with_risk_clusters.parquet",
        "kaggle_only": DATA_DIR / "kaggle_only_with_risk_clusters.parquet",
    }
    for name, path in preferred_order.items():
        if path.exists():
            return name, path
    raise FileNotFoundError(
        "No clustered datasets found. Expected at least one of:\n"
        "  - data/integrated_hybrid_with_risk_clusters.parquet\n"
        "  - data/rbi_priors_only_with_risk_clusters.parquet\n"
        "  - data/kaggle_only_with_risk_clusters.parquet\n"
        "Run `run_credit_risk_clustering_all.py` first."
    )


def build_feature_preprocessor(
    df: pd.DataFrame,
    default_col: str,
    extra_drop_cols: Optional[List[str]] = None,
) -> Tuple[Pipeline, List[str]]:
    """
    Build a preprocessing pipeline (scaling + one-hot encoding) and
    return it along with the list of original feature columns used.
    """
    drop_cols = {default_col}
    drop_cols.update({"CLUSTER_ID", "RISK_SEGMENT"})
    if extra_drop_cols:
        drop_cols.update(extra_drop_cols)

    # Identify numeric and categorical features
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    for col, dtype in df.dtypes.items():
        if col in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    if not numeric_features and not categorical_features:
        raise ValueError("No usable features found for RL environment.")

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())],
    )
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    feature_cols = numeric_features + categorical_features
    return Pipeline(steps=[("preprocess", preprocessor)]), feature_cols


@dataclass
class PreferenceConfig:
    """
    Preference-based reward shaping configuration.

    For each risk segment and action we specify an additive bonus:
      bonus[segment][action] where action in {0=reject, 1=approve}
    """

    bonus: Dict[str, Dict[int, float]]

    @staticmethod
    def default() -> "PreferenceConfig":
        # Lender preferences:
        #   - LOW_RISK: prefer to APPROVE
        #   - MEDIUM_RISK: mildly prefer APPROVE
        #   - HIGH_RISK: prefer to REJECT
        return PreferenceConfig(
            bonus={
                "LOW_RISK": {0: -0.3, 1: +0.3},
                "MEDIUM_RISK": {0: -0.1, 1: +0.1},
                "HIGH_RISK": {0: +0.3, 1: -0.3},
            }
        )

    def get_bonus(self, segment: str, action: int) -> float:
        if segment not in self.bonus:
            return 0.0
        return self.bonus[segment].get(int(action), 0.0)


class CreditApprovalEnv(gym.Env):
    """
    Gymnasium environment for credit approval decisions on synthetic data.

    - Observation: preprocessed feature vector for an applicant.
    - Action: 0 = reject, 1 = approve.
    - Reward: outcome-based signal + preference-based shaping.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        X: np.ndarray,
        defaults: np.ndarray,
        segments: np.ndarray,
        preference_config: PreferenceConfig,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.X = X.astype(np.float32)
        self.defaults = defaults.astype(np.int32)
        self.segments = segments.astype(str)
        self.preference_config = preference_config

        self.n_samples = self.X.shape[0]
        self.max_steps = max_steps or self.n_samples

        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(self.n_samples)

        obs_dim = self.X.shape[1]
        self.action_space = spaces.Discrete(2)  # 0 = reject, 1 = approve
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._current_idx = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.indices)
        self._step_count = 0
        self._current_idx = 0
        obs = self.X[self.indices[self._current_idx]]
        return obs, {}

    def step(self, action: int):
        idx = self.indices[self._current_idx]
        default_flag = int(self.defaults[idx])
        segment = str(self.segments[idx])

        # Outcome-based reward:
        #  - Approve + non-default: strong positive
        #  - Approve + default: strong negative
        #  - Reject + non-default: small penalty (missed opportunity)
        #  - Reject + default: small positive (good risk control)
        if int(action) == 1:  # approve
            if default_flag == 0:
                base_reward = 1.0
            else:
                base_reward = -1.0
        else:  # reject
            if default_flag == 0:
                base_reward = -0.1
            else:
                base_reward = 0.3

        # Preference-based shaping
        bonus = self.preference_config.get_bonus(segment, int(action))
        reward = float(base_reward + bonus)

        self._step_count += 1
        self._current_idx += 1
        terminated = self._step_count >= self.max_steps or self._current_idx >= self.n_samples
        truncated = False

        if not terminated:
            obs = self.X[self.indices[self._current_idx]]
        else:
            obs = np.zeros_like(self.X[0], dtype=np.float32)

        info = {
            "default_flag": default_flag,
            "segment": segment,
            "base_reward": base_reward,
            "preference_bonus": bonus,
        }
        return obs, reward, terminated, truncated, info


def train_ppo_on_dataset(
    df_train: pd.DataFrame,
    preprocessor: Pipeline,
    feature_cols: List[str],
    default_col: str,
    preference_config: PreferenceConfig,
    total_timesteps: int,
    seed: int = 42,
) -> PPO:
    # Fit preprocessor on train
    X_train_raw = df_train[feature_cols].copy()
    X_train = preprocessor.fit_transform(X_train_raw)
    defaults_train = df_train[default_col].values
    segments_train = df_train.get("RISK_SEGMENT", pd.Series(["UNKNOWN"] * len(df_train))).values

    env = CreditApprovalEnv(
        X=X_train,
        defaults=defaults_train,
        segments=segments_train,
        preference_config=preference_config,
        max_steps=min(len(df_train), 5000),
        seed=seed,
    )

    vec_env = DummyVecEnv([lambda: env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=seed,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_policy_on_test(
    model: PPO,
    df_test: pd.DataFrame,
    preprocessor: Pipeline,
    feature_cols: List[str],
    default_col: str,
) -> pd.DataFrame:
    X_test_raw = df_test[feature_cols].copy()
    X_test = preprocessor.transform(X_test_raw)

    defaults = df_test[default_col].values.astype(int)

    actions: List[int] = []
    for i in range(X_test.shape[0]):
        obs = X_test[i].astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))

    actions_arr = np.array(actions, dtype=int)
    df_out = df_test.copy()
    df_out["RL_DECISION"] = actions_arr
    df_out["RL_DECISION_TEXT"] = np.where(actions_arr == 1, "APPROVE", "REJECT")

    # Compute evaluation metrics
    approve_mask = actions_arr == 1
    reject_mask = actions_arr == 0

    if approve_mask.any():
        default_rate_approved = defaults[approve_mask].mean()
    else:
        default_rate_approved = float("nan")

    if reject_mask.any():
        default_rate_rejected = defaults[reject_mask].mean()
    else:
        default_rate_rejected = float("nan")

    overall_default_rate = defaults.mean()
    approval_rate = approve_mask.mean()

    print("\nEVALUATION METRICS (RL POLICY ON TEST SET)")
    print("-----------------------------------------")
    print(f"Number of test samples:          {len(df_test)}")
    print(f"Approval rate (RL_DECISION=1):   {approval_rate:.3f}")
    print(f"Default rate (overall):          {overall_default_rate:.3f}")
    print(f"Default rate among approved:     {default_rate_approved:.3f}")
    print(f"Default rate among rejected:     {default_rate_rejected:.3f}")

    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train PPO with preference-based learning to decide credit approval "
            "on clustered synthetic datasets."
        )
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Total PPO training timesteps (default: 50000).",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for held-out test set (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    dataset_name, dataset_path = get_clustered_dataset()
    print("=" * 80)
    print("PPO + PREFERENCE-BASED RL FOR CREDIT APPROVAL")
    print("=" * 80)
    print(f"Using clustered dataset: {dataset_name} -> {dataset_path}")

    df = pd.read_parquet(dataset_path)
    print(f"Loaded dataset shape: {df.shape}")

    default_col = detect_default_col(df)
    if "RISK_SEGMENT" not in df.columns:
        raise ValueError(
            "RISK_SEGMENT column not found. Ensure you ran run_credit_risk_clustering_all.py first."
        )

    preprocessor, feature_cols = build_feature_preprocessor(
        df,
        default_col=default_col,
    )
    print("\nFeatures used for RL environment:")
    print(f"  Total: {len(feature_cols)}")
    print(f"  Example: {feature_cols[:10]}")

    df_train, df_test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df[default_col],
    )
    print(f"\nTrain size: {df_train.shape}, Test size: {df_test.shape}")

    preference_config = PreferenceConfig.default()

    print("\nTraining PPO agent...")
    model = train_ppo_on_dataset(
        df_train=df_train,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        default_col=default_col,
        preference_config=preference_config,
        total_timesteps=args.timesteps,
        seed=args.seed,
    )

    print("\nEvaluating PPO agent on test set...")
    df_results = evaluate_policy_on_test(
        model=model,
        df_test=df_test,
        preprocessor=preprocessor,
        feature_cols=feature_cols,
        default_col=default_col,
    )

    out_parquet = RESULTS_DIR / f"rl_credit_decisions_ppo_{dataset_name}.parquet"
    out_csv = out_parquet.with_suffix(".csv")
    df_results.to_parquet(out_parquet, index=False)
    df_results.to_csv(out_csv, index=False)

    print("\nSaved RL credit decisions to:")
    print(f"  Parquet: {out_parquet}")
    print(f"  CSV:     {out_csv}")
    print("\nRL_DECISION: 1 = approve, 0 = reject.")
    print("RL_DECISION_TEXT: human-readable label (APPROVE / REJECT).")
    print("\nDone.")


if __name__ == "__main__":
    main()


