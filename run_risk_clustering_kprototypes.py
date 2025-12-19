#!/usr/bin/env python3
"""
Risk-Oriented Clustering on Synthetic Credit Data (K-Prototypes)

Implements Stage 2 of the roadmap:
- Use mixed-type clustering (k-prototypes) on synthetic credit data
- Derive natural borrower risk segments based on default rates and features

Outputs:
- data/synthetic_credit_with_risk_clusters.parquet / .csv
- results/cluster_profiles.csv (size, default rate, key stats per cluster)
"""

from pathlib import Path
import sys
from typing import List, Dict

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).parent


def load_synthetic_data() -> pd.DataFrame:
    """Load the main synthetic dataset (300k priors-only CTGAN or hybrid)."""
    candidates = [
        PROJECT_ROOT / "data" / "synthetic_credit_priors_only_100k.parquet",
        PROJECT_ROOT / "data" / "synthetic_credit_two_stage_priors_ctgan_300k.parquet",
        PROJECT_ROOT / "data" / "synthetic_credit_data_v0.4_hybrid_ctgan_200k.parquet",
        PROJECT_ROOT / "data" / "synthetic_credit_data_v0.3_hybrid_ctgan.parquet",
    ]

    for path in candidates:
        if path.exists():
            print(f"Using synthetic dataset for risk clustering: {path}")
            return pd.read_parquet(path)

    raise FileNotFoundError(
        "No synthetic dataset found. Expected one of:\n"
        "  - data/synthetic_credit_two_stage_priors_ctgan_300k.parquet\n"
        "  - data/synthetic_credit_data_v0.4_hybrid_ctgan_200k.parquet\n"
        "  - data/synthetic_credit_data_v0.3_hybrid_ctgan.parquet"
    )


def select_features_for_clustering(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Select numeric and categorical features as per Stage 2 abstract."""
    numeric_features = [
        col
        for col in [
            "AGE",
            "MONTHLY_INCOME",
            "LOAN_AMOUNT",
            "INTEREST_RATE",
            "LOAN_TENURE_MONTHS",
            "CREDIT_SCORE",
            "MONTHLY_PAYMENT",
        ]
        if col in df.columns
    ]

    categorical_features = [
        col
        for col in [
            "LOAN_TYPE",
            "BANK_GROUP",
            "STATE",
            "WORKER_TYPE",
        ]
        if col in df.columns
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No suitable features found for clustering.")

    print("\nFeatures used for risk clustering (Stage 2):")
    print(f"  Numeric: {numeric_features}")
    print(f"  Categorical: {categorical_features}")

    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
    }


def run_kprototypes_clustering(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    n_clusters: int = 5,
) -> np.ndarray:
    """
    Run k-prototypes on mixed-type data:
    - Scale numeric features
    - Keep categoricals as strings
    """
    # Prepare feature matrix
    feature_cols = numeric_features + categorical_features
    data = df[feature_cols].copy()

    # Scale numeric columns for balanced influence
    if numeric_features:
        scaler = StandardScaler()
        data[numeric_features] = scaler.fit_transform(data[numeric_features])

    # Ensure categorical columns are strings
    for col in categorical_features:
        data[col] = data[col].astype(str)

    # k-prototypes expects numpy array and categorical column indices
    matrix = data.to_numpy()
    categorical_indices = [data.columns.get_loc(col) for col in categorical_features]

    print(f"\nFitting K-Prototypes with {n_clusters} clusters...")
    kproto = KPrototypes(
        n_clusters=n_clusters,
        init="Huang",
        random_state=42,
        n_jobs=-1,
    )

    clusters = kproto.fit_predict(matrix, categorical=categorical_indices)
    print("K-Prototypes clustering complete.")

    return clusters


def build_cluster_profiles(
    df: pd.DataFrame,
    cluster_col: str,
    default_col: str,
    numeric_features: List[str],
) -> pd.DataFrame:
    """Create interpretable profiles for each cluster (size, default rate, key stats)."""
    profiles = []

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        size = len(cluster_data)
        default_rate = (
            cluster_data[default_col].mean() if default_col in cluster_data.columns else np.nan
        )

        profile = {
            "cluster_id": cluster_id,
            "size": size,
            "share": size / len(df),
            "default_rate": default_rate,
        }

        for col in numeric_features:
            profile[f"{col}_mean"] = cluster_data[col].mean()
            profile[f"{col}_median"] = cluster_data[col].median()

        profiles.append(profile)

    profiles_df = pd.DataFrame(profiles).sort_values("default_rate")
    return profiles_df


def assign_risk_segments(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable risk segments based on default_rate ranking:
    - Lowest default_rate → LOW_RISK
    - Highest default_rate → HIGH_RISK
    - Middle clusters → MEDIUM_RISK variants
    """
    profiles = profiles.copy()
    profiles = profiles.sort_values("default_rate").reset_index(drop=True)

    n = len(profiles)
    risk_labels = []
    for rank in range(n):
        if rank == 0:
            risk_labels.append("LOW_RISK")
        elif rank == n - 1:
            risk_labels.append("HIGH_RISK")
        else:
            risk_labels.append("MEDIUM_RISK")

    profiles["risk_segment"] = risk_labels
    return profiles


def main(n_clusters: int = 5) -> pd.DataFrame:
    print("=" * 80)
    print("RISK-ORIENTED CLUSTERING (K-PROTOTYPES, STAGE 2)")
    print("=" * 80)

    # 1. Load synthetic dataset
    synthetic = load_synthetic_data()
    print(f"\nLoaded synthetic dataset with shape: {synthetic.shape}")

    # Handle both DEFAULT_FLAG (priors-only) and TARGET (Kaggle-style)
    default_col = "DEFAULT_FLAG" if "DEFAULT_FLAG" in synthetic.columns else "TARGET"
    if default_col not in synthetic.columns:
        raise ValueError("No default flag column found (expected DEFAULT_FLAG or TARGET).")

    # 2. Select features according to abstract (Stage 2)
    feature_dict = select_features_for_clustering(synthetic)
    numeric_features = feature_dict["numeric"]
    categorical_features = feature_dict["categorical"]

    # 3. Run k-prototypes clustering
    cluster_labels = run_kprototypes_clustering(
        synthetic,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_clusters=n_clusters,
    )
    synthetic["CLUSTER_ID"] = cluster_labels

    # 4. Build cluster profiles and risk segments
    profiles = build_cluster_profiles(
        synthetic,
        cluster_col="CLUSTER_ID",
        default_col=default_col,
        numeric_features=numeric_features,
    )
    profiles_with_risk = assign_risk_segments(profiles)

    # Map risk segments back to main dataframe
    risk_map = dict(
        zip(
            profiles_with_risk["cluster_id"],
            profiles_with_risk["risk_segment"],
        )
    )
    synthetic["RISK_SEGMENT"] = synthetic["CLUSTER_ID"].map(risk_map)

    # 5. Save clustered data
    data_out = PROJECT_ROOT / "data" / "synthetic_credit_with_risk_clusters.parquet"
    data_out.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_parquet(data_out, index=False)

    csv_out = data_out.with_suffix(".csv")
    synthetic.to_csv(csv_out, index=False)

    print("\nSaved risk-clustered dataset to:")
    print(f"  - {data_out}")
    print(f"  - {csv_out}")

    # 6. Save profiles to results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    profiles_out = results_dir / "cluster_profiles.csv"
    profiles_with_risk.to_csv(profiles_out, index=False)

    print("\nCluster profiles (sorted by default rate):")
    print(profiles_with_risk)

    print("\nSaved cluster profiles to:")
    print(f"  - {profiles_out}")

    print("\n" + "=" * 80)
    print("RISK-ORIENTED CLUSTERING COMPLETE")
    print("=" * 80)

    return synthetic


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during risk-oriented clustering: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


