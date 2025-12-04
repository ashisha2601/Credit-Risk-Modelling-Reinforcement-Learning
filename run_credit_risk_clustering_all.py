#!/usr/bin/env python3
"""
Credit Risk-Based Clustering on All Synthetic Datasets

Datasets:
1. Kaggle-only CTGAN synthetic:
   - data/synthetic_credit_kaggle_only_ctgan_300k.parquet
2. RBI priors-only synthetic:
   - data/synthetic_credit_priors_only_300k.parquet
3. Integrated hybrid (Kaggle + RBI priors via hybrid pipeline):
   - data/synthetic_credit_data_v0.4_hybrid_ctgan_200k.parquet

For each dataset, this script:
- Selects risk-relevant features (demographics, income, loan, credit score, EMI, etc.)
- Applies scaling + one-hot encoding
- Runs MiniBatchKMeans with a fixed number of clusters
- Computes default-rate-based risk segments (LOW / MEDIUM / HIGH)
- Saves clustered datasets and cluster profile CSVs under data/ and results/
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_datasets() -> Dict[str, Path]:
    """Return mapping from dataset name to file path."""
    mapping = {
        "kaggle_only": PROJECT_ROOT
        / "data"
        / "synthetic_credit_kaggle_only_ctgan_300k.parquet",
        "rbi_priors_only": PROJECT_ROOT
        / "data"
        / "synthetic_credit_priors_only_300k.parquet",
        "integrated_hybrid": PROJECT_ROOT
        / "data"
        / "synthetic_credit_data_v0.4_hybrid_ctgan_200k.parquet",
    }
    # Filter only existing files
    existing = {name: path for name, path in mapping.items() if path.exists()}
    if not existing:
        raise FileNotFoundError(
            "No synthetic datasets found. Expected at least one of:\n"
            "  - data/synthetic_credit_kaggle_only_ctgan_300k.parquet\n"
            "  - data/synthetic_credit_priors_only_300k.parquet\n"
            "  - data/synthetic_credit_data_v0.4_hybrid_ctgan_200k.parquet"
        )
    return existing


def select_features(df: pd.DataFrame) -> Tuple[List[str], List[str], str]:
    """Select numeric/categorical features and detect default column."""
    default_col = "DEFAULT_FLAG" if "DEFAULT_FLAG" in df.columns else "TARGET"
    if default_col not in df.columns:
        raise ValueError("No default flag column found (expected DEFAULT_FLAG or TARGET).")

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

    return numeric_features, categorical_features, default_col


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    n_clusters: int,
) -> Pipeline:
    """Build preprocessing + MiniBatchKMeans clustering pipeline."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())],
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clusterer = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=4096,
        n_init="auto",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("cluster", clusterer),
        ]
    )
    return model


def assign_risk_segments(
    df: pd.DataFrame,
    cluster_col: str,
    default_col: str,
) -> pd.DataFrame:
    """Assign risk segment labels based on default rates per cluster."""
    stats = (
        df.groupby(cluster_col)[default_col]
        .mean()
        .reset_index()
        .rename(columns={default_col: "default_rate"})
    )
    # Rank clusters by default rate
    stats = stats.sort_values("default_rate").reset_index(drop=True)
    n = len(stats)
    labels = []
    for rank in range(n):
        if rank == 0:
            labels.append("LOW_RISK")
        elif rank == n - 1:
            labels.append("HIGH_RISK")
        else:
            labels.append("MEDIUM_RISK")
    stats["RISK_SEGMENT"] = labels

    # Map back to dataframe
    mapping = dict(zip(stats["CLUSTER_ID"], stats["RISK_SEGMENT"]))
    df["RISK_SEGMENT"] = df[cluster_col].map(mapping)

    return stats


def cluster_dataset(name: str, path: Path, n_clusters: int = 5) -> None:
    print("\n" + "=" * 80)
    print(f"CREDIT RISK CLUSTERING: {name}")
    print("=" * 80)
    print(f"Loading dataset: {path}")

    df = pd.read_parquet(path)
    print(f"Shape: {df.shape}")

    numeric_features, categorical_features, default_col = select_features(df)

    print("\nFeatures for clustering:")
    print(f"  Numeric: {numeric_features}")
    print(f"  Categorical: {categorical_features}")
    print(f"  Default column: {default_col}")

    model = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_clusters=n_clusters,
    )

    feature_cols = numeric_features + categorical_features
    print(f"\nFitting MiniBatchKMeans (k={n_clusters}) on {len(feature_cols)} features...")
    cluster_labels = model.fit_predict(df[feature_cols])
    df["CLUSTER_ID"] = cluster_labels
    print("Clustering complete.")

    # Assign risk segments and build profiles
    profiles = assign_risk_segments(df, "CLUSTER_ID", default_col)

    # Save clustered dataset
    data_out = PROJECT_ROOT / "data" / f"{name}_with_risk_clusters.parquet"
    df.to_parquet(data_out, index=False)
    csv_out = data_out.with_suffix(".csv")
    df.to_csv(csv_out, index=False)

    # Save profiles
    profiles_out = RESULTS_DIR / f"cluster_profiles_{name}.csv"
    profiles.to_csv(profiles_out, index=False)

    print("\nCluster profiles (sorted by default rate):")
    print(profiles)
    print("\nSaved clustered data and profiles to:")
    print(f"  Data (parquet): {data_out}")
    print(f"  Data (csv):     {csv_out}")
    print(f"  Profiles:       {profiles_out}")


def main() -> None:
    datasets = get_datasets()

    print("=" * 80)
    print("CREDIT RISK-BASED CLUSTERING ON ALL SYNTHETIC DATASETS")
    print("=" * 80)
    print("Datasets found:")
    for name, path in datasets.items():
        print(f"  - {name}: {path}")

    # Use a consistent number of clusters across datasets for comparability
    for name, path in datasets.items():
        cluster_dataset(name, path, n_clusters=5)

    print("\n" + "=" * 80)
    print("ALL DATASETS CLUSTERED (CREDIT RISK SEGMENTS ASSIGNED)")
    print("=" * 80)


if __name__ == "__main__":
    main()


