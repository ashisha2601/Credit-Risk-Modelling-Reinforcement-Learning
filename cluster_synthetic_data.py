#!/usr/bin/env python3
"""
Clustering on Synthetic Credit Data

This script:
1. Loads a synthetic dataset (default: two-stage priors-only CTGAN 300k file)
2. Selects key numeric and categorical features
3. Applies preprocessing (scaling + one-hot encoding)
4. Runs MiniBatchKMeans clustering
5. Saves the dataset with a new CLUSTER_ID column and prints summary stats
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


def load_synthetic_data() -> pd.DataFrame:
    """Load the main synthetic dataset, with sensible fallbacks."""
    project_root = Path(__file__).parent
    candidates = [
        project_root / "data" / "synthetic_credit_two_stage_priors_ctgan_300k.parquet",
        project_root / "data" / "synthetic_credit_data_v0.4_hybrid_ctgan_200k.parquet",
        project_root / "data" / "synthetic_credit_data_v0.3_hybrid_ctgan.parquet",
    ]

    for path in candidates:
        if path.exists():
            print(f"Using synthetic dataset: {path}")
            return pd.read_parquet(path)

    raise FileNotFoundError(
        "No synthetic dataset found. Expected one of:\n"
        "  - data/synthetic_credit_two_stage_priors_ctgan_300k.parquet\n"
        "  - data/synthetic_credit_data_v0.4_hybrid_ctgan_200k.parquet\n"
        "  - data/synthetic_credit_data_v0.3_hybrid_ctgan.parquet"
    )


def build_clustering_pipeline(
    numeric_features,
    categorical_features,
    n_clusters: int = 6,
) -> Pipeline:
    """Build a preprocessing + clustering pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
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


def main(n_clusters: int = 6) -> pd.DataFrame:
    project_root = Path(__file__).parent

    print("=" * 80)
    print("CLUSTERING ON SYNTHETIC CREDIT DATA")
    print("=" * 80)

    # 1. Load synthetic dataset
    synthetic = load_synthetic_data()
    print(f"\nLoaded synthetic dataset with shape: {synthetic.shape}")

    # 2. Select features for clustering
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
        if col in synthetic.columns
    ]

    categorical_features = [
        col
        for col in [
            "LOAN_TYPE",
            "BANK_GROUP",
            "STATE",
            "WORKER_TYPE",
        ]
        if col in synthetic.columns
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No suitable features found for clustering.")

    print("\nFeatures used for clustering:")
    print(f"  Numeric: {numeric_features}")
    print(f"  Categorical: {categorical_features}")

    # 3. Build and fit clustering pipeline
    model = build_clustering_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_clusters=n_clusters,
    )

    print(f"\nFitting MiniBatchKMeans with {n_clusters} clusters...")
    cluster_labels = model.fit_predict(synthetic[numeric_features + categorical_features])
    synthetic["CLUSTER_ID"] = cluster_labels
    print("Clustering complete.")

    # 4. Save results
    output_path = project_root / "data" / "synthetic_credit_with_clusters.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    synthetic.to_csv(csv_path, index=False)

    print("\nSaved clustered dataset to:")
    print(f"  - {output_path}")
    print(f"  - {csv_path}")

    # 5. Print quick cluster summary
    print("\nCluster sizes:")
    print(synthetic["CLUSTER_ID"].value_counts().sort_index())

    if numeric_features:
        print("\nCluster-wise numeric means (first few columns):")
        summary = (
            synthetic.groupby("CLUSTER_ID")[numeric_features]
            .mean()
            .round(2)
        )
        print(summary.head())

    print("\n" + "=" * 80)
    print("CLUSTERING COMPLETE")
    print("=" * 80)

    return synthetic


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during clustering: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


