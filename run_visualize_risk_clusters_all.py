#!/usr/bin/env python3
"""
Visualize Credit Risk Clusters for All Synthetic Datasets

For each clustered dataset:
  - kaggle_only_with_risk_clusters.parquet
  - rbi_priors_only_with_risk_clusters.parquet
  - integrated_hybrid_with_risk_clusters.parquet (if present)

This script produces:
  - Cluster size bar plot
  - Default rate bar plot
  - (Optional) PCA scatter on numeric features, colored by cluster

All figures are saved under results/ with dataset-specific filenames.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def available_clustered_datasets():
    candidates = {
        "kaggle_only": PROJECT_ROOT
        / "data"
        / "kaggle_only_with_risk_clusters.parquet",
        "rbi_priors_only": PROJECT_ROOT
        / "data"
        / "rbi_priors_only_with_risk_clusters.parquet",
        "integrated_hybrid": PROJECT_ROOT
        / "data"
        / "integrated_hybrid_with_risk_clusters.parquet",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def get_default_col(df: pd.DataFrame) -> str:
    if "DEFAULT_FLAG" in df.columns:
        return "DEFAULT_FLAG"
    if "TARGET" in df.columns:
        return "TARGET"
    raise ValueError("No default column found (expected DEFAULT_FLAG or TARGET).")


def numeric_features_for_pca(df: pd.DataFrame) -> List[str]:
    candidates = [
        "AGE",
        "MONTHLY_INCOME",
        "LOAN_AMOUNT",
        "INTEREST_RATE",
        "LOAN_TENURE_MONTHS",
        "CREDIT_SCORE",
        "MONTHLY_PAYMENT",
    ]
    return [c for c in candidates if c in df.columns]


def plot_cluster_sizes(df: pd.DataFrame, name: str) -> None:
    counts = df["CLUSTER_ID"].value_counts().sort_index()

    plt.figure(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, color="#4c72b0")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of borrowers")
    plt.title(f"Cluster Sizes ({name})")
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()

    out_path = RESULTS_DIR / f"{name}_cluster_sizes.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved cluster size plot for {name} to: {out_path}")


def plot_default_rates(df: pd.DataFrame, name: str) -> None:
    default_col = get_default_col(df)
    stats = (
        df.groupby("CLUSTER_ID")[default_col]
        .mean()
        .reset_index()
        .rename(columns={default_col: "default_rate"})
    )

    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=stats, x="CLUSTER_ID", y="default_rate", color="#dd8452")
    plt.xlabel("Cluster ID")
    plt.ylabel("Default rate")
    plt.title(f"Default Rate by Cluster ({name})")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f"{x*100:.1f}%")
    )
    # Add percentage labels above bars
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height,
            f"{height*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.tight_layout()

    out_path = RESULTS_DIR / f"{name}_cluster_default_rates.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved default-rate plot for {name} to: {out_path}")


def plot_pca_scatter(df: pd.DataFrame, name: str) -> None:
    num_cols = numeric_features_for_pca(df)
    if not num_cols:
        print(f"No numeric features for PCA scatter for {name}. Skipping.")
        return

    if len(num_cols) < 2:
        print(
            f"Only {len(num_cols)} numeric feature(s) for {name}; "
            "PCA(2D) requires at least 2. Skipping PCA scatter."
        )
        return

    cols = num_cols + ["CLUSTER_ID"]
    sample = df[cols].copy()

    max_points = 20000
    if len(sample) > max_points:
        sample = sample.sample(max_points, random_state=42)

    X = sample[num_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    pca_df["CLUSTER_ID"] = sample["CLUSTER_ID"].values

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="CLUSTER_ID",
        palette="tab10",
        alpha=0.4,
        s=10,
    )
    plt.title(f"PCA Projection by Cluster ({name})")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()

    out_path = RESULTS_DIR / f"{name}_cluster_pca_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved PCA scatter plot for {name} to: {out_path}")


def main() -> None:
    datasets = available_clustered_datasets()
    if not datasets:
        raise SystemExit(
            "No clustered datasets found. Run run_credit_risk_clustering_all.py first."
        )

    print("=" * 80)
    print("VISUALIZATIONS FOR CREDIT RISK CLUSTERS (ALL DATASETS)")
    print("=" * 80)

    for name, path in datasets.items():
        print("\n" + "-" * 80)
        print(f"Dataset: {name}")
        print(f"Path: {path}")

        df = pd.read_parquet(path)
        print(f"Shape: {df.shape}")

        plot_cluster_sizes(df, name)
        plot_default_rates(df, name)
        plot_pca_scatter(df, name)

    print("\n" + "=" * 80)
    print("ALL CLUSTER VISUALIZATIONS GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()


