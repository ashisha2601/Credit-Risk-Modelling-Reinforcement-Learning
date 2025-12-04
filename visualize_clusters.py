#!/usr/bin/env python3
"""
Visualizations for Synthetic Credit Risk Clusters

Generates:
1. Bar plot of cluster sizes
2. Bar plot of default rates by cluster
3. 2D PCA scatter plot of borrowers coloured by cluster

Outputs are saved under results/ as PNG files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).parent


def load_clustered_data() -> pd.DataFrame:
    """Load the clustered synthetic dataset."""
    candidates = [
        PROJECT_ROOT / "data" / "synthetic_credit_with_clusters.parquet",
        PROJECT_ROOT / "data" / "synthetic_credit_with_risk_clusters.parquet",
    ]

    for path in candidates:
        if path.exists():
            print(f"Using clustered dataset: {path}")
            return pd.read_parquet(path)

    raise FileNotFoundError(
        "No clustered dataset found. Expected one of:\n"
        "  - data/synthetic_credit_with_clusters.parquet\n"
        "  - data/synthetic_credit_with_risk_clusters.parquet"
    )


def ensure_results_dir() -> Path:
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def plot_cluster_sizes(df: pd.DataFrame, results_dir: Path) -> None:
    counts = df["CLUSTER_ID"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of borrowers")
    plt.title("Cluster Sizes (Synthetic Credit Data)")
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()

    out_path = results_dir / "cluster_sizes.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved cluster size plot to: {out_path}")


def plot_default_rates(df: pd.DataFrame, results_dir: Path) -> None:
    default_col = "DEFAULT_FLAG" if "DEFAULT_FLAG" in df.columns else "TARGET"
    if default_col not in df.columns:
        print("No default flag column found (DEFAULT_FLAG/TARGET). Skipping default-rate plot.")
        return

    stats = (
        df.groupby("CLUSTER_ID")[default_col]
        .mean()
        .reset_index()
        .rename(columns={default_col: "default_rate"})
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=stats, x="CLUSTER_ID", y="default_rate", palette="rocket")
    plt.xlabel("Cluster ID")
    plt.ylabel("Default rate")
    plt.title("Default Rate by Cluster")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x*100:.1f}%")
    # Add percentage labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2,
            height,
            f"{height*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()

    out_path = results_dir / "cluster_default_rates.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved default-rate plot to: {out_path}")


def plot_pca_scatter(df: pd.DataFrame, results_dir: Path) -> None:
    numeric_cols = [
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
    if not numeric_cols:
        print("No numeric columns available for PCA scatter. Skipping.")
        return

    # Sample for visualization if dataset is huge
    max_points = 20000
    sample_df = df[numeric_cols + ["CLUSTER_ID"]].copy()
    if len(sample_df) > max_points:
        sample_df = sample_df.sample(max_points, random_state=42)

    X = sample_df[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"],
    )
    pca_df["CLUSTER_ID"] = sample_df["CLUSTER_ID"].values

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="CLUSTER_ID",
        palette="tab10",
        alpha=0.4,
        s=10,
    )
    plt.title("PCA Projection of Synthetic Borrowers by Cluster")
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()

    out_path = results_dir / "cluster_pca_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved PCA scatter plot to: {out_path}")


def main() -> None:
    print("=" * 80)
    print("CLUSTER VISUALIZATIONS (SYNTHETIC CREDIT DATA)")
    print("=" * 80)

    df = load_clustered_data()
    print(f"Loaded clustered dataset with shape: {df.shape}")

    results_dir = ensure_results_dir()

    plot_cluster_sizes(df, results_dir)
    plot_default_rates(df, results_dir)
    plot_pca_scatter(df, results_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATIONS GENERATED")
    print("=" * 80)


if __name__ == "__main__":
    main()


