#!/usr/bin/env python3
"""
Advanced visualizations for credit risk clusters across all synthetic datasets.

For each clustered dataset:
  - kaggle_only_with_risk_clusters.parquet
  - rbi_priors_only_with_risk_clusters.parquet
  - integrated_hybrid_with_risk_clusters.parquet

This script generates, per dataset:
  1. Cluster feature profiles:
     - Boxplots of income, loan amount, credit score (if present), and EMI by CLUSTER_ID
  2. Income vs EMI scatter plot coloured by RISK_SEGMENT (or default flag if not present)
  3. Product / bank-group heatmaps:
     - CLUSTER_ID × LOAN_TYPE (or NAME_CONTRACT_TYPE for Kaggle)
     - CLUSTER_ID × BANK_GROUP (if available)

And globally:
  4. Cross-dataset PD calibration plot:
     - Bars of default rate by RISK_SEGMENT and dataset (Kaggle / RBI priors / Hybrid)

All figures are saved under results/ with descriptive filenames.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_clustered_datasets() -> Dict[str, Path]:
    """Return mapping from dataset name to clustered parquet path."""
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


def detect_default_col(df: pd.DataFrame) -> str:
    if "DEFAULT_FLAG" in df.columns:
        return "DEFAULT_FLAG"
    if "TARGET" in df.columns:
        return "TARGET"
    raise ValueError("No default flag column found (expected DEFAULT_FLAG or TARGET).")


def detect_income_loan_emi_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Detect income, loan, credit_score, and EMI columns for a dataset."""
    income_col = None
    loan_col = None
    emi_col = None
    score_col = None

    if "MONTHLY_INCOME" in df.columns:
        income_col = "MONTHLY_INCOME"
    elif "AMT_INCOME_TOTAL" in df.columns:
        income_col = "AMT_INCOME_TOTAL"

    if "LOAN_AMOUNT" in df.columns:
        loan_col = "LOAN_AMOUNT"
    elif "AMT_CREDIT" in df.columns:
        loan_col = "AMT_CREDIT"

    if "MONTHLY_PAYMENT" in df.columns:
        emi_col = "MONTHLY_PAYMENT"
    elif "AMT_ANNUITY" in df.columns:
        emi_col = "AMT_ANNUITY"

    if "CREDIT_SCORE" in df.columns:
        score_col = "CREDIT_SCORE"

    return income_col, loan_col, emi_col, score_col


def plot_feature_boxplot(df: pd.DataFrame, dataset: str, feature: str, pretty_name: str) -> None:
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x="CLUSTER_ID", y=feature)
    plt.xlabel("Cluster ID")
    plt.ylabel(pretty_name)
    plt.title(f"{pretty_name} by Cluster ({dataset})")
    plt.tight_layout()

    out_path = RESULTS_DIR / f"{dataset}_box_{feature}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved boxplot for {feature} ({dataset}) to: {out_path}")


def plot_income_emi_scatter(
    df: pd.DataFrame,
    dataset: str,
    income_col: Optional[str],
    emi_col: Optional[str],
) -> None:
    if not income_col or not emi_col:
        print(f"Skipping income vs EMI scatter for {dataset} (missing columns).")
        return

    hue_col = "RISK_SEGMENT" if "RISK_SEGMENT" in df.columns else detect_default_col(df)

    sample = df[[income_col, emi_col, hue_col]].copy()
    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=42)

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=sample,
        x=income_col,
        y=emi_col,
        hue=hue_col,
        alpha=0.3,
        s=10,
    )
    plt.xlabel(income_col)
    plt.ylabel(emi_col)
    plt.title(f"Income vs EMI by {hue_col} ({dataset})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.tight_layout()

    out_path = RESULTS_DIR / f"{dataset}_scatter_income_vs_emi.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved income vs EMI scatter for {dataset} to: {out_path}")


def plot_heatmap_cluster_vs_category(
    df: pd.DataFrame,
    dataset: str,
    category_col: str,
    label: str,
) -> None:
    default_col = detect_default_col(df)
    if category_col not in df.columns:
        print(f"Skipping heatmap {label} for {dataset} (column {category_col} missing).")
        return

    pivot = (
        df.groupby(["CLUSTER_ID", category_col])[default_col]
        .mean()
        .reset_index()
        .pivot(index="CLUSTER_ID", columns=category_col, values=default_col)
    )

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="Reds",
    )
    plt.title(f"Default Rate Heatmap: Cluster vs {label} ({dataset})")
    plt.xlabel(label)
    plt.ylabel("Cluster ID")
    plt.tight_layout()

    safe_label = label.lower().replace(" ", "_")
    out_path = RESULTS_DIR / f"{dataset}_heatmap_cluster_vs_{safe_label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved heatmap (Cluster vs {label}) for {dataset} to: {out_path}")


def collect_pd_by_risk_segment(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    default_col = detect_default_col(df)
    if "RISK_SEGMENT" not in df.columns:
        return pd.DataFrame()
    stats = (
        df.groupby("RISK_SEGMENT")[default_col]
        .mean()
        .reset_index()
        .rename(columns={default_col: "default_rate"})
    )
    stats["dataset"] = dataset
    return stats


def plot_cross_dataset_pd(pd_df: pd.DataFrame) -> None:
    if pd_df.empty:
        print("No PD-by-risk-segment data available for cross-dataset plot.")
        return

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=pd_df,
        x="RISK_SEGMENT",
        y="default_rate",
        hue="dataset",
    )
    plt.xlabel("Risk Segment")
    plt.ylabel("Default rate")
    plt.title("Default Rate by Risk Segment Across Datasets")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f"{x*100:.1f}%")
    )
    # Add percentage labels on top of grouped bars
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

    out_path = RESULTS_DIR / "cross_dataset_pd_by_risk_segment.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved cross-dataset PD calibration plot to: {out_path}")


def main() -> None:
    datasets = get_clustered_datasets()
    if not datasets:
        raise SystemExit(
            "No clustered datasets found. Run run_credit_risk_clustering_all.py first."
        )

    print("=" * 80)
    print("ADVANCED VISUALIZATIONS FOR CREDIT RISK CLUSTERS (ALL DATASETS)")
    print("=" * 80)

    all_pd_stats = []

    for name, path in datasets.items():
        print("\n" + "-" * 80)
        print(f"Dataset: {name}")
        print(f"Path: {path}")
        df = pd.read_parquet(path)
        print(f"Shape: {df.shape}")

        income_col, loan_col, emi_col, score_col = detect_income_loan_emi_cols(df)

        # 1) Cluster feature profiles (boxplots)
        if income_col:
            plot_feature_boxplot(df, name, income_col, "Monthly Income")
        if loan_col:
            plot_feature_boxplot(df, name, loan_col, "Loan Amount")
        if score_col:
            plot_feature_boxplot(df, name, score_col, "Credit Score")
        if emi_col:
            plot_feature_boxplot(df, name, emi_col, "Monthly Payment / Annuity")

        # 2) Income vs EMI scatter
        plot_income_emi_scatter(df, name, income_col, emi_col)

        # 3) Product / bank-group heatmaps
        if name == "kaggle_only":
            # Use NAME_CONTRACT_TYPE as product
            if "NAME_CONTRACT_TYPE" in df.columns:
                plot_heatmap_cluster_vs_category(
                    df,
                    name,
                    "NAME_CONTRACT_TYPE",
                    "Contract Type",
                )
        else:
            if "LOAN_TYPE" in df.columns:
                plot_heatmap_cluster_vs_category(
                    df,
                    name,
                    "LOAN_TYPE",
                    "Loan Type",
                )
            if "BANK_GROUP" in df.columns:
                plot_heatmap_cluster_vs_category(
                    df,
                    name,
                    "BANK_GROUP",
                    "Bank Group",
                )

        # Collect PD by risk segment for cross-dataset plot
        pd_stats = collect_pd_by_risk_segment(df, name)
        if not pd_stats.empty:
            all_pd_stats.append(pd_stats)

    if all_pd_stats:
        combined_pd = pd.concat(all_pd_stats, ignore_index=True)
        plot_cross_dataset_pd(combined_pd)

    print("\n" + "=" * 80)
    print("ADVANCED VISUALIZATIONS GENERATED FOR ALL DATASETS")
    print("=" * 80)


if __name__ == "__main__":
    main()


