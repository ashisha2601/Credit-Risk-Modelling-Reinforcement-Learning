#!/usr/bin/env python3
"""
Normal Clustering Visualization: Pairwise Scatter Matrix
--------------------------------------------------------
This script generates a Pair Plot (Scatter Matrix) of the top numeric features,
colored by Cluster ID. This provides a "normal" view of the clusters in the
original feature space, distinct from the dimensionality-reduced PCA view.

Inputs:
- data/synthetic_credit_with_risk_clusters.parquet

Outputs:
- results/cluster_pairplot.png
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_credit_with_risk_clusters.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    print("="*80)
    print("NORMAL CLUSTERING VISUALIZATION: PAIR PLOT")
    print("="*80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run Stage 2 first.")

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Select key numeric features for visualization
    # We choose top 4 to keep the plot readable (4x4 matrix)
    features = ['MONTHLY_INCOME', 'LOAN_AMOUNT', 'CREDIT_SCORE', 'LOAN_TENURE_MONTHS']
    
    # Ensure features exist
    plot_features = [f for f in features if f in df.columns]
    
    if len(plot_features) < 2:
        print("Not enough numeric features for pairplot.")
        return

    # Add Cluster ID for coloring
    if 'CLUSTER_ID' not in df.columns:
        print("Cluster ID not found.")
        return
        
    plot_data = df[plot_features + ['CLUSTER_ID']].copy()
    
    # Sample if data is too large (Pairplot is slow)
    if len(plot_data) > 5000:
        print(f"Sampling 5000 rows from {len(plot_data)} for visualization...")
        plot_data = plot_data.sample(5000, random_state=42)
    
    # Create Pair Plot
    print("Generating Pair Plot (this may take a moment)...")
    plt.figure(figsize=(15, 15))
    sns.pairplot(
        plot_data, 
        hue='CLUSTER_ID', 
        palette='viridis', 
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 15}
    )
    
    output_path = RESULTS_DIR / "cluster_pairplot.png"
    plt.savefig(output_path)
    print(f"✅ Saved Pair Plot to {output_path}")

if __name__ == "__main__":
    main()
