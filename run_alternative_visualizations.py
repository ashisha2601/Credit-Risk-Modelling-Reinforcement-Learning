#!/usr/bin/env python3
"""
Alternative Normal Clustering Visualizations
--------------------------------------------
Generates two types of "normal" (non-dimensionality-reduced) visualizations:
1. Radar Chart (Spider Plot): To visualize the "profile" or "centroid" of each cluster.
2. Parallel Coordinates Plot: To visualize the flow of features across clusters.

Inputs:
- data/synthetic_credit_with_risk_clusters.parquet

Outputs:
- results/cluster_radar_chart.png
- results/cluster_parallel_coordinates.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_credit_with_risk_clusters.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def plot_radar_chart(df):
    """Generates a Radar Chart of Cluster Centroids"""
    print("Generating Radar Chart...")
    
    # Select numeric features for profiling
    features = ['MONTHLY_INCOME', 'LOAN_AMOUNT', 'CREDIT_SCORE', 'LOAN_TENURE_MONTHS', 'INTEREST_RATE']
    
    # Check if features exist
    features = [f for f in features if f in df.columns]
    
    # Normalize features to 0-1 range so they can be plotted on the same scale
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    
    # Calculate centroids (mean) for each cluster
    centroids = df_scaled.groupby('CLUSTER_ID')[features].mean().reset_index()
    
    # Prepare plot
    categories = features
    N = len(categories)
    
    # Angles for the axes
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot each cluster
    colors = plt.cm.get_cmap("tab10", len(centroids))
    
    for i, row in centroids.iterrows():
        values = row[features].values.flatten().tolist()
        values += values[:1] # Close the loop
        cluster_id = int(row['CLUSTER_ID'])
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Cluster {cluster_id}", color=colors(i))
        ax.fill(angles, values, alpha=0.1, color=colors(i))
        
    plt.title("Cluster Profiles (Normalized Centroids)", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    out_path = RESULTS_DIR / "cluster_radar_chart.png"
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved Radar Chart to {out_path}")

def plot_parallel_coordinates(df):
    """Generates a Parallel Coordinates Plot"""
    print("Generating Parallel Coordinates Plot...")
    
    features = ['MONTHLY_INCOME', 'LOAN_AMOUNT', 'CREDIT_SCORE']
    features = [f for f in features if f in df.columns]
    
    # Sample data for cleaner plot
    plot_data = df.sample(500, random_state=42).copy()
    plot_data = plot_data.sort_values('CLUSTER_ID')
    
    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(
        plot_data[features + ['CLUSTER_ID']], 
        'CLUSTER_ID', 
        colormap='viridis',
        alpha=0.6
    )
    
    plt.title("Parallel Coordinates Plot (Sample of 500 Borrowers)", size=15)
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    
    out_path = RESULTS_DIR / "cluster_parallel_coordinates.png"
    plt.savefig(out_path, dpi=150)
    print(f"✅ Saved Parallel Coordinates Plot to {out_path}")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
        
    df = pd.read_parquet(DATA_PATH)
    plot_radar_chart(df)
    plot_parallel_coordinates(df)

if __name__ == "__main__":
    main()
