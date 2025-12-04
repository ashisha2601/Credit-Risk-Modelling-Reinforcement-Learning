#!/usr/bin/env python3
"""
Generate Synthetic Credit Data Using RBI Priors Only (No Kaggle, No CTGAN)

This script:
- Uses HybridSyntheticGenerator._generate_from_priors to sample directly from RBI priors
- Does NOT use Kaggle structure or any generative model (CTGAN/GaussianCopula)
- Saves the resulting dataset to data/synthetic_credit_priors_only_100k.parquet / .csv
"""

from pathlib import Path
import sys

import pandas as pd

project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from hybrid_synthetic_generator import HybridSyntheticGenerator  # noqa: E402


def main(n_samples: int = 100_000) -> pd.DataFrame:
    print("=" * 80)
    print("RBI PRIORS-ONLY SYNTHETIC CREDIT DATA GENERATION")
    print("=" * 80)

    priors_path = project_root / "config" / "priors_template.yaml"

    # Initialize generator with no Kaggle data
    generator = HybridSyntheticGenerator(
        kaggle_data_path=None,
        priors_path=str(priors_path),
    )

    print(f"\nGenerating {n_samples:,} samples directly from RBI priors...")
    synthetic = generator._generate_from_priors(n_samples=n_samples)
    print(f"Generated dataset shape: {synthetic.shape}")

    output_path = (
        project_root / "data" / f"synthetic_credit_priors_only_{n_samples//1000}k.parquet"
    )
    generator.save_synthetic_data(synthetic, str(output_path))

    print("\n" + "=" * 80)
    print("PRIORS-ONLY SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"Rows: {len(synthetic):,}")
    print(f"Parquet: {output_path}")
    print(f"CSV: {output_path.with_suffix('.csv')}")

    return synthetic


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during priors-only synthetic data generation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


