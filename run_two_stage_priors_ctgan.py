#!/usr/bin/env python3
"""
Two-Stage Priors-Only Synthetic Data Generation with CTGAN

Stage 1: Generate a smaller micro-dataset (e.g. 20k rows) using RBI priors only
Stage 2: Fit CTGAN on this micro-dataset and sample a larger dataset (e.g. 3 lakh rows)

This script is complementary to run_hybrid_ctgan_pipeline.py:
- That script uses Kaggle structure + RBI priors directly with CTGAN
- This script uses RBI priors only, then learns correlations via CTGAN in a second stage
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from hybrid_synthetic_generator import HybridSyntheticGenerator  # noqa: E402
from sdv.metadata import SingleTableMetadata  # noqa: E402
from sdv.single_table import CTGANSynthesizer  # noqa: E402


def main(
    micro_n: int = 20_000,
    large_n: int = 300_000,
) -> pd.DataFrame:
    print("=" * 80)
    print("TWO-STAGE PRIORS-ONLY SYNTHETIC DATA GENERATION (CTGAN)")
    print("=" * 80)

    priors_path = project_root / "config" / "priors_template.yaml"

    # Stage 1: Generate micro-dataset from RBI priors only
    print("\n" + "=" * 80)
    print(f"STAGE 1: GENERATE MICRO-DATASET FROM PRIORS ONLY (n={micro_n:,})")
    print("=" * 80)

    generator = HybridSyntheticGenerator(
        kaggle_data_path=None,  # Force priors-only mode
        priors_path=str(priors_path),
    )

    micro_data = generator.generate_synthetic(
        n_samples=micro_n,
        method="gaussian_copula",  # Ignored in priors-only path
        use_kaggle_structure=False,
        use_indian_priors=True,
    )

    print(f"\n Micro-dataset generated with shape: {micro_data.shape}")

    # Stage 2: Fit CTGAN on micro-dataset
    print("\n" + "=" * 80)
    print("STAGE 2: FIT CTGAN ON MICRO-DATASET")
    print("=" * 80)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(micro_data)

    synthesizer = CTGANSynthesizer(metadata)

    print(" Fitting CTGAN synthesizer (this may take some time)...")
    synthesizer.fit(micro_data)
    print(" CTGAN training complete.")

    # Sample large dataset
    print("\n" + "=" * 80)
    print(f"STAGE 3: SAMPLE LARGE DATASET FROM CTGAN (n={large_n:,})")
    print("=" * 80)

    synthetic_large = synthesizer.sample(num_rows=large_n)

    # Add RBI compliance flags and Indian market features using helper methods
    synthetic_large = generator._add_rbi_compliance_flags(synthetic_large)
    synthetic_large = generator._add_indian_market_features(synthetic_large)

    print(f"\n Large synthetic dataset generated with shape: {synthetic_large.shape}")

    # Save results
    print("\n" + "=" * 80)
    print("STAGE 4: SAVE SYNTHETIC DATA")
    print("=" * 80)

    output_path = (
        project_root
        / "data"
        / "synthetic_credit_two_stage_priors_ctgan_300k.parquet"
    )
    generator.save_synthetic_data(synthetic_large, str(output_path))

    print("\n" + "=" * 80)
    print("SUMMARY (TWO-STAGE PRIORS-ONLY CTGAN)")
    print("=" * 80)
    print(f"  Micro-dataset size: {len(micro_data):,} rows")
    print(f"  Final synthetic size: {len(synthetic_large):,} rows")
    print(f"  Output file (Parquet): {output_path}")
    print(f"  Output file (CSV): {output_path.with_suffix('.csv')}")
    print("=" * 80)

    return synthetic_large


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Error during two-stage synthetic data generation: {e}")
        import traceback

        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure SDV and CTGAN are installed (see requirements.txt)")
        print("2. Check priors file exists and is valid YAML: config/priors_template.yaml")
        print("3. Try reducing micro_n or large_n if memory is limited")
        raise