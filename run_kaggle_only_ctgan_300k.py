#!/usr/bin/env python3
"""
Generate Synthetic Credit Data Using Kaggle Home Credit Only (CTGAN)

This script:
- Uses Kaggle application_train.csv as the only reference data source
- Does NOT apply RBI/Indian priors (pure Kaggle distribution)
- Trains CTGAN on prepared Kaggle sample and generates 300k synthetic rows
- Saves to data/synthetic_credit_kaggle_only_ctgan_300k.parquet / .csv
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from hybrid_synthetic_generator import HybridSyntheticGenerator  # noqa: E402


def find_kaggle_path() -> Path:
    """Locate application_train.csv from Kaggle."""
    candidates = [
        project_root / "home-credit-default-risk" / "application_train.csv",
        project_root / "data" / "kaggle_home_credit" / "application_train.csv",
        project_root / "application_train.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Kaggle application_train.csv not found. Expected at one of:\n"
        "  - home-credit-default-risk/application_train.csv\n"
        "  - data/kaggle_home_credit/application_train.csv\n"
        "  - application_train.csv"
    )


def main(n_samples: int = 300_000) -> pd.DataFrame:
    print("=" * 80)
    print("KAGGLE-ONLY SYNTHETIC DATA GENERATION (CTGAN, 300K ROWS)")
    print("=" * 80)

    kaggle_path = find_kaggle_path()
    print(f"\nUsing Kaggle dataset: {kaggle_path}")

    priors_path = project_root / "config" / "priors_template.yaml"

    # Initialize generator with Kaggle data
    generator = HybridSyntheticGenerator(
        kaggle_data_path=str(kaggle_path),
        priors_path=str(priors_path),
    )

    # Generate synthetic data using CTGAN and Kaggle structure only
    print("\nGenerating synthetic data from Kaggle structure only (no Indian priors)...")
    synthetic = generator.generate_synthetic(
        n_samples=n_samples,
        method="ctgan",
        use_kaggle_structure=True,
        use_indian_priors=False,
    )

    print(f"\nSynthetic dataset shape: {synthetic.shape}")

    output_path = (
        project_root
        / "data"
        / "synthetic_credit_kaggle_only_ctgan_300k.parquet"
    )
    generator.save_synthetic_data(synthetic, str(output_path))

    print("\n" + "=" * 80)
    print("KAGGLE-ONLY SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"Rows: {len(synthetic):,}")
    print(f"Parquet: {output_path}")
    print(f"CSV: {output_path.with_suffix('.csv')}")

    return synthetic


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during Kaggle-only synthetic data generation: {e}")
        import traceback

        traceback.print_exc()
        raise

#!/usr/bin/env python3
"""
Kaggle-Only Synthetic Data Generation (CTGAN, 300k rows)

This script:
- Uses only the Kaggle Home Credit dataset structure and distributions
- Does NOT apply Indian/RBI priors (use_indian_priors=False)
- Trains a CTGAN model on prepared Kaggle data
- Samples 300,000 synthetic rows
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from hybrid_synthetic_generator import HybridSyntheticGenerator  # noqa: E402


def main(n_samples: int = 300_000) -> pd.DataFrame:
    print("=" * 80)
    print("KAGGLE-ONLY SYNTHETIC DATA GENERATION (CTGAN)")
    print("=" * 80)

    # Locate Kaggle dataset
    possible_paths = [
        project_root / "home-credit-default-risk" / "application_train.csv",
        project_root / "data" / "kaggle_home_credit" / "application_train.csv",
        project_root / "application_train.csv",
    ]

    kaggle_train_path = None
    for path in possible_paths:
        if path.exists():
            kaggle_train_path = path
            break

    if kaggle_train_path is None:
        raise FileNotFoundError(
            "Kaggle application_train.csv not found. "
            "Expected at one of:\n"
            "  - home-credit-default-risk/application_train.csv\n"
            "  - data/kaggle_home_credit/application_train.csv\n"
            "  - application_train.csv"
        )

    print(f"\nUsing Kaggle dataset: {kaggle_train_path}")
    file_size_mb = kaggle_train_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.1f} MB")

    # Initialize generator with Kaggle path; priors still required by class but not applied
    priors_path = project_root / "config" / "priors_template.yaml"
    generator = HybridSyntheticGenerator(
        kaggle_data_path=str(kaggle_train_path),
        priors_path=str(priors_path),
    )

    # Generate Kaggle-only synthetic data (no Indian priors)
    print("\n" + "=" * 80)
    print(f"TRAINING CTGAN ON KAGGLE DATA AND SAMPLING {n_samples:,} ROWS")
    print("=" * 80)

    synthetic = generator.generate_synthetic(
        n_samples=n_samples,
        method="ctgan",
        use_kaggle_structure=True,
        use_indian_priors=False,  # <— Kaggle-only distributions
    )

    print(f"\nGenerated synthetic Kaggle-only dataset shape: {synthetic.shape}")

    # Save
    output_path = (
        project_root / "data" / f"synthetic_credit_kaggle_ctgan_{n_samples//1000}k.parquet"
    )
    generator.save_synthetic_data(synthetic, str(output_path))

    print("\n" + "=" * 80)
    print("KAGGLE-ONLY SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"Rows: {len(synthetic):,}")
    print(f"Parquet: {output_path}")
    print(f"CSV: {output_path.with_suffix('.csv')}")

    return synthetic


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during Kaggle-only CTGAN generation: {e}")
        import traceback

        traceback.print_exc()
        raise

#!/usr/bin/env python3
"""
Generate 300k Synthetic Credit Records Using Kaggle Data Only (CTGAN)

- Uses Home Credit 'application_train.csv' as the reference dataset
- Learns structure via CTGAN (no Indian/RBI priors calibration on core fields)
- Saves synthetic dataset to:
    data/synthetic_credit_kaggle_only_ctgan_300k.parquet / .csv
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from hybrid_synthetic_generator import HybridSyntheticGenerator  # noqa: E402


def find_kaggle_path() -> Path:
    """Locate Kaggle Home Credit application_train.csv."""
    candidates = [
        project_root / "home-credit-default-risk" / "application_train.csv",
        project_root / "data" / "kaggle_home_credit" / "application_train.csv",
        project_root / "application_train.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Kaggle application_train.csv not found. Tried:\n"
        "  - home-credit-default-risk/application_train.csv\n"
        "  - data/kaggle_home_credit/application_train.csv\n"
        "  - application_train.csv"
    )


def main(n_samples: int = 300_000) -> pd.DataFrame:
    print("=" * 80)
    print("KAGGLE-ONLY SYNTHETIC CREDIT DATA GENERATION (CTGAN)")
    print("=" * 80)

    kaggle_path = find_kaggle_path()
    priors_path = project_root / "config" / "priors_template.yaml"

    print(f"\nUsing Kaggle dataset at: {kaggle_path}")

    # Initialize generator with Kaggle data
    generator = HybridSyntheticGenerator(
        kaggle_data_path=str(kaggle_path),
        priors_path=str(priors_path),
    )

    print(f"\nGenerating {n_samples:,} synthetic samples from Kaggle structure (CTGAN, no Indian calibration)...")
    synthetic = generator.generate_synthetic(
        n_samples=n_samples,
        method="ctgan",
        use_kaggle_structure=True,
        use_indian_priors=False,  # do NOT apply Indian priors; keep Kaggle-style distributions
    )

    print(f"\nGenerated Kaggle-only synthetic dataset shape: {synthetic.shape}")

    output_path = (
        project_root
        / "data"
        / f"synthetic_credit_kaggle_only_ctgan_{n_samples//1000}k.parquet"
    )
    generator.save_synthetic_data(synthetic, str(output_path))

    print("\n" + "=" * 80)
    print("KAGGLE-ONLY CTGAN SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"Rows: {len(synthetic):,}")
    print(f"Parquet: {output_path}")
    print(f"CSV: {output_path.with_suffix('.csv')}")

    return synthetic


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during Kaggle-only CTGAN synthetic data generation: {e}")
        import traceback

        traceback.print_exc()
        raise

#!/usr/bin/env python3
"""
Kaggle-Only Synthetic Data Generation with CTGAN (300k rows)

This script:
- Uses only the Kaggle Home Credit dataset structure (no RBI priors adjustments)
- Trains a CTGAN model on a prepared subset of Kaggle data
- Generates 300,000 synthetic rows in the Kaggle-style schema

Outputs:
- data/synthetic_credit_kaggle_only_ctgan_300k.parquet
- data/synthetic_credit_kaggle_only_ctgan_300k.csv
"""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from hybrid_synthetic_generator import HybridSyntheticGenerator  # noqa: E402


def find_kaggle_train_path() -> Path:
    """Locate Kaggle application_train.csv in common locations."""
    candidates = [
        project_root / "home-credit-default-risk" / "application_train.csv",
        project_root / "data" / "kaggle_home_credit" / "application_train.csv",
        project_root / "application_train.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Kaggle application_train.csv not found. Expected in one of:\n"
        "  - home-credit-default-risk/application_train.csv\n"
        "  - data/kaggle_home_credit/application_train.csv\n"
        "  - application_train.csv (project root)"
    )


def main(n_samples: int = 300_000) -> pd.DataFrame:
    print("=" * 80)
    print("KAGGLE-ONLY SYNTHETIC DATA GENERATION (CTGAN, 300k ROWS)")
    print("=" * 80)

    kaggle_path = find_kaggle_train_path()
    print(f"\nUsing Kaggle dataset: {kaggle_path}")

    # Quick sanity check on Kaggle file
    sample_df = pd.read_csv(kaggle_path, nrows=5)
    print(f"Columns: {len(sample_df.columns)}")
    print(f"Sample columns: {list(sample_df.columns[:10])}")

    priors_path = project_root / "config" / "priors_template.yaml"

    # Initialize generator with Kaggle data; we'll disable Indian priors downstream
    generator = HybridSyntheticGenerator(
        kaggle_data_path=str(kaggle_path),
        priors_path=str(priors_path),
    )

    print("\nPreparing Kaggle reference data (no Indian priors applied)...")
    reference_data = generator.prepare_kaggle_data(max_rows=10000)
    print(f"Prepared reference data shape: {reference_data.shape}")

    # Build metadata and CTGAN synthesizer manually so we can keep priors disabled
    from sdv.metadata import SingleTableMetadata  # noqa: E402
    from sdv.single_table import CTGANSynthesizer  # noqa: E402

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(reference_data)

    synthesizer = CTGANSynthesizer(metadata)

    print("\nFitting CTGAN on Kaggle-only reference data (this may take time)...")
    synthesizer.fit(reference_data)
    print("CTGAN training complete.")

    print(f"\nSampling {n_samples:,} synthetic rows from Kaggle-only CTGAN model...")
    synthetic = synthesizer.sample(num_rows=n_samples)
    print(f"Synthetic dataset shape: {synthetic.shape}")

    # Save outputs
    output_path = (
        project_root / "data" / "synthetic_credit_kaggle_only_ctgan_300k.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_parquet(output_path, index=False)

    csv_path = output_path.with_suffix(".csv")
    synthetic.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("KAGGLE-ONLY SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 80)
    print(f"Rows: {len(synthetic):,}")
    print(f"Parquet: {output_path}")
    print(f"CSV: {csv_path}")

    return synthetic


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during Kaggle-only CTGAN generation: {e}")
        import traceback

        traceback.print_exc()
        raise


