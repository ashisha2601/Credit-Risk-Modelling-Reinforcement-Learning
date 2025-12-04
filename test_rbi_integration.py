#!/usr/bin/env python3
"""
Quick test script to verify RBI compliance flags integration
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from hybrid_synthetic_generator import HybridSyntheticGenerator
import pandas as pd

def test_rbi_integration():
    """Test that RBI flags are properly integrated"""
    print("="*70)
    print("TESTING RBI COMPLIANCE FLAGS INTEGRATION")
    print("="*70)
    
    # Initialize generator
    priors_path = project_root / "config" / "priors_template.yaml"
    generator = HybridSyntheticGenerator(
        kaggle_data_path=None,  # Generate from priors only
        priors_path=str(priors_path)
    )
    
    # Generate small sample
    print("\n🔧 Generating 1000 synthetic samples...")
    synthetic_data = generator.generate_synthetic(
        n_samples=1000,
        method='gaussian_copula',
        use_kaggle_structure=False,
        use_indian_priors=True
    )
    
    print(f"\n✅ Generated {len(synthetic_data)} samples")
    print(f"\nColumns: {list(synthetic_data.columns)}")
    
    # Verify RBI flags
    print("\n" + "="*70)
    print("VERIFYING RBI COMPLIANCE FLAGS")
    print("="*70)
    
    # Check NSFR_RSF_FACTOR
    if 'NSFR_RSF_FACTOR' in synthetic_data.columns:
        nsfr_values = synthetic_data['NSFR_RSF_FACTOR'].unique()
        print(f"✓ NSFR_RSF_FACTOR: {sorted(nsfr_values)}")
    else:
        print("✗ NSFR_RSF_FACTOR: MISSING")
    
    # Check INOPERATIVE_FLAG
    if 'INOPERATIVE_FLAG' in synthetic_data.columns:
        inop_rate = synthetic_data['INOPERATIVE_FLAG'].mean()
        print(f"✓ INOPERATIVE_FLAG: {inop_rate:.2%} (expected ~3%)")
    else:
        print("✗ INOPERATIVE_FLAG: MISSING")
    
    # Check FX_HEDGING_FLAG
    if 'FX_HEDGING_FLAG' in synthetic_data.columns:
        fx_rate = synthetic_data['FX_HEDGING_FLAG'].mean()
        print(f"✓ FX_HEDGING_FLAG: {fx_rate:.2%} (expected ~30%)")
    else:
        print("✗ FX_HEDGING_FLAG: MISSING")
    
    # Check CP_NCD_FLAG
    if 'CP_NCD_FLAG' in synthetic_data.columns:
        cp_rate = synthetic_data['CP_NCD_FLAG'].mean()
        print(f"✓ CP_NCD_FLAG: {cp_rate:.2%} (expected ~10%)")
    else:
        print("✗ CP_NCD_FLAG: MISSING")
    
    # Check Census data
    print("\n" + "="*70)
    print("VERIFYING CENSUS DATA")
    print("="*70)
    
    if 'STATE' in synthetic_data.columns:
        state_counts = synthetic_data['STATE'].value_counts()
        print(f"✓ STATE: {len(state_counts)} unique states")
        print(f"  Top 5 states: {', '.join(state_counts.head(5).index.tolist())}")
    else:
        print("✗ STATE: MISSING")
    
    if 'WORKER_TYPE' in synthetic_data.columns:
        worker_counts = synthetic_data['WORKER_TYPE'].value_counts()
        print(f"✓ WORKER_TYPE: {len(worker_counts)} categories")
        print(f"  Distribution: {dict(worker_counts)}")
    else:
        print("✗ WORKER_TYPE: MISSING")
    
    if 'CURRENCY' in synthetic_data.columns:
        currency = synthetic_data['CURRENCY'].unique()
        print(f"✓ CURRENCY: {currency[0] if len(currency) > 0 else 'MISSING'}")
    else:
        print("✗ CURRENCY: MISSING")
    
    # Display sample data
    print("\n" + "="*70)
    print("SAMPLE DATA (First 5 rows)")
    print("="*70)
    display_cols = ['AGE', 'MONTHLY_INCOME', 'CREDIT_SCORE', 'STATE', 
                    'NSFR_RSF_FACTOR', 'INOPERATIVE_FLAG', 'FX_HEDGING_FLAG', 'CP_NCD_FLAG']
    available_cols = [c for c in display_cols if c in synthetic_data.columns]
    print(synthetic_data[available_cols].head())
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)
    
    return synthetic_data

if __name__ == "__main__":
    test_rbi_integration()

