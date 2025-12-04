#!/usr/bin/env python3
"""
Hybrid Synthetic Data Generation Pipeline with Conditional GANs
Following System Design: External Data Source → Data Preprocessing → Conditional GANs Created

This script implements the hybrid approach:
1. Load Kaggle dataset (External Data Source)
2. Preprocess and calibrate with RBI priors (Data Preprocessing)
3. Train Conditional GANs (CTGAN) on blended data
4. Generate high-fidelity synthetic credit data
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))

from hybrid_synthetic_generator import HybridSyntheticGenerator
import pandas as pd
import numpy as np

def main():
    print("="*80)
    print("HYBRID SYNTHETIC DATA GENERATION PIPELINE")
    print("Following System Design: Conditional GANs Approach")
    print("="*80)
    
    # Step 1: External Data Source - Load Kaggle Dataset
    print("\n" + "="*80)
    print("STEP 1: EXTERNAL DATA SOURCE")
    print("="*80)
    
    # Check multiple possible locations for Kaggle dataset
    possible_paths = [
        project_root / "home-credit-default-risk" / "application_train.csv",
        project_root / "data" / "kaggle_home_credit" / "application_train.csv",
        project_root / "application_train.csv"
    ]
    
    kaggle_train_path = None
    for path in possible_paths:
        if path.exists():
            kaggle_train_path = path
            break
    
    kaggle_test_path = project_root / "home-credit-default-risk" / "application_test.csv"
    priors_path = project_root / "config" / "priors_template.yaml"
    
    if kaggle_train_path.exists():
        print(f"✅ Found Kaggle dataset: {kaggle_train_path}")
        file_size_mb = kaggle_train_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {file_size_mb:.1f} MB")
        
        # Quick check of dataset structure
        sample_df = pd.read_csv(kaggle_train_path, nrows=5)
        print(f"   Columns: {len(sample_df.columns)}")
        print(f"   Sample columns: {list(sample_df.columns[:10])}")
    else:
        print(f"⚠️  Kaggle dataset not found at: {kaggle_train_path}")
        print("   Will generate from RBI priors only")
        kaggle_train_path = None
    
    # Step 2: Initialize Hybrid Generator
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING & RBI PRIORS CALIBRATION")
    print("="*80)
    
    generator = HybridSyntheticGenerator(
        kaggle_data_path=str(kaggle_train_path) if kaggle_train_path and kaggle_train_path.exists() else None,
        priors_path=str(priors_path)
    )
    
    # Step 3: Generate Synthetic Data using Conditional GANs
    print("\n" + "="*80)
    print("STEP 3: CONDITIONAL GANs CREATED (CTGAN)")
    print("="*80)
    print("Training CTGAN to generate high-fidelity synthetic data...")
    print("⏳ Generating a larger dataset (~200,000 rows). This may take longer (up to 1–2 hours) depending on your machine.")
    print("   (CTGAN needs to train on the data structure and then sample many rows)")
    
    # Generate a larger synthetic dataset (~2 lakh rows)
    synthetic_data = generator.generate_synthetic(
        n_samples=200000,
        method='ctgan',  # Using Conditional GANs as per system design
        use_kaggle_structure=kaggle_train_path is not None and kaggle_train_path.exists(),
        use_indian_priors=True
    )
    
    print(f"\n✅ Generated {len(synthetic_data)} synthetic samples using CTGAN")
    
    # Step 4: Save Results
    print("\n" + "="*80)
    print("STEP 4: SAVE SYNTHETIC DATA")
    print("="*80)
    
    # Save as a new version so previous 10k-sample files remain available
    output_path = project_root / "data" / "synthetic_credit_data_v0.4_hybrid_ctgan_200k.parquet"
    generator.save_synthetic_data(synthetic_data, str(output_path))
    
    # Step 5: Summary Statistics
    print("\n" + "="*80)
    print("SYNTHETIC DATA SUMMARY")
    print("="*80)
    
    # Handle both TARGET (Kaggle) and DEFAULT_FLAG (priors-only) column names
    default_col = 'TARGET' if 'TARGET' in synthetic_data.columns else 'DEFAULT_FLAG'
    
    print(f"\n📊 Basic Statistics:")
    print(f"  Total Records: {len(synthetic_data):,}")
    print(f"  Total Features: {len(synthetic_data.columns)}")
    print(f"  Default Rate: {synthetic_data[default_col].mean()*100:.2f}%")
    
    # Income and loan amount (handle different column names)
    income_col = 'MONTHLY_INCOME' if 'MONTHLY_INCOME' in synthetic_data.columns else 'AMT_INCOME_TOTAL'
    loan_col = 'LOAN_AMOUNT' if 'LOAN_AMOUNT' in synthetic_data.columns else 'AMT_CREDIT'
    credit_col = 'CREDIT_SCORE' if 'CREDIT_SCORE' in synthetic_data.columns else None
    
    if income_col in synthetic_data.columns:
        print(f"  Average Income: ₹{synthetic_data[income_col].mean():,.0f}")
    if loan_col in synthetic_data.columns:
        print(f"  Average Loan Amount: ₹{synthetic_data[loan_col].mean():,.0f}")
    if credit_col and credit_col in synthetic_data.columns:
        print(f"  Average Credit Score: {synthetic_data[credit_col].mean():.0f}")
    
    print(f"\n📋 Loan Type Distribution:")
    if 'LOAN_TYPE' in synthetic_data.columns:
        loan_counts = synthetic_data['LOAN_TYPE'].value_counts()
        for loan_type, count in loan_counts.items():
            pct = (count/len(synthetic_data))*100
            default_rate = synthetic_data[synthetic_data['LOAN_TYPE']==loan_type][default_col].mean()*100
            print(f"  {loan_type}: {count:,} ({pct:.1f}%) - Default: {default_rate:.2f}%")
    
    print(f"\n🏦 Bank Group Distribution:")
    if 'BANK_GROUP' in synthetic_data.columns:
        bank_counts = synthetic_data['BANK_GROUP'].value_counts()
        for bank, count in bank_counts.items():
            pct = (count/len(synthetic_data))*100
            print(f"  {bank}: {count:,} ({pct:.1f}%)")
    
    print(f"\n✅ RBI Compliance Flags:")
    if 'NSFR_RSF_FACTOR' in synthetic_data.columns:
        print(f"  NSFR RSF Factor: {synthetic_data['NSFR_RSF_FACTOR'].value_counts().to_dict()}")
    if 'INOPERATIVE_FLAG' in synthetic_data.columns:
        print(f"  Inoperative Accounts: {synthetic_data['INOPERATIVE_FLAG'].mean()*100:.2f}%")
    if 'FX_HEDGING_FLAG' in synthetic_data.columns:
        print(f"  FX Hedging: {synthetic_data['FX_HEDGING_FLAG'].mean()*100:.2f}%")
    if 'CP_NCD_FLAG' in synthetic_data.columns:
        print(f"  CP/NCD: {synthetic_data['CP_NCD_FLAG'].mean()*100:.2f}%")
    
    print(f"\n📊 Sample Columns (first 20):")
    print(f"  {list(synthetic_data.columns[:20])}")
    
    print(f"\n💾 Files Saved:")
    print(f"  - {output_path}")
    print(f"  - {output_path.with_suffix('.csv')}")
    
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. Proceed to Stage 2: Unsupervised Learning (Clustering)")
    print("2. Proceed to Stage 3: Reinforcement Learning Policy Training")
    print("="*80)
    
    return synthetic_data

if __name__ == "__main__":
    try:
        synthetic_data = main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure CTGAN is installed: pip install ctgan")
        print("2. Check Kaggle dataset path is correct")
        print("3. Verify RBI priors file exists: config/priors_template.yaml")
        raise

