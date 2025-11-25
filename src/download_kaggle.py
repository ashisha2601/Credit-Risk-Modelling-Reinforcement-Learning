"""
Kaggle Dataset Download Helper
Downloads Home Credit Default Risk dataset for hybrid approach
"""

import os
import subprocess
import zipfile
from pathlib import Path

def check_kaggle_installed():
    """Check if Kaggle CLI is installed"""
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_kaggle_credentials():
    """Check if Kaggle credentials are set up"""
    kaggle_dir = Path.home() / '.kaggle'
    credentials_file = kaggle_dir / 'kaggle.json'
    return credentials_file.exists()

def setup_instructions():
    """Print setup instructions"""
    print("\n" + "="*70)
    print("KAGGLE SETUP INSTRUCTIONS")
    print("="*70)
    print("\n1. Install Kaggle CLI:")
    print("   pip install kaggle")
    print("\n2. Get your API credentials:")
    print("   - Go to: https://www.kaggle.com/account")
    print("   - Scroll to 'API' section")
    print("   - Click 'Create New Token'")
    print("   - This downloads 'kaggle.json' file")
    print("\n3. Set up credentials:")
    print("   mkdir -p ~/.kaggle")
    print("   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
    print("   chmod 600 ~/.kaggle/kaggle.json")
    print("\n4. Accept competition rules:")
    print("   - Visit: https://www.kaggle.com/c/home-credit-default-risk/rules")
    print("   - Click 'I Understand and Accept'")
    print("\n5. Then run this script again!")
    print("="*70 + "\n")

def download_home_credit_dataset():
    """Download Home Credit Default Risk dataset"""
    data_dir = Path("data/kaggle_home_credit")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📥 Downloading Home Credit Default Risk dataset...")
    print("This may take a few minutes (dataset is ~500MB)...\n")
    
    try:
        # Download using Kaggle CLI
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', 'home-credit-default-risk'],
            cwd=str(data_dir),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ Error downloading dataset:")
            print(result.stderr)
            return False
        
        print("✓ Download complete!")
        
        # Extract zip file
        zip_files = list(data_dir.glob("*.zip"))
        if zip_files:
            print(f"\n📦 Extracting {zip_files[0].name}...")
            with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print("✓ Extraction complete!")
            
            # Optionally remove zip file to save space
            # zip_files[0].unlink()
        
        print(f"\n✅ Dataset ready at: {data_dir.absolute()}")
        print("\nFiles downloaded:")
        for file in sorted(data_dir.glob("*.csv")):
            print(f"  - {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*70)
    print("KAGGLE DATASET DOWNLOADER - Hybrid Approach")
    print("="*70)
    
    # Check Kaggle installation
    if not check_kaggle_installed():
        print("\n❌ Kaggle CLI not found!")
        setup_instructions()
        return
    
    # Check credentials
    if not check_kaggle_credentials():
        print("\n❌ Kaggle credentials not found!")
        setup_instructions()
        return
    
    # Download dataset
    success = download_home_credit_dataset()
    
    if success:
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("="*70)
        print("1. Review the downloaded CSV files")
        print("2. Load data: pd.read_csv('data/kaggle_home_credit/application_train.csv')")
        print("3. Extract structure and use for synthetic data generation")
        print("4. Combine with RBI priors (see config/priors_template.yaml)")
        print("="*70)

if __name__ == "__main__":
    main()

