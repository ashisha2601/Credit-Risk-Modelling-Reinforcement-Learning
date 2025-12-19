# List of Tables for Project Report

## Table 2.1: Software Requirements Specifications
| Component | Technology/Version | Purpose |
| :--- | :--- | :--- |
| **Programming Language** | Python 3.9+ | Core logic for generation, clustering, and RL. |
| **Generative AI** | SDV (Synthetic Data Vault) 1.28.0 | Implementation of Gaussian Copula and CTGAN models. |
| **Machine Learning** | Scikit-learn 1.7.1 | Data preprocessing, scaling, and PCA. |
| **Clustering** | KModes / KPrototypes | Unsupervised clustering for mixed-type data. |
| **Reinforcement Learning** | Stable-Baselines3 2.7.1 | PPO algorithm implementation and policy training. |
| **Environment** | Gymnasium 1.2.2 | Custom Credit Lending Environment. |
| **Data Processing** | Pandas 2.3.2, NumPy | Data manipulation and statistical calculations. |
| **Visualization** | Matplotlib, Seaborn | Generating plots for clusters and policy analysis. |

---

## Table 2.2: Hardware Requirements Specifications
| Component | Minimum Specification | Recommended Specification | Purpose |
| :--- | :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 (10th Gen) / Apple M1 | Intel Core i7 / Apple M2/M3 | Handling copula math and RL simulations. |
| **Memory (RAM)** | 16 GB | 32 GB or higher | Required to load 300k+ rows and train ML models. |
| **Storage** | 256 GB SSD | 512 GB NVMe SSD | Fast I/O for Parquet files and model checkpoints. |
| **GPU (Optional)** | Integrated Graphics | NVIDIA RTX 3060 (8GB VRAM) | Acceleration for Deep Learning (CTGAN, PPO). |

---

## Table 2.3: RBI Statistical Priors (Input Parameters)
| Parameter Category | Variable | Value Range / Distribution | Source |
| :--- | :--- | :--- | :--- |
| **Regulatory Flags** | NSFR_RSF_FACTOR | {65%, 100%} | RBI Master Direction 2023 |
| **Compliance** | INOPERATIVE_FLAG | ~3% Probability | RBI Guidelines 2024 |
| **Interest Rates** | Home Loan Rate | 8.35% - 8.75% | RBI/Bank Average Rates |
| **Interest Rates** | Personal Loan Rate | 9.55% - 11.56% | Market Data |
| **Credit Risk** | Default Rate (Retail) | ~2.5% | RBI Financial Stability Report |
| **Demographics** | Worker Type | Main, Marginal, Non-Worker | Census 2011 |

---

## Table 5.1: Test Cases for System Functionality
| Test Case ID | Description | Input | Expected Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC-GEN-01** | **Generate Hybrid Data** | Priors YAML + Kaggle Template | Dataset with N samples created successfully. | Pass |
| **TC-RBI-01** | **Verify NSFR Flags** | Generated Dataset | Column contains only valid values {65, 100}. | Pass |
| **TC-PRIV-01** | **Privacy Check** | Real vs Synthetic Data | No exact row matches (Distance > 0). | Pass |
| **TC-CLUST-01** | **Clustering Execution** | Mixed-Type Data | Borrowers assigned to discrete clusters (0-4). | Pass |
| **TC-RL-01** | **Policy Training** | Clustered Data | PPO Agent completes training without NaN errors. | Pass |
| **TC-RL-02** | **Policy Evaluation** | Test Dataset | RL Policy Profit > Random Policy Profit. | Pass |

---

## Table 5.2: Synthetic Data Quality Metrics
| Metric | Description | Target Score | Achieved Score |
| :--- | :--- | :--- | :--- |
| **KS-Test (Average)** | Kolmogorov-Smirnov statistic (Lower is better). Measures distribution similarity. | < 0.15 | **0.09** |
| **TVD (Categorical)** | Total Variation Distance for categorical columns (e.g., Loan Type). | < 0.15 | **0.12** |
| **Correlation Similarity** | Comparison of Heatmaps (Real vs Synthetic). | > 0.90 | **0.94** |
| **Privacy Risk Score** | Distance to Closest Record (DCR) analysis. | > Threshold | **Safe** |

---

## Table 5.3: Risk Clustering Model Performance
| Cluster ID | Risk Segment | Default Rate | Size (Share) | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **4** | **Low Risk** | 0.99% | 17.4% | High Income, Long Tenure, Home Loans. |
| **0** | **Medium Risk A** | 3.10% | 31.5% | Older (45y), Moderate Income, Standard Loans. |
| **2** | **Medium Risk B** | 3.13% | 12.7% | **Highest Income**, but High EMI Burden. |
| **3** | **High Risk** | 3.21% | 36.8% | **Youngest (26y)**, Low Income, New-to-Credit. |
| **1** | **Medium Risk C** | 1.77% | 1.6% | Short-term, High-Interest Personal Loans. |

---

## Table 5.4: RL Policy Performance Evaluation
| Policy Strategy | Total Profit (INR) | Avg Profit per Loan | Performance vs Baseline |
| :--- | :--- | :--- | :--- |
| **Always Approve** | ₹1.18 Billion | ₹59,303 | N/A (Theoretical Max) |
| **PPO Agent (RL)** | **₹0.76 Billion** | **₹38,032** | **+30% over Random** |
| **Random Policy** | ₹0.58 Billion | ₹29,234 | Baseline |
