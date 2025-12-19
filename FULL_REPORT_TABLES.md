# Complete List of Tables for Project Report

## Table 2.1: Software Requirements Specification
| Component | Technology/Version | Purpose |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 / macOS 12+ / Linux | Development and Execution Environment. |
| **Programming Language** | Python 3.9+ | Core logic implementation. |
| **Database** | Apache Parquet / CSV | Efficient data storage for large datasets. |
| **IDE** | VS Code / Jupyter Notebook | Code development and experimentation. |
| **Version Control** | Git & GitHub | Source code management. |

---

## Table 2.2: Hardware Requirements Specification
| Component | Minimum Specification | Recommended Specification | Purpose |
| :--- | :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 (8th Gen) / AMD Ryzen 5 | Intel Core i7 / Apple M2 | Data processing and model training. |
| **Memory (RAM)** | 8 GB | 16 GB or higher | In-memory dataset handling and clustering. |
| **Storage** | 256 GB SSD | 512 GB NVMe SSD | Fast I/O for synthetic data generation. |
| **GPU (Optional)** | Integrated Graphics | NVIDIA GTX 1660+ | Accelerates Deep Learning (CTGAN, RL). |

---

## Table 3.1: Feature Categorization of Credit Dataset
| Category | Features Included | Description |
| :--- | :--- | :--- |
| **Demographic** | `AGE`, `GENDER`, `STATE`, `WORKER_TYPE` | Personal attributes of the borrower. |
| **Financial** | `MONTHLY_INCOME`, `EXISTING_DEBT`, `CREDIT_SCORE` | Financial health indicators. |
| **Loan Details** | `LOAN_AMOUNT`, `LOAN_TENURE_MONTHS`, `INTEREST_RATE` | Specifics of the requested credit. |
| **RBI/Compliance** | `NSFR_RSF_FACTOR`, `INOPERATIVE_FLAG`, `FX_HEDGING_FLAG` | Regulatory flags mandated by RBI. |
| **Target** | `DEFAULT_FLAG` | Binary indicator (0=Repay, 1=Default). |

---

## Table 3.2: Summary of Synthetic Data Attributes
| Attribute | Value |
| :--- | :--- |
| **Total Samples** | 100,000 (Priors-Only Dataset) |
| **Number of Features** | 18 Columns |
| **Data Types** | Numerical (12), Categorical (6) |
| **File Size** | ~15 MB (Parquet) |
| **Key Distribution** | Income (Lognormal), Age (Normal), Loan Amount (Derived) |

---

## Table 4.1: Synthetic Borrower Dataset Schema
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `SK_ID_CURR` | Integer | Unique identifier for the loan application. |
| `MONTHLY_INCOME` | Float | Borrower's monthly income in INR. |
| `LOAN_AMOUNT` | Float | Total principal amount requested. |
| `LOAN_TYPE` | String | Category (Home, Personal, Vehicle, etc.). |
| `CREDIT_SCORE` | Integer | CIBIL-like credit score (300-900). |
| `DEFAULT_FLAG` | Integer | 1 if borrower defaulted, 0 otherwise. |
| `CLUSTER_ID` | Integer | Assigned Risk Segment (0-4). |

---

## Table 4.2: Risk Cluster Feature Summary
| Feature Name | Type | Usage in Clustering |
| :--- | :--- | :--- |
| `MONTHLY_INCOME` | Numeric | Standard Scaled & used for Euclidean distance. |
| `LOAN_AMOUNT` | Numeric | Standard Scaled & used for Euclidean distance. |
| `CREDIT_SCORE` | Numeric | Standard Scaled & used for Euclidean distance. |
| `INTEREST_RATE` | Numeric | Standard Scaled & used for Euclidean distance. |
| `LOAN_TYPE` | Categorical | Used for Hamming distance (matching). |
| `STATE` | Categorical | Used for Hamming distance. |

---

## Table 5.1: Implementation Environment and Tools
| Tool/Library | Version | Functionality |
| :--- | :--- | :--- |
| **SDV** | 1.28.0 | Synthetic Data Generation (Gaussian Copula). |
| **Scikit-learn** | 1.7.1 | Preprocessing (StandardScaler) and PCA. |
| **KModes** | 0.12.2 | K-Prototypes Clustering Algorithm. |
| **Stable-Baselines3** | 2.7.1 | PPO Reinforcement Learning Agent. |
| **Gymnasium** | 1.2.2 | RL Environment Interface. |
| **Seaborn** | 0.13.2 | Statistical Data Visualization. |

---

## Table 5.2: Clustering Algorithm Parameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Algorithm** | K-Prototypes | Mixed-type clustering. |
| **n_clusters** | 5 | Number of risk segments identified. |
| **init** | 'Huang' | Initialization method for centroids. |
| **n_init** | 10 | Number of times algorithm runs with different seeds. |
| **max_iter** | 100 | Maximum iterations for convergence. |
| **random_state** | 42 | Seed for reproducibility. |

---

## Table 6.1: Sample Test Cases for System Validation
| Test Case ID | Description | Input | Expected Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | **Data Generation** | Priors Config | 100k rows generated with valid ranges. | Pass |
| **TC-02** | **Logic Check** | Generated Data | Loan Amount < Income * Max_LTI. | Pass |
| **TC-03** | **Compliance Check** | Generated Data | NSFR column contains {65, 100} only. | Pass |
| **TC-04** | **Clustering** | Mixed Data | Clusters assigned 0-4 for all rows. | Pass |
| **TC-05** | **RL Training** | Clustered Data | Agent learns policy (reward increases). | Pass |

---

## Table 6.2: Synthetic Data Quality Metrics
| Metric | Description | Target | Achieved |
| :--- | :--- | :--- | :--- |
| **KS-Test** | Measures distribution similarity (Lower is better). | < 0.15 | **0.09** |
| **TVD** | Categorical distribution distance (Lower is better). | < 0.15 | **0.12** |
| **Correlation** | Matrix correlation similarity (Higher is better). | > 0.90 | **0.94** |

---

## Table 6.3: Cluster Quality Evaluation Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Cost Function** | 2.45e11 | Total intra-cluster distance (lower indicates tighter clusters). |
| **Segment Distinctness** | High | Clear separation between "Home Loan" (Low Risk) and "Personal Loan" clusters. |
| **Default Rate Range** | 0.99% - 3.21% | Good spread of risk, allowing for meaningful differentiation. |

---

## Table 6.4: Reinforcement Learning Policy Performance Comparison
| Metric | Always Approve | Random Policy | PPO Agent (RL) |
| :--- | :--- | :--- | :--- |
| **Total Profit** | ₹1.18 Billion | ₹0.58 Billion | ₹0.76 Billion |
| **Avg Profit/Loan** | ₹59,303 | ₹29,234 | ₹38,032 |
| **Improvement** | N/A | Baseline | **+30% vs Random** |

---

## Table 7.1: Risk Cluster Profiles and Default Rates
| Cluster ID | Risk Segment | Default Rate | Size (%) | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **4** | **Low Risk** | 0.99% | 17.4% | High Income, Home Loans (Prime). |
| **1** | **Medium Risk** | 1.77% | 1.6% | Short-term, High Interest Loans. |
| **0** | **Medium Risk** | 3.10% | 31.5% | Older Borrowers, Standard Profile. |
| **2** | **Medium Risk** | 3.13% | 12.7% | **Highest Income**, High Debt Burden. |
| **3** | **High Risk** | 3.21% | 36.8% | **Youngest**, Low Income, New-to-Credit. |

---

## Table 7.2: Policy Performance Comparison (Always Approve vs RL vs Random)
| Strategy | Description | Net Profit | Risk Behavior |
| :--- | :--- | :--- | :--- |
| **Always Approve** | Approves every application. | **₹1.18 B** | High Risk (Accepts all defaults). |
| **PPO Agent (RL)** | Learns to select profitable loans. | **₹0.76 B** | **Balanced** (Avoids bad loans). |
| **Random Policy** | Approves 50% randomly. | ₹0.58 B | No Strategy. |
