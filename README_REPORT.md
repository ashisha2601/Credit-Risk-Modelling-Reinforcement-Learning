# Privacy-Preserving Credit Risk Modeling System

## Table of Contents
- [Acknowledgments](#acknowledgments)
- [Abstract](#abstract)
- [List of Tables](#list-of-tables)
- [List of Figures](#list-of-figures)
- [1. Introduction](#1-introduction)
  - [1.1. Problem Description](#11-problem-description)
  - [1.2. Problem Statement](#12-problem-statement)
  - [1.3 Existing System And Limitations](#13-existing-system-and-limitations)
    - [1.3.1. Synthesis Of Limitations From Existing Systems](#131-synthesis-of-limitations-from-existing-systems)
  - [1.4 Objectives](#14-objectives)
  - [1.5 Scope And Boundary](#15-scope-and-boundary)
    - [1.5.1 In-Scope](#151-in-scope)
    - [1.5.2 Boundary](#152-boundary)
- [2. System Analysis](#2-system-analysis)
  - [2.1 Functional Specifications](#21-functional-specifications)
  - [2.2 Block Diagram](#22-block-diagram)
  - [2.3 System Requirements](#23-system-requirements)
    - [2.3.1 Software Requirements](#231-software-requirements)
    - [2.3.2 Hardware Requirements](#232-hardware-requirements)
  - [2.4 Dataset Overview](#24-dataset-overview)
    - [2.4.1 Home Credit Default Risk Dataset](#241-home-credit-default-risk-dataset)
    - [2.4.2 RBI Statistical Priors & Census Data](#242-rbi-statistical-priors--census-data)
- [3. System Design](#3-system-design)
  - [3.1 System Architecture](#31-system-architecture)
  - [3.2 Module Design](#32-module-design)
    - [3.2.1 Data Generation Module](#321-data-generation-module)
    - [3.2.2 RBI Compliance & Market Features Module](#322-rbi-compliance--market-features-module)
    - [3.2.3 Risk Clustering Module](#323-risk-clustering-module)
    - [3.2.4 RL Policy Module](#324-rl-policy-module)
    - [3.2.5 Visualization Module](#325-visualization-module)
    - [3.2.6 Data Storage Module](#326-data-storage-module)
  - [3.3 Database Design](#33-database-design)
    - [3.3.1 Schema Design](#331-schema-design)
    - [3.3.2 Entity-Relationship (Er) Diagram](#332-entity-relationship-er-diagram)
    - [3.3.3 Data Flow Diagram](#333-data-flow-diagram)
  - [3.4 Interface Design](#34-interface-design)
    - [3.4.1 User Interface Screen Design](#341-user-interface-screen-design)
    - [3.4.2 Application Flow](#342-application-flow)
- [4. Implementation](#4-implementation)
  - [4.1 Coding Standards And Version Control](#41-coding-standards-and-version-control)
  - [4.2 Synthetic Data Generation Implementation](#42-synthetic-data-generation-implementation)
    - [4.2.1 Gaussian Copula & CTGAN](#421-gaussian-copula--ctgan)
  - [4.3 Unsupervised Risk Profiling Implementation](#43-unsupervised-risk-profiling-implementation)
    - [4.3.1 K-Prototypes Clustering](#431-k-prototypes-clustering)
  - [4.4 Reinforcement Learning Implementation](#44-reinforcement-learning-implementation)
    - [4.4.1 PPO Agent Training](#441-ppo-agent-training)
  - [4.5 Backend Implementation (Python)](#45-backend-implementation-python)
    - [4.5.1 Structure](#451-structure)
    - [4.5.2 Features](#452-features)
    - [4.5.3 Key Scripts](#453-key-scripts)
  - [4.6 Visualization Implementation](#46-visualization-implementation)
  - [4.7 Screenshots Of Implemented System](#47-screenshots-of-implemented-system)
- [5. Testing](#5-testing)
  - [5.1 Test Cases And Strategies](#51-test-cases-and-strategies)
  - [5.2 Synthetic Quality Testing And Evaluation Metrics](#52-synthetic-quality-testing-and-evaluation-metrics)
  - [5.3 Clustering Evaluation Metrics](#53-clustering-evaluation-metrics)
  - [5.4 RL Policy Evaluation](#54-rl-policy-evaluation)
  - [5.5 System Integration Testing](#55-system-integration-testing)
  - [5.6 Test Reports](#56-test-reports)
- [6. Conclusion](#6-conclusion)
  - [6.1 Design And Implementation Summary](#61-design-and-implementation-summary)
  - [6.2 Advantages And Limitations](#62-advantages-and-limitations)
    - [6.2.1 Advantages](#621-advantages)
    - [6.2.2 Limitations](#622-limitations)
  - [6.3 Future Enhancements](#63-future-enhancements)

---

## Acknowledgments
[Content for acknowledgments goes here...]

## Abstract
This project implements a privacy-preserving credit risk modeling system designed for the Indian digital lending market. By leveraging Generative AI (CTGAN, Gaussian Copula), the system generates high-fidelity synthetic credit data that mirrors the statistical properties of real-world borrowers while ensuring compliance with RBI regulations. Furthermore, it employs unsupervised machine learning (K-Prototypes) to discover natural risk segments and uses **Reinforcement Learning (Deep RL)** to train an autonomous agent that learns optimal, profit-maximizing lending policies.

## List of Tables
- Table 1: RBI Prior Distributions
- Table 2: Hardware Specifications for Training
- Table 3: Synthetic Data Quality Metrics (KS-Test, TVD)
- Table 4: RL Policy Performance vs Baseline
- ...

## List of Figures
- 2.1 Block Diagram for Synthetic Generation, Clustering, and RL
- 3.1 High-Level System Architecture Diagram
- 3.2 Schema Design (Synthetic Dataset)
- 3.3 Entity-Relationship (ER) Diagram
- 3.4 Data Flow Diagram (DFD)
- 3.5 Cluster Size Distribution Plot
- 3.6 Default Rate by Cluster Plot
- 3.7 PCA Visualization of Risk Clusters
- 3.8 RL Training Learning Curve
- 3.9 Policy Profit Comparison Chart
- 3.10 Application Flowchart
- 4.1 Hybrid Synthetic Generator Architecture
- 4.2 Risk Segmentation Process
- 4.3 Feature Correlation Heatmap

---

## 1. Introduction

### 1.1. Problem Description
Access to high-quality, granular credit data in India is severely restricted due to privacy laws (DPDPA 2023) and competitive secrecy. Financial institutions struggle to train robust AI models without exposing sensitive customer information. Existing public datasets (e.g., Kaggle) often lack the specific regulatory flags and economic context of the Indian market.

### 1.2. Problem Statement
To design and develop a "Hybrid Synthetic Data Pipeline" that combines the structural richness of global datasets with the specific statistical priors of the Indian market, and to use this data to identify borrower risk profiles through unsupervised learning and optimize lending decisions using Reinforcement Learning.

### 1.3 Existing System And Limitations
Current approaches rely either on static, foreign datasets that don't reflect local realities, or on purely statistical simulations that lack complex correlations. Traditional scorecards are static and do not adapt to changing risk environments dynamically.

#### 1.3.1. Synthesis Of Limitations From Existing Systems
- **Lack of Local Context:** Foreign datasets miss Indian-specific variables like "State GSDP" or "RBI Sectoral Credit Growth".
- **Privacy Risks:** Using real customer data for experimentation violates privacy norms.
- **Static Rules:** Traditional credit scoring relies on rigid rules rather than discovering dynamic risk clusters.
- **Suboptimal Decisioning:** Rule-based systems often reject profitable customers or approve hidden risks.

### 1.4 Objectives
- Develop a Hybrid Synthetic Generator to create privacy-safe credit data.
- Integrate RBI compliance flags (NSFR, Inoperative, FX Hedging).
- Implement K-Prototypes clustering to segment borrowers by risk.
- Train a Deep RL agent (PPO) to learn optimal lending policies.
- Visualize risk profiles to interpret model decisions.

### 1.5 Scope And Boundary

#### 1.5.1 In-Scope
- Generation of synthetic credit application data (tabular).
- Injection of Indian market priors (Census 2011, RBI Tables).
- Unsupervised clustering for risk segmentation.
- Reinforcement Learning for policy optimization.
- Static visualization of cluster characteristics and policy performance.

#### 1.5.2 Boundary
- Real-time loan processing is out of scope.
- Integration with live Credit Bureaus (CIBIL/Equifax) is excluded.
- The system is a modeling sandbox, not a production lending platform.

---

## 2. System Analysis

### 2.1 Functional Specifications
- **Data Generation:** Generate $N$ samples of synthetic borrowers with defined distributions.
- **Compliance Injection:** Automatically tag accounts with NSFR and Inoperative status based on probability.
- **Risk Profiling:** Group borrowers into Low, Medium, and High-risk clusters.
- **Policy Training:** Train an agent to Maximize Profit = (Interest Income - Default Loss).
- **Visualization:** Plot distributions of Income, Loan Amount, and Default Rates.

### 2.2 Block Diagram
(See Figure 2.1)
Priors (YAML) + Kaggle Structure -> [Hybrid Generator] -> Synthetic Data -> [Clustering Engine] -> Risk Profiles -> [RL Agent] -> Optimized Policy.

### 2.3 System Requirements

#### 2.3.1 Software Requirements
- **OS**: Linux/Windows/MacOS
- **Language**: Python 3.9+
- **Libraries**: Pandas, NumPy, Scikit-learn, SDV (Synthetic Data Vault), KModes, Seaborn, Stable-Baselines3 (RL).
- **Format**: Parquet, CSV.

#### 2.3.2 Hardware Requirements
- **CPU**: Multi-core processor (Intel i7/M1 recommended for CTGAN).
- **RAM**: 16GB minimum (for processing 200k+ rows).
- **GPU**: Optional (accelerates CTGAN training).

### 2.4 Dataset Overview

#### 2.4.1 Home Credit Default Risk Dataset
Used as a "structural template" to learn correlations between features like Income, Loan Amount, and Age. It provides the skeleton for the synthetic data.

#### 2.4.2 RBI Statistical Priors & Census Data
- **RBI Data**: Sectoral credit deployment, NPA rates, Interest rate ranges.
- **Census 2011**: State-wise population distribution, Worker classifications.
These priors are used to calibrate the synthetic data to the Indian context.

---

## 3. System Design

### 3.1 System Architecture
The system follows a pipeline architecture. The **Generator** serves as the source, feeding data into the **Validator**, which then passes valid data to the **Clustering Module**, and finally to the **RL Agent**.

### 3.2 Module Design

#### 3.2.1 Data Generation Module
Implements `HybridSyntheticGenerator`.
- **Inputs**: `priors_template.yaml`, Kaggle CSV.
- **Process**: Fits Gaussian Copula/CTGAN -> Samples Data -> Injects Priors.
- **Outputs**: `synthetic_credit_data.parquet`.

#### 3.2.2 RBI Compliance & Market Features Module
A sub-module that post-processes generated data to add:
- `NSFR_RSF_FACTOR` (Net Stable Funding Ratio)
- `INOPERATIVE_FLAG`
- `STATE` and `WORKER_TYPE`

#### 3.2.3 Risk Clustering Module
Implements `run_risk_clustering_kprototypes.py`.
- **Algorithm**: K-Prototypes (handles mixed numeric/categorical data).
- **Features**: Income, Loan Amount, Credit Score, Loan Type.
- **Output**: Cluster IDs assigned to each borrower.

#### 3.2.4 RL Policy Module
Implements `run_rl_policy_training.py`.
- **Algorithm**: Proximal Policy Optimization (PPO).
- **Environment**: `CreditLendingEnv` (Gymnasium).
- **Reward Function**: Profit maximization (Interest - Principal Loss).

#### 3.2.5 Visualization Module
Implements `run_visualize_risk_clusters_all.py`.
- Generates Bar Charts for Cluster Sizes.
- Generates Heatmaps for Default Rates.
- Generates PCA Scatter plots.

#### 3.2.6 Data Storage Module
Uses Parquet files for efficient storage of large synthetic datasets (>100MB) and CSV for human-readable samples.

### 3.3 Database Design

#### 3.3.1 Schema Design
Flat table structure simulating a Credit Application:
- **Demographics**: `AGE`, `GENDER`, `STATE`, `WORKER_TYPE`
- **Financials**: `MONTHLY_INCOME`, `EXISTING_DEBT`
- **Loan Details**: `LOAN_AMOUNT`, `TENURE`, `INTEREST_RATE`, `EMI`
- **Risk Labels**: `DEFAULT_FLAG`, `CREDIT_SCORE`, `CLUSTER_ID`

#### 3.3.2 Entity-Relationship (Er) Diagram
(See Figure 3.3)
Single major entity "Borrower/Application" with lookup relationships to "State" and "Bank Group".

#### 3.3.3 Data Flow Diagram
Priors -> [Generator] -> Raw Synthetic Data -> [Enricher] -> Compliance Data -> [Clustering] -> [RL Training] -> Policy Model.

### 3.4 Interface Design

#### 3.4.1 User Interface Screen Design
(CLI / Notebook based)
- **Configuration**: YAML file editing for priors.
- **Execution**: Command-line scripts with progress bars (`tqdm`).
- **Results**: Static PNG plots generated in `results/` folder.

#### 3.4.2 Application Flow
1. User updates `config/priors_template.yaml`.
2. User runs `run_hybrid_ctgan_pipeline.py`.
3. System generates and saves data.
4. User runs `run_risk_clustering_kprototypes.py`.
5. System assigns clusters.
6. User runs `run_rl_policy_training.py`.
7. System trains agent and saves model.
8. User runs `run_visualize_risk_clusters_all.py` to view reports.

---

## 4. Implementation

### 4.1 Coding Standards And Version Control
- **PEP 8** compliance.
- **Type Hinting** (`typing` module) used for function signatures.
- **Modular Code**: Separate files for generation, clustering, and visualization.

### 4.2 Synthetic Data Generation Implementation

#### 4.2.1 Gaussian Copula & CTGAN
Used `sdv` library.
- **Gaussian Copula**: Models correlations between variables using statistical copulas. Fast and efficient.
- **CTGAN**: Conditional GANs for modeling complex, non-linear distributions.

```python
synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(reference_data)
synthetic_data = synthesizer.sample(num_rows=n_samples)
```

### 4.3 Unsupervised Risk Profiling Implementation

#### 4.3.1 K-Prototypes Clustering
Used `kmodes` library.
- Solves the problem of clustering data with both numerical (Income, Age) and categorical (Loan Type, State) features.
- Cost function combines Euclidean distance (for numerical) and Hamming distance (for categorical).

### 4.4 Reinforcement Learning Implementation

#### 4.4.1 PPO Agent Training
Used `stable_baselines3`.
- **Environment**: Custom `CreditLendingEnv` defining State, Action (Approve/Reject), and Reward.
- **Agent**: PPO with MLP Policy.

```python
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
```

### 4.5 Backend Implementation (Python)

#### 4.5.1 Structure
- `src/`: Core logic (`hybrid_synthetic_generator.py`)
- `config/`: Configuration files.
- `scripts/`: Execution wrappers (`run_*.py`).

#### 4.5.2 Features
- **Config-Driven**: Behavior controlled by YAML.
- **Reproducible**: Random seeds fixed for consistency.
- **Scalable**: Batch processing for large datasets.

#### 4.5.3 Key Scripts
- `src/hybrid_synthetic_generator.py`: The brain of the generation.
- `run_risk_clustering_kprototypes.py`: The intelligence for segmentation.
- `run_rl_policy_training.py`: The decision-making agent.

### 4.6 Visualization Implementation
Uses `matplotlib` and `seaborn` to generate static report images.

### 4.7 Screenshots Of Implemented System
- Figure 4.1: Terminal output of Generation Pipeline.
- Figure 4.2: Cluster Profile CSV output.
- Figure 4.3: RL Training Logs.

---

## 5. Testing

### 5.1 Test Cases And Strategies
- **Statistical Similarity**: Check if Synthetic Mean $\approx$ Prior Mean.
- **Logic Checks**: Ensure `LOAN_AMOUNT` < `MONTHLY_INCOME` * `MAX_LTI`.
- **Compliance Checks**: Verify `INOPERATIVE_FLAG` is present.

### 5.2 Synthetic Quality Testing And Evaluation Metrics
- **KS-Test (Kolmogorov-Smirnov)**: Measures similarity of column distributions.
- **Correlation Similarity**: Compares correlation matrices of Real vs. Synthetic data.

### 5.3 Clustering Evaluation Metrics
- **Elbow Method**: To determine optimal $k$ (number of clusters).
- **Cluster Stability**: Consistency of profiles across runs.

### 5.4 RL Policy Evaluation
- **Profit Comparison**: Compare Total Profit of RL Policy vs. "Always Approve" vs. "Random".
- **Default Rate**: Ensure RL Policy Default Rate < Baseline.

### 5.5 System Integration Testing
Verified the full pipeline flow:
Generation $\rightarrow$ Save $\rightarrow$ Load $\rightarrow$ Cluster $\rightarrow$ Train RL $\rightarrow$ Visualize.

### 5.6 Test Reports
Generated in `results/` folder, summarizing distribution matches, cluster distinctiveness, and policy profitability.

---

## 6. Conclusion

### 6.1 Design And Implementation Summary
The project successfully delivered a robust pipeline for generating privacy-preserving credit data that respects Indian regulatory norms. The clustering module effectively identified distinct borrower segments, and the Reinforcement Learning agent demonstrated the ability to learn profitable lending strategies autonomously.

### 6.2 Advantages And Limitations

#### 6.2.1 Advantages
- **Zero Privacy Risk**: No real user data is exposed.
- **Regulatory Readiness**: Built-in RBI flags.
- **Adaptive Decisioning**: RL agent learns from data, not just static rules.
- **Cost-Effective**: No need to purchase expensive bureau data for initial modeling.

#### 6.2.2 Limitations
- **Synthetic Fidelity**: Complex "black swan" events may not be captured.
- **Dependency on Priors**: Quality of output depends on the accuracy of input statistics.

### 6.3 Future Enhancements
- Integration with Real-time credit bureaus for hybrid decisioning.
- Development of a web-based UI (Streamlit/React) for interactive parameter tuning.
- Expansion to include SME and Corporate lending modules.
