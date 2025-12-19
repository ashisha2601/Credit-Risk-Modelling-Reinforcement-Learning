# System Diagrams Flowcharts

## 1. Block Diagram (2.1)
**Flow:**
[Data Sources] --> [Hybrid Generator] --> [Synthetic Data] --> [Clustering Engine] --> [Risk Insights] --> [RL Policy Agent]

**Details:**
- **Data Sources:** RBI Priors (YAML), Kaggle Template (CSV)
- **Hybrid Generator:** Gaussian Copula / CTGAN + RBI Rules Injection
- **Synthetic Data:** Privacy-Preserving Dataset (Parquet)
- **Clustering Engine:** K-Prototypes Algorithm (Unsupervised Learning)
- **Risk Insights:** Low/Medium/High Risk Segments
- **RL Policy Agent:** PPO (Reinforcement Learning) for Credit Decisioning

---

## 2. High-Level Architecture (3.1)
**Layers:**
1.  **Input Layer**
    - `priors_template.yaml` (Config)
    - `application_train.csv` (Template Data)

2.  **Processing Layer**
    - `SDV Library` (Data Synthesis)
    - `NumPy/Pandas` (Rule Injection)
    - `Scikit-learn` (Scaling/Encoding)

3.  **Model Layer**
    - `GaussianCopulaSynthesizer` (Generative Model)
    - `KPrototypes` (Clustering Model)
    - `PPO Agent` (Reinforcement Learning Policy)

4.  **Output Layer**
    - `Synthetic Dataset` (File Storage)
    - `Visualizations` (Matplotlib/Seaborn Plots)
    - `Policy Evaluation` (Profit/Loss Reports)

---

## 3. Data Flow Diagram (DFD) (3.4)
**Steps:**
1.  **User** starts `run_hybrid_ctgan_pipeline.py`.
2.  **System** loads `priors_template.yaml`.
3.  **System** fits Generative Model to Kaggle Template.
4.  **System** samples $N$ rows of base synthetic data.
5.  **System** injects RBI Flags (NSFR, Inoperative) based on rules.
6.  **System** saves `synthetic_data.parquet`.
7.  **System** loads data into `run_risk_clustering.py`.
8.  **System** performs Clustering & Assigns Labels (Low/Med/High).
9.  **System** saves `clustered_data.parquet`.
10. **System** feeds clustered data to `run_rl_policy_training.py`.
11. **System** trains PPO Agent and outputs `policy_evaluation.csv`.

---

## 4. Application Flowchart (3.9)
**Decision Flow:**
1.  **Start**
2.  **Is Kaggle Data Available?**
    - *Yes:* Load and learn structure.
    - *No:* Use Priors-Only mode.
3.  **Generate Base Data** (Age, Income, Loan Amt).
4.  **Apply RBI Rules?**
    - *Yes:* Inject NSFR, Sectoral Growth, etc.
5.  **Save Synthetic Data**.
6.  **Run Clustering?**
    - *Yes:* Execute K-Prototypes.
7.  **Run RL Training?**
    - *Yes:* Train PPO Agent on Clustered Data.
8.  **Generate Plots & Reports**.
9.  **End**.

---

## 5. Model Architecture (4.1 - Gaussian Copula)
**Components:**
1.  **Input Data** (Marginal Distributions).
2.  **Transformation** (Convert to Standard Normal).
3.  **Copula Function** (Learn Correlations / Covariance Matrix).
4.  **Sampling** (Sample from Multivariate Normal).
5.  **Inverse Transformation** (Convert back to original distributions).
6.  **Output** (Synthetic Row).

---

## 6. PCA Cluster Visualization Flow (4.2)
**Steps:**
1.  **Input:** Clustered Dataset (High Dimensional).
2.  **Preprocessing:** Standard Scaler (Normalize Income, Loan Amt).
3.  **Dimensionality Reduction:** PCA (Reduce to 2 Components: PC1, PC2).
4.  **Plotting:** Scatter Plot (X=PC1, Y=PC2).
5.  **Coloring:** Color points by `RISK_SEGMENT` (Low=Green, High=Red).

---

## 7. RL Agent Interaction Loop (4.3 - NEW)
**Cycle:**
1.  **Environment State ($S_t$):** Customer Profile (Income, Loan Amt, Cluster ID).
2.  **Agent Action ($A_t$):** Decision (Approve / Reject).
3.  **Environment Step:**
    - If **Reject**: Reward = 0.
    - If **Approve**: Check True Outcome (Default vs Repay).
        - **Repay**: Reward = +Interest.
        - **Default**: Reward = -Principal Loss.
4.  **Update Policy:** PPO algorithm adjusts weights to maximize expected reward.
5.  **Next State ($S_{t+1}$):** Next Customer Profile.
