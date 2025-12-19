# System Diagrams Flowcharts

## Figure 1: System Design Overview
**Flow:**
[Data Sources] --> [Synthetic Generation Module] --> [Risk Clustering Module] --> [RL Policy Module] --> [Dashboard/Reporting]

**Details:**
- **Data Sources:** RBI Priors (Config), Kaggle Dataset (Template)
- **Synthetic Generation:** Hybrid Engine (Copula/CTGAN)
- **Risk Clustering:** Unsupervised Learning (K-Prototypes)
- **RL Policy:** PPO Agent (Reinforcement Learning)
- **Dashboard:** Plots, CSV Reports, Metric Logs

---

## Figure 2: High-Level Architecture
**Layers:**
1.  **Input Layer**
    - `priors_template.yaml`
    - `application_train.csv`
2.  **Processing Layer**
    - `SDV` (Synthesis)
    - `Scikit-learn` (Preprocessing)
    - `Gymnasium` (RL Environment)
3.  **Core Engines**
    - `GaussianCopulaSynthesizer`
    - `KPrototypes`
    - `Stable-Baselines3 PPO`
4.  **Storage & Output Layer**
    - `Parquet Files` (Data)
    - `Seaborn/Matplotlib` (Viz)

---

## Figure 3: Synthetic Data Generation Block Diagram
**Steps:**
1.  **Input:** Real Data Template + Statistical Priors.
2.  **Learn:** Fit Probabilistic Model (Copula) to learn correlations.
3.  **Sample:** Generate random samples from learned distribution.
4.  **Enrich:** Apply Deterministic Rules (RBI Flags: NSFR, Inoperative).
5.  **Validate:** Check constraints (Loan < Income * Limit).
6.  **Output:** Synthetic Credit Dataset.

---

## Figure 4: Entity Relationship Diagram (ERD)
**Entities & Relationships:**
1.  **Borrower (Primary Entity)**
    - *Attributes:* ID, Age, Income, Gender, Credit Score
    - *Relationships:* Has-Many Loans.
2.  **Loan (Weak Entity)**
    - *Attributes:* Loan ID, Type (Home/Personal), Amount, Tenure, EMI
    - *Relationships:* Belongs-To Borrower.
3.  **Bank_Group (Lookup Entity)**
    - *Attributes:* Type (PSB/PVB), Interest Rate Range
    - *Relationships:* Issues Loan.
4.  **State_Economics (Lookup Entity)**
    - *Attributes:* State Name, GSDP, Default Risk Factor
    - *Relationships:* Borrower Resides-In State.

---

## Figure 5: Data Flow Diagram (DFD)
**Process:**
1.  **Config (User)** -> [Load Settings] -> **Generator**.
2.  **Generator** -> [Compute Stats] -> **Synthesizer**.
3.  **Synthesizer** -> [Sample Rows] -> **Raw Data**.
4.  **Raw Data** -> [Inject RBI Flags] -> **Processed Data**.
5.  **Processed Data** -> [Clustering Alg] -> **Labeled Data**.
6.  **Labeled Data** -> [RL Environment] -> **Policy Training**.
7.  **Policy Training** -> [Evaluation] -> **Final Report**.

---

## Figure 6: Cluster Visualization (PCA)
**Steps:**
1.  **Input:** High-Dimensional Data (Income, Loan, Score, Rate...).
2.  **Transform:** Standard Scaling (Mean=0, Var=1).
3.  **Reduce:** PCA (Principal Component Analysis) -> Project to 2D (PC1, PC2).
4.  **Map:** Assign Colors to Clusters (0=Blue, 1=Orange, etc.).
5.  **Plot:** Scatter Plot showing distinct separation of Risk Groups.

---

## Figure 7: Correlation Heatmap
**Components:**
1.  **Matrix Grid:** X-axis (Features) vs Y-axis (Features).
2.  **Cells:** Color intensity represents Correlation Coefficient (-1 to +1).
3.  **Goal:** Compare **Real Data Matrix** vs **Synthetic Data Matrix**.
4.  **Success Criteria:** Similar patterns (e.g., Income & Loan Amount should be highly positive in both).

---

## Figure 8: Reinforcement Learning Agent Architecture
**Loop:**
1.  **Agent (PPO):** Observes State $S_t$.
2.  **Policy ($\pi$):** Outputs Action $A_t$ (Approve/Reject).
3.  **Environment (Credit Market):**
    - Processes Action.
    - Simulates Outcome (Repay or Default).
4.  **Feedback:**
    - Returns Reward $R_t$ (Profit or Loss).
    - Returns Next State $S_{t+1}$.
5.  **Update:** Agent updates weights to maximize long-term Reward.

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
