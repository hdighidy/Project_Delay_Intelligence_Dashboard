# 📊 Project Delay Intelligence

> **Explainable AI for Construction Project Delay Risk Prediction & Operational Decision Support**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![CatBoost](https://img.shields.io/badge/ML-CatBoost-yellowgreen)](https://catboost.ai/)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple)](https://shap.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Visualization-Plotly-3F4F75)](https://plotly.com/)

---

## 🎯 Executive Summary

**Project Delay Intelligence** is an end-to-end Data Science and Explainable AI prototype designed to support construction project managers and executives in identifying projects with a higher risk of schedule delay.

The solution combines:

- Historical project portfolio analysis
- Machine Learning-based delay risk prediction
- New-project risk scoring
- SHAP-based explainability
- Interactive management dashboards
- Data leakage and model governance controls

The key business question is:

> **"Given what we know before a project starts, how likely is this project to experience a delay, and what factors are driving that risk?"**

The current dataset contains **150 construction projects** with a historical delay rate of approximately **26%**.

---

# 🏗️ Business Problem

Construction organizations typically manage large project portfolios where schedule delays can affect:

- Project completion dates
- Resource planning
- Procurement and material planning
- Cash-flow expectations
- Client commitments
- Management attention
- Overall portfolio performance

Traditional reporting often answers:

> **"Which projects are already delayed?"**

This project moves the analytical approach toward:

> **"Which new projects are more likely to experience delay, and why?"**

This transforms the solution from **descriptive reporting** into **predictive and explainable decision support**.

---

# 💡 Solution

The solution contains two complementary analytical layers.

### 1. Portfolio Intelligence

The dashboard analyzes the historical project portfolio to identify patterns in delay behavior.

Examples include:

- Delay rate by project type
- Delay rate by project complexity
- Delay rate by location
- Contract value vs. planned duration
- Delayed vs. on-time projects
- Project-level portfolio analysis

### 2. New Project Risk Prediction

Management can enter information available **before project execution**:

| Input | Purpose |
|---|---|
| Contract Value | Project scale / commercial exposure |
| Planned Duration | Schedule exposure |
| Project Type | Project category |
| Location | Geographic context |
| Client Type | Client segmentation |
| Project Complexity | Operational complexity |

The model returns:

**Predicted Delay Probability**

together with:

**LOW / MEDIUM / HIGH risk classification**

---

# 🧠 Data Science Methodology

## Target Definition

The prediction target is:

```text
delay_flag
```

where:

```text
0 = On Time
1 = Delayed
```

Current portfolio:

```text
Projects:      150
Delay Rate:    ~26%
```

---

# 🚨 Target Leakage Prevention

One of the most important design decisions in this project is preventing **target leakage**.

### Prediction-time features

The model uses information that can realistically be available before project execution:

```text
contract_value
planned_duration_days
project_type
location
client_type
project_complexity
```

### Excluded features

The following fields are deliberately excluded from the new-project prediction model:

```text
actual_start_date
actual_end_date
actual_duration_days
delay_days
project_status
delay_flag
```

These variables either describe the outcome itself or become available only after/during project execution.

For example, using:

```text
delay_days
```

to predict:

```text
delay_flag
```

would produce a misleadingly strong model.

Therefore:

> **Only information genuinely available at prediction time is used to score a new project.**

---

# 🤖 Machine Learning

The prototype uses **CatBoost Classifier**.

### Why CatBoost?

Construction project datasets commonly contain mixed numerical and categorical information:

- Project type
- Location
- Client type
- Complexity
- Contract value
- Planned duration

CatBoost is well suited to this type of data and provides an effective baseline without requiring extensive manual categorical encoding.

### Model configuration

```text
Iterations:       300
Depth:              4
Learning Rate:   0.04
L2 Regularization: 5
Random Seed:       42
```

The model is evaluated using **5-fold stratified cross-validation**.

---

# 📈 Model Evaluation

The notebook evaluates the model using:

- Fold-level ROC-AUC
- Mean ROC-AUC
- Standard deviation

Because the current dataset contains only **150 projects**, the resulting performance should be treated as an early prototype signal rather than a production guarantee.

For production deployment, the recommended approach is to:

1. Expand the historical project portfolio.
2. Use chronological train/validation/test datasets.
3. Calibrate predicted probabilities.
4. Perform detailed error analysis.
5. Monitor model performance after deployment.

---

# 🔎 Explainable AI

A risk probability alone is not sufficient for operational decision-making.

The project therefore integrates **SHAP (SHapley Additive exPlanations)**.

For every new project, SHAP identifies the features contributing most strongly to the prediction.

### Example interpretation

```text
Project Complexity
        ↓
Increases delay risk

Planned Duration
        ↓
Increases delay risk

Client Type
        ↓
Reduces delay risk
```

This allows management to move from:

> **"The project has 68% delay risk."**

to:

> **"The model predicts elevated delay risk, primarily because of these specific project characteristics."**

### Important

SHAP explains **model contribution**.

It does not automatically prove that a feature is a causal driver of project delay.

---

# 📊 Interactive Dashboard

The project uses **Streamlit + Plotly** to provide an interactive analytical application.

## 📈 Portfolio Analysis

The dashboard provides:

- Total projects
- Historical delay rate
- Delayed project count
- Average delay days
- Portfolio contract value
- Delay rate by complexity
- Delay rate by project type
- Delay rate by location
- Contract value vs. planned duration
- Project-level portfolio table

---

## 🚨 New Project Risk

Users can enter a hypothetical new project:

```text
Contract Value
Planned Duration
Project Type
Location
Client Type
Project Complexity
```

The system produces:

```text
Predicted Delay Probability
Risk Classification
Historical Portfolio Benchmark
```

Example:

```text
Predicted Delay Probability
        ↓
      64.7%

Risk Level
        ↓
       HIGH
```

The actual value is generated dynamically by the trained model.

---

## 🔎 Explainability

The dashboard provides:

- Global feature importance
- Individual project explanation
- SHAP contribution
- Risk-increasing factors
- Risk-reducing factors

This creates an interpretable bridge between:

```text
Machine Learning
        ↓
Business Risk
        ↓
Management Decision
```

---

# 🏛️ Solution Architecture

```text
                 ┌────────────────────────┐
                 │ Historical Project Data │
                 │      CSV Dataset       │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Data Quality / EDA   │
                 │   Target & Leakage     │
                 │       Audit            │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │ Prediction-Time        │
                 │ Feature Selection      │
                 └────────────┬───────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │     CatBoost ML        │
                 │   Delay Classifier     │
                 └────────────┬───────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
     ┌────────────────────┐       ┌────────────────────┐
     │ Delay Probability  │       │   SHAP Explainable │
     │    Risk Score      │       │       AI           │
     └─────────┬──────────┘       └─────────┬──────────┘
               │                            │
               └─────────────┬──────────────┘
                             ▼
                 ┌────────────────────────┐
                 │   Streamlit Dashboard  │
                 │                        │
                 │ Portfolio Intelligence │
                 │ New Project Risk       │
                 │ Explainability         │
                 └────────────────────────┘
```

---

# 🧰 Technology Stack

| Layer | Technology |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | CatBoost |
| Model Evaluation | Scikit-learn |
| Explainable AI | SHAP |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Data Source | CSV |
| Development | Jupyter Notebook |

The implementation uses open-source technologies.

---

# 📁 Repository Structure

```text
project-delay-intelligence/
│
├── Project_Delay_Intelligence_Dashboard.ipynb
├── app.py
├── project_dataset.csv
├── requirements.txt
├── README.md
│
└── data/
    ├── raw/
    └── processed/
```

For a production implementation, the architecture can be further modularized:

```text
src/
├── data/
├── features/
├── models/
├── evaluation/
├── explainability/
└── dashboard/
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd project-delay-intelligence
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Dashboard

Place the dataset beside `app.py`:

```text
project_dataset.csv
```

Then run:

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

# 📓 Run the Notebook

Open:

```text
Project_Delay_Intelligence_Dashboard.ipynb
```

The notebook covers:

1. Dataset loading
2. Data inspection
3. Target analysis
4. Leakage audit
5. Prediction-time feature selection
6. Cross-validation
7. CatBoost training
8. Global feature importance
9. New-project prediction
10. SHAP explanation
11. Streamlit dashboard generation

---

# 📌 Example Decision Workflow

A project manager receives a new project opportunity.

### Step 1 — Enter project information

```text
Contract Value
Planned Duration
Project Type
Location
Client Type
Project Complexity
```

### Step 2 — Generate risk score

```text
Delay Probability = XX.X%
```

### Step 3 — Classify risk

```text
LOW
MEDIUM
HIGH
```

### Step 4 — Investigate drivers

SHAP identifies the strongest factors contributing to the prediction.

### Step 5 — Management action

The risk information can support:

- Early project review
- Planning scrutiny
- Resource allocation
- Procurement planning
- Management attention
- Risk mitigation planning

The model is designed to **support**, not replace, professional project-management judgment.

---

# ⚠️ Model Governance & Limitations

## Dataset Size

The current dataset contains only:

```text
150 projects
```

A production model should be trained and validated using a substantially larger historical portfolio.

## Probability Calibration

The predicted percentage should not yet be interpreted as a formally calibrated probability.

Probability calibration should be performed before using the score as a formal enterprise risk measure.

## Temporal Validation

A production implementation should preferably use:

```text
Historical Projects
        ↓
     Training
        ↓
Later Projects
        ↓
    Validation
        ↓
Future Projects
        ↓
       Test
```

This better represents how the model will perform on future projects.

## Data Provenance

If the dataset is synthetic or generated for portfolio development, observed relationships should not be presented as proven real-world causal relationships.

## Explainability

SHAP explains how the model contributed to a prediction.

It does **not** establish causal relationships.

---

# 🔮 Future Roadmap

## Phase 1 — Current Prototype

- [x] Historical portfolio analysis
- [x] Delay target definition
- [x] Leakage audit
- [x] CatBoost model
- [x] Cross-validation
- [x] New-project scoring
- [x] SHAP explainability
- [x] Streamlit dashboard

## Phase 2 — Advanced Data Science

- [ ] Temporal train / validation / test split
- [ ] Probability calibration
- [ ] Hyperparameter optimization
- [ ] Model comparison
- [ ] Precision / Recall analysis
- [ ] Threshold optimization
- [ ] Error analysis
- [ ] Model drift monitoring

## Phase 3 — Construction Operations Intelligence

Future features could include:

- Procurement risk
- Material lead-time risk
- Resource availability
- Equipment utilization
- Project progress signals
- Cash-flow indicators
- Contractor/subcontractor performance
- Primavera / scheduling integration

## Phase 4 — Enterprise AI

```text
                 Project Data
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   Delay Risk     Cost Risk    Procurement Risk
        │             │             │
        └─────────────┼─────────────┘
                      ▼
              Enterprise Risk Engine
                      │
                      ▼
              AI Decision Support
                      │
                      ▼
             Executive Control Tower
```

---

# 💼 Business Value

The long-term objective is to evolve from:

> **Reporting what happened**

to:

> **Predicting what is likely to happen**

and ultimately:

> **Explaining why it may happen and helping management determine where to focus.**

This creates a foundation for an **Enterprise Operations Intelligence** platform for construction organizations.

---

# 🧑‍💻 Data Science Engineering Highlights

This project demonstrates practical experience across:

- **Predictive Analytics**
- **Machine Learning**
- **Explainable AI**
- **Feature Engineering**
- **Target Leakage Detection**
- **Model Validation**
- **Risk Scoring**
- **Business Analytics**
- **Interactive Data Visualization**
- **Decision Support**
- **Model Governance**
- **Construction Analytics**
- **Operational Intelligence**

---

# 📚 Design Principles

### 1. Business-first modeling

The model is designed around a practical management question rather than purely optimizing a statistical metric.

### 2. Prediction-time realism

Only information available at the time of prediction should be used.

### 3. Explainability

Risk scores should be accompanied by understandable drivers.

### 4. Model governance

Limitations, leakage, validation and calibration should be explicitly documented.

### 5. Actionable analytics

The output should help management decide:

> **Where should we investigate and intervene?**

---

# 👤 Portfolio Positioning

This project is relevant to roles such as:

- Senior Data Scientist
- Senior Machine Learning Engineer
- Applied Data Scientist
- AI / ML Engineer
- Industrial Data Scientist
- Operations Research / Analytics
- Business Intelligence & Advanced Analytics
- AI Transformation
- Construction Technology / PropTech

It demonstrates an end-to-end workflow:

```text
Business Problem
      ↓
Data
      ↓
Data Quality
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Model Validation
      ↓
Explainable AI
      ↓
Risk Prediction
      ↓
Interactive Dashboard
      ↓
Business Decision Support
```

---

# ⭐ Project Philosophy

> **A good Data Science solution does not stop at predicting risk.**
>
> **It explains the risk, validates the assumptions, exposes the limitations, and connects the prediction to a business decision.**

---

