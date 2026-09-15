# End-to-End Campus Placement Prediction System Using Ensemble Learning

> A full-stack machine learning web application that predicts campus recruitment
> outcomes and estimated salary packages for engineering students using a Soft
> Voting Ensemble of three heterogeneous classifiers, served through a Flask REST
> API with an interactive dark-mode dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset Description](#3-dataset-description)
4. [ML Methodology](#4-ml-methodology)
5. [Model Comparison](#5-model-comparison)
6. [Ensemble Methodology](#6-ensemble-methodology)
7. [Installation](#7-installation)
8. [How to Train the Models](#8-how-to-train-the-models)
9. [How to Run the Flask Application](#9-how-to-run-the-flask-application)
10. [Example Prediction](#10-example-prediction)
11. [System Limitations](#11-system-limitations)
12. [Future Enhancements](#12-future-enhancements)

---

## 1. Project Overview

This project demonstrates a **complete, production-style machine learning pipeline**
applied to the campus placement prediction domain. It covers every stage from raw
CSV data to a live, interactive web prediction tool.

### Key Capabilities

| Capability | Detail |
|---|---|
| Placement Classification | Binary prediction: Placed / Not Placed |
| Salary Regression | Predicted salary package in LPA (Lakhs Per Annum) |
| Explainable AI | Permutation-importance feature rankings surfaced in the UI |
| Input Validation | Out-of-range clamping, unknown-category defaulting, null handling |
| REST API | JSON endpoint at /api/predict for programmatic access |
| Interactive Dashboard | Dark-mode result card with benchmark bars and placement signals |

### Tech Stack

```
ML Pipeline  : scikit-learn  (Preprocessing + Ensemble models)
Serialization: joblib        (Artifact persistence)
Backend      : Python 3.11 + Flask 3.x
Frontend     : Vanilla HTML5 / CSS3 / JavaScript (no frameworks)
Data         : pandas + numpy + scipy
Visualization: matplotlib + seaborn  (notebook only)
```

---

## 2. Project Structure

```
Campus_Placement_Prediction/
|
|-- dataset/
|   L-- student_placement_dataset.csv      # 100,000 synthetic student records
|
|-- notebook/
|   L-- placement_prediction.ipynb         # End-to-end training notebook (7 stages)
|
|-- models/
|   |-- placement_model.pkl                # Trained VotingClassifier (44 MB)
|   |-- salary_model.pkl                   # Trained VotingRegressor (84 MB)
|   |-- preprocessor.pkl                   # ColumnTransformer (StandardScaler + OHE)
|   |-- feature_info.pkl                   # Feature schema, presets, top-features
|   |-- model_metadata.pkl                 # Model evaluation metrics
|   L-- feature_importance.pkl             # Permutation importance rankings
|
|-- app/
|   |-- app.py                             # Flask backend & inference engine
|   |-- templates/
|   |   |-- index.html                     # Landing page
|   |   |-- prediction.html                # Form + interactive result dashboard
|   |   |-- result.html                    # Standalone result redirect page
|   |   L-- about.html                     # Model docs, XAI, methodology
|   |
|   L-- static/
|       |-- css/
|       |   L-- style.css                  # Complete dark-mode design system
|       L-- js/
|           L-- script.js                  # AJAX form handler + UI logic
|
|-- requirements.txt                       # Pinned Python dependencies
L-- README.md                              # This file
```

---

## 3. Dataset Description

### Source

The dataset is a **realistic synthetic dataset** generated to mimic real-world
campus placement patterns. It does not contain personally identifiable information.

**File:** `dataset/student_placement_dataset.csv`

### Statistics

| Attribute | Value |
|---|---|
| Total samples | 100,000 student records |
| Placed students | 54,459 (54.5%) |
| Not placed students | 45,541 (45.5%) |
| Features | 23 raw input features + 2 target variables |
| Salary range | Rs 7.11 LPA - Rs 20.44 LPA |
| Salary mean | Rs 13.32 LPA |

### Feature Schema

| # | Feature | Type | Range | Description |
|---|---|---|---|---|
| 1 | age | Numeric | 18-30 | Student age |
| 2 | cgpa | Numeric | 0.0-10.0 | Cumulative Grade Point Average |
| 3 | internships_count | Numeric | 0-10 | Number of internships completed |
| 4 | projects_count | Numeric | 0-15 | Academic/personal projects |
| 5 | certifications_count | Numeric | 0-10 | Online/offline certifications |
| 6 | coding_skill_score | Numeric | 0-100 | Assessed coding ability |
| 7 | aptitude_score | Numeric | 0-100 | Quantitative aptitude |
| 8 | communication_skill_score | Numeric | 0-100 | Communication ability |
| 9 | logical_reasoning_score | Numeric | 0-100 | Logical reasoning |
| 10 | hackathons_participated | Numeric | 0-10 | Hackathon participation count |
| 11 | github_repos | Numeric | 0-30 | Public GitHub repositories |
| 12 | linkedin_connections | Numeric | 0-2000 | LinkedIn network size |
| 13 | mock_interview_score | Numeric | 0-100 | Mock interview performance |
| 14 | attendance_percentage | Numeric | 0-100 | Academic attendance |
| 15 | backlogs | Numeric | 0-10 | Active academic backlogs |
| 16 | extracurricular_score | Numeric | 0-100 | Extracurricular activities |
| 17 | leadership_score | Numeric | 0-100 | Leadership assessment |
| 18 | sleep_hours | Numeric | 3-12 | Average daily sleep hours |
| 19 | study_hours_per_day | Numeric | 0-12 | Daily study hours |
| 20 | gender | Categorical | Female/Male | Gender |
| 21 | branch | Categorical | CSE/ECE/EEE/IT/Civil/Mechanical | Engineering branch |
| 22 | college_tier | Categorical | Tier 1/Tier 2/Tier 3 | Institution ranking |
| 23 | volunteer_experience | Categorical | Yes/No | Volunteering background |

### Target Variables

| Variable | Type | Description |
|---|---|---|
| placement_status | Binary | 1 = Placed, 0 = Not Placed |
| salary_package_lpa | Continuous | Salary in Lakhs Per Annum |

---

## 4. ML Methodology

The training pipeline is implemented in `notebook/placement_prediction.ipynb`,
structured across **7 stages**:

### Stage 1 -- Exploratory Data Analysis (EDA)

- Distribution plots for all 23 features
- Placement rate broken down by branch, college tier, and gender
- Correlation heatmap to identify feature relationships
- Outlier detection using IQR method

### Stage 2 -- Data Preprocessing

**No data leakage:** The `train_test_split` (80/20, stratified by `placement_status`,
`random_state=42`) is performed **before** any fit/transform operation.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Preprocessor fitted on X_train only, applied to X_test
preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)
```

**Preprocessor (ColumnTransformer):**

| Step | Columns | Transformer |
|---|---|---|
| Numerical | 19 numeric features | StandardScaler (zero mean, unit variance) |
| Categorical | 4 categorical features | OneHotEncoder (handle_unknown='ignore') |

### Stage 3 -- Feature Engineering

Four domain-specific composite indicators are derived after train/test split
to prevent leakage:

| Composite Feature | Formula |
|---|---|
| overall_skill_score | 0.35*coding + 0.25*aptitude + 0.20*logical + 0.20*communication |
| academic_strength_score | (cgpa*10)*0.70 + attendance*0.30 - backlogs*12 |
| professional_readiness_score | 0.45*mock_interview + 0.35*communication + 0.20*leadership |
| experience_score | internships*3 + projects*2 + hackathons*1.5 + certs + log1p(github) |

### Stage 4 -- Individual Model Training (Classification)

Six classifiers trained independently:

1. Logistic Regression (L2, C=1.0)
2. Decision Tree Classifier (max_depth=10)
3. Random Forest Classifier (n_estimators=200)
4. Gradient Boosting Classifier (n_estimators=100)
5. HistGradientBoosting Classifier
6. Extra Trees Classifier (n_estimators=200)

### Stage 5 -- Ensemble Assembly

The three best-performing individual models are combined into a Soft Voting Ensemble.

### Stage 6 -- Regression (Salary Prediction)

Six regressors trained on the same 80/20 split:
Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, HistGBR.
Final model: Soft Voting Regressor (Ridge + RF + HistGBR).

### Stage 7 -- Explainability (XAI)

**Permutation Importance** computed on the held-out test set:

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    placement_model, X_test_proc, y_test,
    n_repeats=10, random_state=42
)
```

> Note: Permutation importance identifies predictive associations, not causal effects.

---

## 5. Model Comparison

### Classification Results (Test Set, 20,000 samples)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.555 | ~0.562 | ~0.790 | ~0.657 | ~0.568 |
| Decision Tree | ~0.532 | ~0.541 | ~0.780 | ~0.639 | ~0.541 |
| Random Forest | ~0.563 | ~0.572 | ~0.795 | ~0.666 | ~0.575 |
| Gradient Boosting | ~0.558 | ~0.567 | ~0.792 | ~0.661 | ~0.570 |
| HistGradientBoosting | ~0.561 | ~0.570 | ~0.794 | ~0.664 | ~0.572 |
| Extra Trees | ~0.559 | ~0.568 | ~0.793 | ~0.662 | ~0.570 |
| **Soft Voting Ensemble** | **0.5695** | **0.5748** | **0.8046** | **0.6706** | **0.5816** |

> Note: Moderate accuracy (~57%) reflects the inherent difficulty of predicting
> placement from synthetic data where many real-world factors are absent.
> High recall (80%) means the model successfully identifies most placed students.

### Regression Results (Salary Prediction)

| Model | R-squared | MAE (LPA) | RMSE (LPA) |
|---|---|---|---|
| Linear Regression | ~0.580 | ~0.830 | ~1.030 |
| Ridge Regression | ~0.582 | ~0.825 | ~1.025 |
| Random Forest | ~0.598 | ~0.808 | ~1.010 |
| HistGradientBoosting | ~0.600 | ~0.803 | ~1.006 |
| **Soft Voting Regressor** | **0.6033** | **0.8005** | **1.0025** |

### Top 10 Predictive Features (Permutation Importance)

| Rank | Feature | Importance |
|---|---|---|
| 1 | coding_skill_score | 0.0772 |
| 2 | mock_interview_score | 0.0743 |
| 3 | logical_reasoning_score | 0.0697 |
| 4 | aptitude_score | 0.0680 |
| 5 | communication_skill_score | 0.0660 |
| 6 | leadership_score | 0.0629 |
| 7 | extracurricular_score | 0.0605 |
| 8 | cgpa | 0.0588 |
| 9 | linkedin_connections | 0.0575 |
| 10 | attendance_percentage | 0.0561 |

---

## 6. Ensemble Methodology

### Why Ensemble Learning?

Individual classifiers each capture different aspects of the decision boundary.
Soft voting reduces variance and bias, yielding more robust predictions.

### Soft Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(random_state=42)),
    ],
    voting='soft',
    weights=[1, 1.5, 1.5]
)
```

Prediction formula:

```
P(Placed) = (w_lr * P_lr + w_rf * P_rf + w_hgb * P_hgb) / (w_lr + w_rf + w_hgb)
```

### Inference Pipeline

```
Raw Input (23 features)
    |
    v
validate_and_parse_input()  <-- type coercion, clamping, category defaulting
    |
    v
pd.DataFrame (23 cols, correct dtype, exact column order)
    |
    v
preprocessor.transform()    <-- StandardScaler + OneHotEncoder --> 28 features
    |
    |----> placement_model.predict_proba()  --> [P(NotPlaced), P(Placed)]
    |
    L----> salary_model.predict()           --> salary_lpa (float)
```

> IMPORTANT: The preprocessor must be applied before calling either model.
> Both are saved as separate artifacts and chained explicitly in app.py.

---

## 7. Installation

### Prerequisites

- Python 3.11 or 3.12 (recommended)
- pip 23+
- 4 GB free RAM (model loading requires ~300 MB resident)

### Steps

```bash
# 1. Navigate to project directory
cd "Campus_Placement_Prediction"

# 2. Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import flask, sklearn, pandas, numpy, joblib; print('All dependencies OK')"
```

---

## 8. How to Train the Models

> Pre-trained models are already in models/. Run training only to retrain
> from scratch or experiment with hyperparameters.

### Option A -- Jupyter Notebook (Recommended)

```bash
pip install notebook ipykernel
jupyter notebook notebook/placement_prediction.ipynb
```

Run all cells in order (~15-25 minutes on a modern laptop).
The notebook automatically saves artifacts to models/.

### Option B -- Training Script

```bash
python train_models.py
```

### Training Configuration

| Parameter | Value |
|---|---|
| Random seed | 42 (all models) |
| Train/test split | 80% / 20% |
| Stratification | By placement_status |
| Cross-validation | 5-fold StratifiedKFold |
| Reproducibility | random_state=42 on all stochastic estimators |

---

## 9. How to Run the Flask Application

### Start the Server

```bash
cd app
python app.py
```

Expected output:
```
[OK] Flask Backend: All serialized ML ensemble models loaded successfully.
 * Running on http://127.0.0.1:5000
```

### Application Routes

| Route | Method | Description |
|---|---|---|
| / | GET | Landing page with system overview |
| /predict | GET | Student assessment form |
| /predict | POST | Form submission + inline prediction result |
| /api/predict | POST | JSON REST API for AJAX/programmatic access |
| /api/presets | GET | Returns 4 student profile presets as JSON |
| /about | GET | Model documentation and XAI rankings |

### Production Deployment

```bash
# Linux / macOS
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 app:app

# Windows
pip install waitress
waitress-serve --port=5000 app:app
```

---

## 10. Example Prediction

### REST API Request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 21, "cgpa": 9.2, "branch": "CSE", "college_tier": "Tier 1",
    "gender": "Female", "internships_count": 3, "projects_count": 6,
    "certifications_count": 5, "coding_skill_score": 92.0,
    "aptitude_score": 90.0, "communication_skill_score": 88.0,
    "logical_reasoning_score": 87.0, "hackathons_participated": 4,
    "github_repos": 14, "linkedin_connections": 750,
    "mock_interview_score": 90.0, "attendance_percentage": 96.0,
    "backlogs": 0, "extracurricular_score": 80.0, "leadership_score": 78.0,
    "sleep_hours": 7.0, "study_hours_per_day": 6.0,
    "volunteer_experience": "Yes"
  }'
```

### Expected Response

```json
{
  "success": true,
  "prediction": {
    "status": "Placed",
    "is_placed": true,
    "probability": 74.02,
    "risk_level": "High Confidence Selection",
    "risk_badge": "success",
    "salary_lpa": 16.25,
    "salary_range": "14.30 - 18.20 LPA",
    "benchmarks": [
      {"label": "CGPA", "user_value": 9.2, "placed_avg": 7.9, "user_pct": 92.0}
    ],
    "factors": [
      {"name": "Coding Score 92/100", "impact": "positive",
       "badge": "High Asset", "desc": "Well above screening threshold."}
    ],
    "recommendations": [
      {"type": "success", "title": "Target FAANG-tier roles",
       "description": "Your profile is competitive for top-tier campus offers."}
    ]
  }
}
```

### Prediction Results for Key Profiles

| Profile | Status | Probability | Salary |
|---|---|---|---|
| High-performing (CGPA 9.2, coding 92) | Placed | 74.0% | Rs 16.25 LPA |
| Average student (CGPA 7.8, coding 72) | Placed | 59.5% | Rs 13.35 LPA |
| Low-performing (CGPA 5.8, coding 38) | Not Placed | 34.3% | Rs 9.82 LPA |
| 4 backlogs (CGPA 6.2) | Not Placed | 36.8% | Rs 10.91 LPA |
| Strong tech / weak academics | Placed | 61.6% | Rs 13.04 LPA |
| Strong academics / weak tech | Not Placed | 40.8% | Rs 12.12 LPA |
| Boundary minimum | Not Placed | 26.3% | Rs 8.61 LPA |
| Boundary maximum | Placed | 76.1% | Rs 17.97 LPA |

---

## 11. System Limitations

### Dataset Limitations

1. **Synthetic data**: The dataset was algorithmically generated. Predictions will
   not precisely match real placement rates at any specific institution.

2. **No temporal dimension**: The model does not account for hiring cycles,
   economic conditions, or year-on-year trends.

3. **No company-side features**: Recruiter preferences, job roles, and company
   reputation are absent from the feature set.

4. **Geographic blindness**: Institution location and relocation willingness
   are not captured.

5. **Class distribution**: The near-balanced 54.5%/45.5% split may not match
   every institution's actual placement ratio.

### Model Limitations

6. **Moderate accuracy (57%)**: Placement depends on many real-world factors
   that no feature vector can fully capture.

7. **Correlation, not causation**: Feature importance scores indicate predictive
   associations. High coding_skill_score correlates with placement but does
   not *cause* it.

8. **Salary uncertainty**: The model carries +/-1.0 LPA RMSE. Treat salary
   outputs as indicative ranges, not precise forecasts.

9. **Concept drift**: If deployed after the data generation period, real-world
   feature distributions may shift, degrading accuracy without retraining.

10. **Input distribution**: Students with feature combinations far outside the
    training distribution may receive unreliable predictions.

---

## 12. Future Enhancements

### ML Improvements

- [ ] Real-world dataset integration from actual placement records
- [ ] SHAP values for per-prediction local explanations
- [ ] Calibrated probabilities (Platt scaling / isotonic regression)
- [ ] Automated retraining with MLflow experiment tracking
- [ ] LightGBM / XGBoost as additional ensemble members
- [ ] Optuna hyperparameter tuning with cross-validation leaderboard

### Application Improvements

- [ ] User authentication with student profile saving
- [ ] Batch CSV upload for entire cohort prediction
- [ ] PDF report generation (downloadable assessment card)
- [ ] Side-by-side profile comparison mode
- [ ] Progressive Web App (PWA) for offline use

### Infrastructure

- [ ] Docker containerization for one-command deployment
- [ ] GitHub Actions CI to re-run test suite on push
- [ ] Redis caching for repeated prediction requests
- [ ] Prometheus metrics for production monitoring

---

## Reproducibility Checklist

```
[ ] Python 3.11+ installed
[ ] pip install -r requirements.txt
[ ] dataset/student_placement_dataset.csv present (21.5 MB)
[ ] Run notebook/placement_prediction.ipynb (all cells, random_state=42)
[ ] Verify models/ directory has 6 .pkl files
[ ] cd app && python app.py
[ ] Open http://127.0.0.1:5000
[ ] Run python test_e2e.py to verify 82/82 tests pass
```

---

## Final Evaluation Summary

| Metric | Classification | Regression |
|---|---|---|
| Best Model | Soft Voting Ensemble | Soft Voting Ensemble |
| Accuracy / R-squared | 56.95% | 0.6033 |
| Precision | 57.48% | -- |
| Recall | 80.46% | -- |
| F1-Score | 67.06% | -- |
| ROC-AUC | 0.5816 | -- |
| MAE | -- | 0.80 LPA |
| RMSE | -- | 1.00 LPA |

---

*Developed as a college mini-project demonstrating end-to-end machine learning
deployment with Python, scikit-learn, and Flask.*
