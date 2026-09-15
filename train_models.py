"""
End-to-End Campus Placement Prediction System Using Ensemble Learning
Updated Pipeline for: dataset/student_placement_synthetic.csv (100,000 Records, 16 Features)

Trains & Compares:
  Classification Models:
    1. Logistic Regression (Baseline)
    2. Decision Tree Classifier
    3. Random Forest Classifier (Bagging)
    4. XGBoost Classifier (Boosting)
    5. HistGradientBoosting Classifier (LightGBM-inspired Boosting)
    6. Deep Learning Neural Network (MLP 128-64-32)
    7. Voting Classifier Ensemble (Soft Voting of diverse models)

  Regression Models (Placed Cohort):
    1. Linear Regression (Baseline)
    2. Ridge Regression
    3. Random Forest Regressor (Bagging)
    4. XGBoost Regressor (Boosting)
    5. HistGradientBoosting Regressor (Boosting)
    6. Deep Learning Neural Network Regressor (MLP)
    7. Voting Regressor Ensemble (Weighted average ensemble)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

def main():
    print("=" * 70)
    print("CAMPUS PLACEMENT PREDICTION SYSTEM — SYNTHETIC DATASET 2026")
    print("=" * 70)

    dataset_path = os.path.join("dataset", "student_placement_synthetic.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("dataset", "student_placement_dataset.csv")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset Loaded: {df.shape[0]:,} rows and {df.shape[1]} columns.")

    # Drop non-predictive student_id if present
    if "student_id" in df.columns:
        df_clean = df.drop(columns=["student_id"])
    else:
        df_clean = df.copy()

    # Define Feature Sets for student_placement_synthetic.csv
    num_cols = [
        "cgpa",
        "backlogs",
        "coding_skills",
        "dsa_score",
        "aptitude_score",
        "communication_skills",
        "ml_knowledge",
        "system_design",
        "internships",
        "projects_count",
        "certifications",
        "hackathons",
        "open_source_contributions",
        "extracurriculars"
    ]

    cat_cols = ["branch", "college_tier"]
    all_features = num_cols + cat_cols

    X = df_clean[all_features]
    
    # Handle binary target (1 or 'Placed' -> 1, 0 or 'Not Placed' -> 0)
    if df_clean["placement_status"].dtype == object:
        y_cls = (df_clean["placement_status"].str.lower() == "placed").astype(int)
    else:
        y_cls = df_clean["placement_status"].astype(int)

    # Reusable Preprocessor Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    # Stratified 80/20 Train-Test Split (Zero Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )

    print("\nFitting ColumnTransformer on training split...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"Processed feature matrix: {X_train_proc.shape[1]} encoded features.")

    # ----------------------------------------------------
    # 1. Systematic Hyperparameter Tuning (Phase 1)
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 1: HYPERPARAMETER TUNING (RANDOMIZED SEARCH CV)")
    print("=" * 50)

    cv_3fold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # A. Tune XGBoost Classifier
    print("Tuning XGBoost Classifier...")
    xgb_param_dist = {
        "n_estimators": [100, 150, 200],
        "max_depth": [5, 6, 7],
        "learning_rate": [0.05, 0.08, 0.12],
        "reg_alpha": [0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 3.0],
        "subsample": [0.85, 1.0],
    }
    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=42, n_jobs=-1, eval_metric="logloss"),
        param_distributions=xgb_param_dist,
        n_iter=6,
        scoring="f1",
        cv=cv_3fold,
        random_state=42,
        n_jobs=-1,
    )
    xgb_search.fit(X_train_proc, y_train)
    m_xgb_tuned = xgb_search.best_estimator_
    print(f"  [XGBoost Best Params]: {xgb_search.best_params_} (CV F1: {xgb_search.best_score_:.4f})")

    # B. Tune Random Forest Classifier
    print("Tuning Random Forest Classifier...")
    rf_param_dist = {
        "n_estimators": [100, 150],
        "max_depth": [12, 16],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=rf_param_dist,
        n_iter=4,
        scoring="f1",
        cv=cv_3fold,
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_train_proc, y_train)
    m_rf_tuned = rf_search.best_estimator_
    print(f"  [Random Forest Best Params]: {rf_search.best_params_} (CV F1: {rf_search.best_score_:.4f})")

    # C. Tune HistGradientBoosting Classifier
    print("Tuning HistGradientBoosting Classifier...")
    hgb_param_dist = {
        "max_iter": [150, 200],
        "learning_rate": [0.05, 0.08, 0.1],
        "max_leaf_nodes": [20, 31, 50],
        "l2_regularization": [0.5, 1.5, 3.0],
    }
    hgb_search = RandomizedSearchCV(
        estimator=HistGradientBoostingClassifier(random_state=42),
        param_distributions=hgb_param_dist,
        n_iter=6,
        scoring="f1",
        cv=cv_3fold,
        random_state=42,
        n_jobs=-1,
    )
    hgb_search.fit(X_train_proc, y_train)
    m_hgb_tuned = hgb_search.best_estimator_
    print(f"  [HistGB Best Params]: {hgb_search.best_params_} (CV F1: {hgb_search.best_score_:.4f})")

    # D. Base Linear & Deep Learning Models
    m_l2_lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=500, random_state=42)
    m_l2_lr.fit(X_train_proc, y_train)

    m_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=50,
        early_stopping=True,
        random_state=42
    )
    m_mlp.fit(X_train_proc, y_train)

    # E. Hyper-Tuned Voting Ensemble
    m_voting_clf = VotingClassifier(
        estimators=[
            ("lr", m_l2_lr),
            ("rf", m_rf_tuned),
            ("xgb", m_xgb_tuned),
            ("hgb", m_hgb_tuned),
        ],
        voting="soft",
        n_jobs=-1,
    )
    print("Fitting Hyper-Tuned Voting Classifier Ensemble...")
    m_voting_clf.fit(X_train_proc, y_train)

    clf_models = {
        "L2 Logistic Regression (Baseline)": m_l2_lr,
        "Hyper-Tuned Random Forest": m_rf_tuned,
        "Hyper-Tuned XGBoost Classifier": m_xgb_tuned,
        "Hyper-Tuned HistGradientBoosting": m_hgb_tuned,
        "Deep Learning Neural Network (MLP)": m_mlp,
        "Hyper-Tuned Voting Ensemble": m_voting_clf,
    }

    cls_comparison = []

    for name, model in clf_models.items():
        preds = model.predict(X_test_proc)
        probs = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else preds

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        roc = roc_auc_score(y_test, probs)

        cls_comparison.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(roc), 4),
        })

    df_cls_res = pd.DataFrame(cls_comparison)
    print("\n--- Holdout Evaluation: Hyperparameter Tuned Classifiers ---")
    print(df_cls_res.to_string(index=False))

    # Select Best Production Classifier (Hyper-Tuned Voting Ensemble)
    best_clf_model = m_voting_clf
    y_pred_best = best_clf_model.predict(X_test_proc)
    y_prob_best = best_clf_model.predict_proba(X_test_proc)[:, 1]
    best_cm = confusion_matrix(y_test, y_pred_best).tolist()

    # ----------------------------------------------------
    # 2. Hyperparameter Tuning for Salary Regression (Phase 2)
    # ----------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 2: SALARY REGRESSION HYPERPARAMETER TUNING")
    print("=" * 50)

    # Filter placed students with valid non-null salary
    placed_mask = (y_cls == 1) & (df_clean["salary_package_lpa"].notnull())
    X_placed = df_clean.loc[placed_mask, all_features]
    y_salary = df_clean.loc[placed_mask, "salary_package_lpa"]

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_placed, y_salary, test_size=0.20, random_state=42
    )

    X_r_train_proc = preprocessor.transform(X_r_train)
    X_r_test_proc = preprocessor.transform(X_r_test)

    # A. Tune Ridge Regressor
    print("Tuning Ridge Regressor...")
    ridge_grid = GridSearchCV(Ridge(), param_grid={"alpha": [0.1, 0.5, 1.0, 5.0, 10.0]}, cv=3, scoring="r2", n_jobs=-1)
    ridge_grid.fit(X_r_train_proc, y_r_train)
    r_ridge_tuned = ridge_grid.best_estimator_
    print(f"  [Ridge Best Params]: {ridge_grid.best_params_} (CV R2: {ridge_grid.best_score_:.4f})")

    # B. Tune XGBoost Regressor
    print("Tuning XGBoost Regressor...")
    xgb_r_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=42, n_jobs=-1),
        param_distributions={
            "n_estimators": [100, 150, 200],
            "max_depth": [5, 6, 7],
            "learning_rate": [0.05, 0.08, 0.12],
            "reg_alpha": [0.01, 0.1, 1.0],
            "reg_lambda": [0.5, 1.0, 3.0],
        },
        n_iter=6,
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    xgb_r_search.fit(X_r_train_proc, y_r_train)
    r_xgb_tuned = xgb_r_search.best_estimator_
    print(f"  [XGBoost Regressor Best Params]: {xgb_r_search.best_params_} (CV R2: {xgb_r_search.best_score_:.4f})")

    # C. Tune HistGradientBoosting Regressor
    print("Tuning HistGradientBoosting Regressor...")
    hgb_r_search = RandomizedSearchCV(
        estimator=HistGradientBoostingRegressor(random_state=42),
        param_distributions={
            "max_iter": [150, 200],
            "learning_rate": [0.05, 0.08, 0.1],
            "max_leaf_nodes": [20, 31, 50],
            "l2_regularization": [0.5, 1.5, 3.0],
        },
        n_iter=6,
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    hgb_r_search.fit(X_r_train_proc, y_r_train)
    r_hgb_tuned = hgb_r_search.best_estimator_
    print(f"  [HistGB Regressor Best Params]: {hgb_r_search.best_params_} (CV R2: {hgb_r_search.best_score_:.4f})")

    # D. Tune Random Forest Regressor & Deep Learning MLP Regressor
    print("Tuning Random Forest Regressor...")
    rf_r_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
        param_distributions={
            "n_estimators": [100, 150],
            "max_depth": [10, 14, 18],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        n_iter=4,
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )
    rf_r_search.fit(X_r_train_proc, y_r_train)
    r_rf_tuned = rf_r_search.best_estimator_
    print(f"  [Random Forest Regressor Best Params]: {rf_r_search.best_params_} (CV R2: {rf_r_search.best_score_:.4f})")

    r_mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.01,
        batch_size=256,
        learning_rate="adaptive",
        max_iter=50,
        early_stopping=True,
        random_state=42
    )
    r_mlp.fit(X_r_train_proc, y_r_train)

    # E. Hyper-Tuned Voting Regressor Ensemble
    r_voting = VotingRegressor(
        estimators=[
            ("ridge", r_ridge_tuned),
            ("rf", r_rf_tuned),
            ("xgb", r_xgb_tuned),
            ("hgb", r_hgb_tuned),
        ],
        n_jobs=-1,
    )
    print("Fitting Hyper-Tuned Voting Regressor Ensemble...")
    r_voting.fit(X_r_train_proc, y_r_train)

    reg_models = {
        "Hyper-Tuned Ridge Regression": r_ridge_tuned,
        "Hyper-Tuned Random Forest Regressor": r_rf_tuned,
        "Hyper-Tuned XGBoost Regressor": r_xgb_tuned,
        "Hyper-Tuned HistGradientBoosting": r_hgb_tuned,
        "Deep Learning Neural Network (MLP)": r_mlp,
        "Hyper-Tuned Voting Regressor Ensemble": r_voting,
    }

    reg_comparison = []

    for name, model in reg_models.items():
        print(f"Training {name}...")
        model.fit(X_r_train_proc, y_r_train)
        preds = model.predict(X_r_test_proc)

        r2 = r2_score(y_r_test, preds)
        mae = mean_absolute_error(y_r_test, preds)
        rmse = root_mean_squared_error(y_r_test, preds)

        reg_comparison.append({
            "Model": name,
            "R2-Score": round(float(r2), 4),
            "MAE (LPA)": round(float(mae), 4),
            "RMSE (LPA)": round(float(rmse), 4),
        })

    df_reg_res = pd.DataFrame(reg_comparison)
    print("\n--- Salary Regression Performance Comparison ---")
    print(df_reg_res.to_string(index=False))

    best_reg_model = r_voting

    # ----------------------------------------------------
    # 3. Feature Importances Analysis
    # ----------------------------------------------------
    encoded_cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    all_proc_feature_names = num_cols + encoded_cat_names

    rf_importances = m_rf_tuned.feature_importances_
    sorted_idx = np.argsort(rf_importances)[::-1]
    top_features = [
        {"feature": all_proc_feature_names[i], "importance": round(float(rf_importances[i]), 4)}
        for i in sorted_idx[:10]
    ]

    print("\nTop 10 Influential Features for Placement (Random Forest):")
    for f in top_features:
        print(f"  - {f['feature']:30s}: {f['importance']*100:.2f}%")

    # ----------------------------------------------------
    # 4. Feature Statistics, Ranges & Quick Presets
    # ----------------------------------------------------
    num_stats = {}
    for col in num_cols:
        num_stats[col] = {
            "min": float(df_clean[col].min()),
            "max": float(df_clean[col].max()),
            "mean": float(df_clean[col].mean()),
            "median": float(df_clean[col].median()),
            "std": float(df_clean[col].std()),
            "placed_mean": float(df_clean.loc[placed_mask, col].mean()),
            "not_placed_mean": float(df_clean.loc[~placed_mask, col].mean()),
        }

    categorical_options = {col: sorted(df_clean[col].unique().tolist()) for col in cat_cols}

    presets = {
        "High Achiever": {
            "branch": "CSE",
            "college_tier": "Tier-1",
            "cgpa": 9.20,
            "backlogs": 0,
            "coding_skills": 9.5,
            "dsa_score": 9.0,
            "aptitude_score": 88.0,
            "communication_skills": 8.5,
            "ml_knowledge": 8.0,
            "system_design": 7.5,
            "internships": 2,
            "projects_count": 4,
            "certifications": 3,
            "hackathons": 2,
            "open_source_contributions": 1,
            "extracurriculars": 2
        },
        "Balanced Performer": {
            "branch": "ECE",
            "college_tier": "Tier-2",
            "cgpa": 7.50,
            "backlogs": 0,
            "coding_skills": 6.5,
            "dsa_score": 6.0,
            "aptitude_score": 68.0,
            "communication_skills": 6.5,
            "ml_knowledge": 4.5,
            "system_design": 4.0,
            "internships": 1,
            "projects_count": 2,
            "certifications": 2,
            "hackathons": 1,
            "open_source_contributions": 0,
            "extracurriculars": 1
        },
        "Tech Specialist": {
            "branch": "IT",
            "college_tier": "Tier-2",
            "cgpa": 7.80,
            "backlogs": 0,
            "coding_skills": 9.0,
            "dsa_score": 8.5,
            "aptitude_score": 75.0,
            "communication_skills": 6.0,
            "ml_knowledge": 7.5,
            "system_design": 7.0,
            "internships": 2,
            "projects_count": 4,
            "certifications": 2,
            "hackathons": 3,
            "open_source_contributions": 2,
            "extracurriculars": 1
        },
        "Needs Improvement": {
            "branch": "Chemical",
            "college_tier": "Tier-3",
            "cgpa": 5.40,
            "backlogs": 2,
            "coding_skills": 3.0,
            "dsa_score": 2.5,
            "aptitude_score": 42.0,
            "communication_skills": 4.0,
            "ml_knowledge": 1.5,
            "system_design": 1.0,
            "internships": 0,
            "projects_count": 1,
            "certifications": 0,
            "hackathons": 0,
            "open_source_contributions": 0,
            "extracurriculars": 0
        }
    }

    feature_info = {
        "all_features": all_features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_stats": num_stats,
        "categorical_options": categorical_options,
        "presets": presets,
        "top_features": top_features,
        "models_comparison": {
            "classification": cls_comparison,
            "regression": reg_comparison,
        },
        "best_metrics": {
            "classification": {
                "model_name": "Voting Classifier Ensemble (LR + RF + XGB + HistGB)",
                "accuracy": float(accuracy_score(y_test, y_pred_best)),
                "precision": float(precision_score(y_test, y_pred_best)),
                "recall": float(recall_score(y_test, y_pred_best)),
                "f1_score": float(f1_score(y_test, y_pred_best)),
                "roc_auc": float(roc_auc_score(y_test, y_prob_best)),
                "confusion_matrix": best_cm,
            },
            "regression": {
                "model_name": "Voting Regressor Ensemble (Ridge + RF + XGB + HistGB)",
                "r2_score": float(r2_score(y_r_test, best_reg_model.predict(X_r_test_proc))),
                "mae": float(mean_absolute_error(y_r_test, best_reg_model.predict(X_r_test_proc))),
                "rmse": float(root_mean_squared_error(y_r_test, best_reg_model.predict(X_r_test_proc))),
            },
        },
        "dataset_summary": {
            "total_samples": int(df_clean.shape[0]),
            "placed_count": int(placed_mask.sum()),
            "not_placed_count": int((~placed_mask).sum()),
            "salary_mean": float(y_salary.mean()),
            "salary_min": float(y_salary.min()),
            "salary_max": float(y_salary.max()),
        },
    }

    # ----------------------------------------------------
    # 5. Save Artifacts to models/
    # ----------------------------------------------------
    os.makedirs("models", exist_ok=True)

    joblib.dump(preprocessor, os.path.join("models", "preprocessor.pkl"))
    joblib.dump(best_clf_model, os.path.join("models", "placement_model.pkl"))
    joblib.dump(best_reg_model, os.path.join("models", "salary_model.pkl"))
    joblib.dump(feature_info, os.path.join("models", "feature_info.pkl"))

    print("\n" + "=" * 50)
    print("SAVED SERIALIZED ARTIFACTS TO models/:")
    print("  [OK] models/preprocessor.pkl")
    print("  [OK] models/placement_model.pkl  (VotingClassifier Ensemble)")
    print("  [OK] models/salary_model.pkl     (VotingRegressor Ensemble)")
    print("  [OK] models/feature_info.pkl      (Full metadata & comparisons)")
    print("=" * 50)
    print("Pipeline execution completed successfully!\n")

if __name__ == "__main__":
    main()
