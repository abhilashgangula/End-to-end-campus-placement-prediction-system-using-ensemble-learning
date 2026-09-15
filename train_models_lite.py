"""
train_models_lite.py
====================
Lightweight model training script optimized for Render free tier (512 MB RAM).

Strategy:
  - Classification: XGBoost Classifier (best accuracy/size tradeoff) - ~3 MB
  - Regression:     XGBoost Regressor + Ridge ensemble via light VotingRegressor - ~5 MB
  - NO Random Forest (they are huge: 100 trees × 100k rows = 300+ MB on disk)
  - Compressed joblib output (compress=3)

Run this locally to regenerate the models/, then commit them via Git LFS.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
)
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    print("=" * 70)
    print("CAMPUS PLACEMENT PREDICTION — LIGHTWEIGHT MODEL TRAINING")
    print("Target: <100 MB total model size for Render free tier (512 MB RAM)")
    print("=" * 70)

    dataset_path = os.path.join("dataset", "student_placement_synthetic.csv")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("dataset", "student_placement_dataset.csv")

    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset Loaded: {df.shape[0]:,} rows and {df.shape[1]} columns.")

    if "student_id" in df.columns:
        df_clean = df.drop(columns=["student_id"])
    else:
        df_clean = df.copy()

    num_cols = [
        "cgpa", "backlogs", "coding_skills", "dsa_score", "aptitude_score",
        "communication_skills", "ml_knowledge", "system_design", "internships",
        "projects_count", "certifications", "hackathons",
        "open_source_contributions", "extracurriculars"
    ]
    cat_cols = ["branch", "college_tier"]
    all_features = num_cols + cat_cols

    X = df_clean[all_features]

    if df_clean["placement_status"].dtype == object:
        y_cls = (df_clean["placement_status"].str.lower() == "placed").astype(int)
    else:
        y_cls = df_clean["placement_status"].astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.20, random_state=42, stratify=y_cls
    )

    print("\nFitting ColumnTransformer...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"Processed feature matrix: {X_train_proc.shape[1]} encoded features.")

    # -----------------------------------------------------------------------
    # PHASE 1: Classification — XGBoost (compact, fast, accurate)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 1: CLASSIFICATION MODEL (XGBoost - Lightweight)")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    xgb_search = RandomizedSearchCV(
        estimator=XGBClassifier(
            random_state=42, n_jobs=-1, eval_metric="logloss",
            tree_method="hist",           # histogram-based — fast & memory efficient
            n_estimators=100,
            max_depth=5,
        ),
        param_distributions={
            "n_estimators": [80, 100, 120],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.05, 0.10, 0.15],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0.0, 0.1],
            "reg_lambda": [1.0, 2.0],
        },
        n_iter=8, scoring="f1", cv=cv, random_state=42, n_jobs=-1,
    )
    print("Tuning XGBoost Classifier...")
    xgb_search.fit(X_train_proc, y_train)
    best_clf_model = xgb_search.best_estimator_
    print(f"  Best params: {xgb_search.best_params_}")
    print(f"  CV F1: {xgb_search.best_score_:.4f}")

    y_pred = best_clf_model.predict(X_test_proc)
    y_prob = best_clf_model.predict_proba(X_test_proc)[:, 1]
    print(f"\n  Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Test F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  Test ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    best_cm = confusion_matrix(y_test, y_pred).tolist()
    cls_comparison = [
        {
            "Model": "XGBoost Classifier (Lite)",
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "F1-Score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
        }
    ]

    # -----------------------------------------------------------------------
    # PHASE 2: Salary Regression — lightweight XGBoost + HistGB + Ridge voting
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("PHASE 2: SALARY REGRESSION (Lightweight Ensemble)")
    print("=" * 50)

    placed_mask = (y_cls == 1) & (df_clean["salary_package_lpa"].notnull())
    X_placed = df_clean.loc[placed_mask, all_features]
    y_salary = df_clean.loc[placed_mask, "salary_package_lpa"]

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_placed, y_salary, test_size=0.20, random_state=42
    )
    X_r_train_proc = preprocessor.transform(X_r_train)
    X_r_test_proc = preprocessor.transform(X_r_test)

    # a) Ridge (tiny)
    r_ridge = Ridge(alpha=1.0)
    r_ridge.fit(X_r_train_proc, y_r_train)
    print("  Ridge trained.")

    # b) XGBoost Regressor (compact)
    xgb_r_search = RandomizedSearchCV(
        estimator=XGBRegressor(
            random_state=42, n_jobs=-1, tree_method="hist",
            n_estimators=100, max_depth=5,
        ),
        param_distributions={
            "n_estimators": [80, 100, 120],
            "max_depth": [4, 5, 6],
            "learning_rate": [0.05, 0.10, 0.15],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0.0, 0.1],
            "reg_lambda": [1.0, 2.0],
        },
        n_iter=8, scoring="r2", cv=3, random_state=42, n_jobs=-1,
    )
    print("  Tuning XGBoost Regressor...")
    xgb_r_search.fit(X_r_train_proc, y_r_train)
    r_xgb = xgb_r_search.best_estimator_
    print(f"  XGBoost Best params: {xgb_r_search.best_params_}")

    # c) HistGradientBoosting Regressor (compact)
    r_hgb = HistGradientBoostingRegressor(
        max_iter=100, max_depth=5, learning_rate=0.1,
        max_leaf_nodes=31, random_state=42
    )
    r_hgb.fit(X_r_train_proc, y_r_train)
    print("  HistGradientBoosting Regressor trained.")

    # Lightweight VotingRegressor (no Random Forest!)
    best_reg_model = VotingRegressor(
        estimators=[
            ("ridge", r_ridge),
            ("xgb", r_xgb),
            ("hgb", r_hgb),
        ],
    )
    print("  Fitting Lightweight Voting Regressor...")
    best_reg_model.fit(X_r_train_proc, y_r_train)

    r_pred = best_reg_model.predict(X_r_test_proc)
    print(f"  Test R²:  {r2_score(y_r_test, r_pred):.4f}")
    print(f"  Test MAE: {mean_absolute_error(y_r_test, r_pred):.4f} LPA")

    reg_comparison = [
        {
            "Model": "Lightweight Voting Regressor (Ridge + XGB + HistGB)",
            "R2-Score": round(float(r2_score(y_r_test, r_pred)), 4),
            "MAE (LPA)": round(float(mean_absolute_error(y_r_test, r_pred)), 4),
        }
    ]

    # -----------------------------------------------------------------------
    # Feature importances (from XGBoost classifier)
    # -----------------------------------------------------------------------
    encoded_cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    all_proc_feature_names = num_cols + encoded_cat_names

    fi = best_clf_model.feature_importances_
    sorted_idx = np.argsort(fi)[::-1]
    top_features = [
        {"feature": all_proc_feature_names[i], "importance": round(float(fi[i]), 4)}
        for i in sorted_idx[:10]
    ]

    # -----------------------------------------------------------------------
    # Feature statistics & presets (kept identical to original)
    # -----------------------------------------------------------------------
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
            "branch": "CSE", "college_tier": "Tier-1", "cgpa": 9.20, "backlogs": 0,
            "coding_skills": 9.5, "dsa_score": 9.0, "aptitude_score": 88.0,
            "communication_skills": 8.5, "ml_knowledge": 8.0, "system_design": 7.5,
            "internships": 2, "projects_count": 4, "certifications": 3,
            "hackathons": 3, "open_source_contributions": 5, "extracurriculars": 4,
        },
        "Average Student": {
            "branch": "ECE", "college_tier": "Tier-2", "cgpa": 7.20, "backlogs": 1,
            "coding_skills": 6.0, "dsa_score": 5.5, "aptitude_score": 65.0,
            "communication_skills": 6.0, "ml_knowledge": 4.5, "system_design": 4.0,
            "internships": 1, "projects_count": 2, "certifications": 1,
            "hackathons": 1, "open_source_contributions": 1, "extracurriculars": 2,
        },
        "Struggling Student": {
            "branch": "Mechanical", "college_tier": "Tier-3", "cgpa": 5.80, "backlogs": 3,
            "coding_skills": 3.5, "dsa_score": 3.0, "aptitude_score": 45.0,
            "communication_skills": 4.0, "ml_knowledge": 2.0, "system_design": 2.0,
            "internships": 0, "projects_count": 1, "certifications": 0,
            "hackathons": 0, "open_source_contributions": 0, "extracurriculars": 1,
        },
    }

    y_salary_all = df_clean.loc[placed_mask, "salary_package_lpa"]
    feature_info = {
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "all_features": all_features,
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
                "model_name": "XGBoost Classifier (Lite)",
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)),
                "confusion_matrix": best_cm,
            },
            "regression": {
                "model_name": "Lightweight Voting Regressor (Ridge + XGB + HistGB)",
                "r2_score": float(r2_score(y_r_test, r_pred)),
                "mae": float(mean_absolute_error(y_r_test, r_pred)),
            },
        },
        "dataset_summary": {
            "total_samples": int(df_clean.shape[0]),
            "placed_count": int(placed_mask.sum()),
            "not_placed_count": int((~placed_mask).sum()),
            "salary_mean": float(y_salary_all.mean()),
            "salary_min": float(y_salary_all.min()),
            "salary_max": float(y_salary_all.max()),
        },
    }

    # -----------------------------------------------------------------------
    # Save with compression (compress=3 reduces file size ~3-5x)
    # -----------------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    print("\n" + "=" * 50)
    print("SAVING COMPRESSED ARTIFACTS TO models/")
    print("=" * 50)

    joblib.dump(preprocessor, os.path.join("models", "preprocessor.pkl"), compress=3)
    joblib.dump(best_clf_model, os.path.join("models", "placement_model.pkl"), compress=3)
    joblib.dump(best_reg_model, os.path.join("models", "salary_model.pkl"), compress=3)
    joblib.dump(feature_info, os.path.join("models", "feature_info.pkl"), compress=3)

    # Report sizes
    for fname in ["preprocessor.pkl", "placement_model.pkl", "salary_model.pkl", "feature_info.pkl"]:
        fpath = os.path.join("models", fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  [OK] {fname:30s}  {size_mb:.2f} MB")

    total = sum(
        os.path.getsize(os.path.join("models", f)) for f in
        ["preprocessor.pkl", "placement_model.pkl", "salary_model.pkl", "feature_info.pkl"]
    ) / (1024 * 1024)
    print(f"\n  Total model size: {total:.2f} MB")
    print("\nDone! Now commit models/ and push to GitHub.")


if __name__ == "__main__":
    main()
