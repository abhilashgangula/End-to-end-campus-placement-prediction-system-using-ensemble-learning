"""
==============================================================================
🎓 CAMPUS PLACEMENT PREDICTION SYSTEM USING ENSEMBLE LEARNING
Flask Backend & Inference Engine
==============================================================================
Technology Stack:
  - Python 3.9+
  - Flask (Web Framework & Routing)
  - Scikit-Learn & Joblib (Model Deserialization & Preprocessing Pipelines)
  - Pandas & NumPy (Data Structuring & Matrix Manipulations)

Routes:
  - GET  /              : Landing page with ensemble highlights & stats
  - GET  /predict       : Student assessment form & interactive dashboard
  - POST /predict       : Form submission prediction endpoint
  - POST /api/predict   : REST API endpoint for JSON/AJAX client requests
  - GET  /api/presets   : REST API endpoint for quick student profile presets
  - GET  /about         : Model comparison matrices, XAI rankings & viva notes
==============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from sklearn.base import BaseEstimator, TransformerMixin

# ----------------------------------------------------------------------------
# 1. Custom Feature Engineering Transformer Definition
# ----------------------------------------------------------------------------
class PlacementFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer matching the training pipeline.
    Constructs 4 domain-specific composite indicators from raw inputs:
      1. overall_skill_score
      2. academic_strength_score
      3. professional_readiness_score
      4. experience_score
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # 1. Overall Skill Score (0 - 100)
        X_out['overall_skill_score'] = (
            0.35 * X_out['coding_skill_score'] +
            0.25 * X_out['aptitude_score'] +
            0.20 * X_out['logical_reasoning_score'] +
            0.20 * X_out['communication_skill_score']
        )
        
        # 2. Academic Strength Score (Penalized by backlogs)
        X_out['academic_strength_score'] = (
            (X_out['cgpa'] * 10.0) * 0.70 +
            (X_out['attendance_percentage'] * 0.30) -
            (X_out['backlogs'] * 12.0)
        )
        
        # 3. Professional Readiness Score (0 - 100)
        X_out['professional_readiness_score'] = (
            0.45 * X_out['mock_interview_score'] +
            0.35 * X_out['communication_skill_score'] +
            0.20 * X_out['leadership_score']
        )
        
        # 4. Experience & Portfolio Score
        X_out['experience_score'] = (
            X_out['internships_count'] * 3.0 +
            X_out['projects_count'] * 2.0 +
            X_out['hackathons_participated'] * 1.5 +
            X_out['certifications_count'] * 1.0 +
            np.log1p(X_out['github_repos'])
        )
        
        return X_out

# ----------------------------------------------------------------------------
# 2. Flask Application Initialization & Model Loading
# ----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = "placement_ensemble_secret_key_2026"

# Define base file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Global model containers
preprocessor = None
placement_model = None
salary_model = None
feature_info = {}

def load_ml_artifacts():
    """
    Safely loads serialized models and metadata from models/ directory.
    Handles missing file exceptions gracefully to prevent application crash.
    """
    global preprocessor, placement_model, salary_model, feature_info
    try:
        preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
        placement_path = os.path.join(MODELS_DIR, "placement_model.pkl")
        salary_path = os.path.join(MODELS_DIR, "salary_model.pkl")
        metadata_path = os.path.join(MODELS_DIR, "feature_info.pkl")

        if not os.path.exists(metadata_path):
            metadata_path = os.path.join(MODELS_DIR, "model_metadata.pkl")

        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
        if os.path.exists(placement_path):
            placement_model = joblib.load(placement_path)
        if os.path.exists(salary_path):
            salary_model = joblib.load(salary_path)
        if os.path.exists(metadata_path):
            feature_info = joblib.load(metadata_path)

        print("[OK] Flask Backend: All serialized ML ensemble models loaded successfully.")
    except Exception as err:
        print(f"[ERROR] Failed to load ML artifacts: {err}")

# Load models on server boot
load_ml_artifacts()

# ----------------------------------------------------------------------------
# 3. Input Validation & Data Structuring Logic
# ----------------------------------------------------------------------------
RAW_NUMERICAL_COLS = [
    'cgpa', 'backlogs', 'coding_skills', 'dsa_score', 'aptitude_score',
    'communication_skills', 'ml_knowledge', 'system_design', 'internships',
    'projects_count', 'certifications', 'hackathons', 'open_source_contributions',
    'extracurriculars'
]

RAW_CATEGORICAL_COLS = ['branch', 'college_tier']
ALL_RAW_FEATURES = RAW_NUMERICAL_COLS + RAW_CATEGORICAL_COLS

FEATURE_VALIDATION_RULES = {
    'cgpa': (4.0, 10.0, 7.50),
    'backlogs': (0, 10, 0),
    'coding_skills': (1.0, 10.0, 6.0),
    'dsa_score': (1.0, 10.0, 5.5),
    'aptitude_score': (20.0, 100.0, 65.0),
    'communication_skills': (1.0, 10.0, 6.0),
    'ml_knowledge': (0.0, 10.0, 4.5),
    'system_design': (0.0, 10.0, 4.0),
    'internships': (0, 10, 1),
    'projects_count': (0, 10, 2),
    'certifications': (0, 10, 1),
    'hackathons': (0, 10, 1),
    'open_source_contributions': (0, 5, 0),
    'extracurriculars': (0, 5, 1),
}

VALID_CATEGORIES = {
    'branch': ['CSE', 'IT', 'ECE', 'EE', 'ME', 'CE', 'Chemical'],
    'college_tier': ['Tier-1', 'Tier-2', 'Tier-3']
}

def validate_and_parse_input(data_dict):
    """
    Sanitizes, validates bounds, and casts incoming client request parameters.
    Prevents server crashes caused by malformed or missing form input.
    """
    parsed = {}
    validation_warnings = []

    # Validate numerical fields
    for col in RAW_NUMERICAL_COLS:
        min_val, max_val, default_val = FEATURE_VALIDATION_RULES.get(col, (0, 100, 0))
        raw_val = data_dict.get(col)
        
        if raw_val is None or str(raw_val).strip() == "":
            parsed[col] = float(default_val)
        else:
            try:
                val = float(raw_val)
                # Clamp within safe physical boundaries
                if val < min_val:
                    val = min_val
                    validation_warnings.append(f"{col} clamped to minimum {min_val}")
                elif val > max_val:
                    val = max_val
                    validation_warnings.append(f"{col} clamped to maximum {max_val}")
                parsed[col] = val
            except (ValueError, TypeError):
                parsed[col] = float(default_val)
                validation_warnings.append(f"Invalid input for {col}, used default {default_val}")

    # Validate categorical fields
    for col in RAW_CATEGORICAL_COLS:
        valid_opts = VALID_CATEGORIES.get(col, [])
        raw_val = str(data_dict.get(col, "")).strip()
        
        if raw_val in valid_opts:
            parsed[col] = raw_val
        elif valid_opts:
            parsed[col] = valid_opts[0]
            validation_warnings.append(f"Invalid category for {col}, defaulted to {valid_opts[0]}")
        else:
            parsed[col] = "Unknown"

    return parsed, validation_warnings

# ----------------------------------------------------------------------------
# 4. Explainable AI & Recommendation Helpers
# ----------------------------------------------------------------------------
def generate_explainable_factors(student_data):
    """
    Extracts individual positive and negative drivers influencing candidate outcome.
    """
    factors = []
    
    # Backlogs Check
    backlogs = int(student_data.get("backlogs", 0))
    if backlogs == 0:
        factors.append({
            "name": "Zero Active Backlogs",
            "impact": "positive",
            "badge": "High Asset",
            "desc": "Clean academic record meeting strict corporate eligibility thresholds."
        })
    else:
        factors.append({
            "name": f"{backlogs} Active Backlog(s)",
            "impact": "negative",
            "badge": "Severe Penalty",
            "desc": f"Pending arrears filter out candidates in >70% of preliminary screening drives."
        })

    # Coding & DSA Skills
    coding = float(student_data.get("coding_skills", 0))
    dsa = float(student_data.get("dsa_score", 0))
    if coding >= 8.0:
        factors.append({
            "name": f"High Coding Skills ({coding:.1f}/10)",
            "impact": "positive",
            "badge": "Core Strength",
            "desc": "Top tier for online assessments (OA) and technical coding rounds."
        })
    elif coding < 5.0:
        factors.append({
            "name": f"Low Coding Skills ({coding:.1f}/10)",
            "impact": "negative",
            "badge": "Skill Deficit",
            "desc": "Below typical competitive cutoff score for software development profiles."
        })

    if dsa >= 8.0:
        factors.append({
            "name": f"Strong DSA Foundations ({dsa:.1f}/10)",
            "impact": "positive",
            "badge": "Algorithm Asset",
            "desc": "High proficiency in algorithmic problem solving and time complexity optimization."
        })

    # Internships Check
    internships = int(student_data.get("internships", 0))
    if internships >= 2:
        factors.append({
            "name": f"{internships} Industrial Internships",
            "impact": "positive",
            "badge": "Practical Edge",
            "desc": "Demonstrates hands-on industry experience and workplace readiness."
        })
    elif internships == 0:
        factors.append({
            "name": "Zero Internships",
            "impact": "negative",
            "badge": "Missing Factor",
            "desc": "Lack of corporate exposure reduces shortlist odds during resume screening."
        })

    # CGPA Check
    cgpa = float(student_data.get("cgpa", 0))
    if cgpa >= 8.5:
        factors.append({
            "name": f"Distinction CGPA ({cgpa:.2f}/10)",
            "impact": "positive",
            "badge": "Academic Star",
            "desc": "Unlocks dream company tier eligibility and higher salary slabs."
        })
    elif cgpa < 6.5:
        factors.append({
            "name": f"Below Average CGPA ({cgpa:.2f}/10)",
            "impact": "negative",
            "badge": "Filter Risk",
            "desc": "At risk of disqualification on minimum academic threshold filters (< 6.5 CGPA)."
        })

    return factors

def generate_growth_recommendations(student_data, prob):
    """
    Generates tailored, actionable career guidance based on student inputs.
    """
    recs = []
    
    # Backlog Priority
    backlogs = int(student_data.get("backlogs", 0))
    if backlogs > 0:
        recs.append({
            "type": "danger",
            "title": "Clear Active Backlogs Urgently",
            "description": f"You have {backlogs} active backlog(s). Clearing all subject arrears must be your highest priority before placement drives commence."
        })

    # Coding & DSA Skill
    coding = float(student_data.get("coding_skills", 0))
    dsa = float(student_data.get("dsa_score", 0))
    if coding < 7.0 or dsa < 7.0:
        recs.append({
            "type": "warning",
            "title": "Sharpen Data Structures & Problem Solving",
            "description": f"Current coding rating is {coding:.1f}/10 and DSA is {dsa:.1f}/10. Practice standard DSA problems on LeetCode/HackerRank to clear technical interviews."
        })
    else:
        recs.append({
            "type": "success",
            "title": "Strong Technical & DSA Foundations",
            "description": f"Your coding rating of {coding:.1f}/10 is highly competitive. Focus on system design and scalable clean code architecture."
        })

    # Internships & Projects
    internships = int(student_data.get("internships", 0))
    projects = int(student_data.get("projects_count", 0))
    if internships < 1:
        recs.append({
            "type": "warning",
            "title": "Secure at Least 1 Industrial Internship",
            "description": "Candidates with at least 1 internship demonstrate practical readiness and achieve significantly higher interview conversion rates."
        })
    if projects < 3:
        recs.append({
            "type": "info",
            "title": "Expand Project Portfolio & GitHub Presence",
            "description": f"Currently you have {projects} project(s). Build and deploy 2-3 end-to-end full-stack or ML projects with documented READMEs."
        })

    return recs

def compute_peer_benchmarks(student_data):
    """
    Computes comparative benchmarks against placed student averages.
    """
    stats = feature_info.get("num_stats", {})
    key_metrics = [
        ("CGPA", "cgpa", 10.0),
        ("Coding Skills", "coding_skills", 10.0),
        ("DSA Score", "dsa_score", 10.0),
        ("Aptitude Score", "aptitude_score", 100.0),
        ("Communication", "communication_skills", 10.0),
        ("ML Knowledge", "ml_knowledge", 10.0),
        ("System Design", "system_design", 10.0),
    ]

    benchmarks = []
    for label, key, max_val in key_metrics:
        user_val = float(student_data.get(key, 0.0))
        placed_avg = float(stats.get(key, {}).get("placed_mean", 7.0))
        benchmarks.append({
            "label": label,
            "user_value": round(user_val, 2),
            "placed_avg": round(placed_avg, 2),
            "user_pct": round(min(100.0, max(0.0, (user_val / max_val) * 100)), 1),
            "placed_pct": round(min(100.0, max(0.0, (placed_avg / max_val) * 100)), 1),
        })
    return benchmarks



# ----------------------------------------------------------------------------
# 5. ML Inference Engine Execution
# ----------------------------------------------------------------------------
def execute_inference(student_data):
    """
    Executes end-to-end inference for Placement Classification & Salary Regression.
    Handles pipeline execution and preprocessor transformations reliably.
    """
    if placement_model is None or salary_model is None:
        raise RuntimeError("ML Models are not loaded on server. Please verify models/ directory.")

    # Convert sanitized dictionary to single-row DataFrame
    df_input = pd.DataFrame([student_data])[ALL_RAW_FEATURES]

    # Preprocessing & Model Evaluation
    try:
        # Check if placement_model is a standalone estimator or full pipeline
        if hasattr(placement_model, "named_steps"):
            # Full Pipeline: transforms raw input internally
            prob = float(placement_model.predict_proba(df_input)[0, 1])
            pred_class = int(placement_model.predict(df_input)[0])
        else:
            # Standalone Estimator: requires preprocessor transformation
            if preprocessor is None:
                raise RuntimeError("Preprocessor artifact is missing.")
            X_proc = preprocessor.transform(df_input)
            prob = float(placement_model.predict_proba(X_proc)[0, 1])
            pred_class = int(placement_model.predict(X_proc)[0])

        is_placed = bool(prob >= 0.50 or pred_class == 1)
        status_label = "Placed" if is_placed else "Not Placed"

        # Salary Regression Execution
        if hasattr(salary_model, "named_steps"):
            pred_salary = float(salary_model.predict(df_input)[0])
        else:
            if preprocessor is None:
                raise RuntimeError("Preprocessor artifact is missing.")
            X_proc = preprocessor.transform(df_input)
            pred_salary = float(salary_model.predict(X_proc)[0])

        pred_salary = max(3.5, round(pred_salary, 2))
        salary_min = max(3.0, round(pred_salary * 0.88, 2))
        salary_max = round(pred_salary * 1.15, 2)

    except Exception as inference_err:
        print(f"[ERROR] Inference execution failed: {inference_err}")
        raise inference_err

    # Selection Confidence & Risk Tier
    if prob >= 0.70:
        risk_level = "High Confidence Selection"
        risk_badge = "success"
    elif prob >= 0.45:
        risk_level = "Moderate Probability (Borderline)"
        risk_badge = "warning"
    else:
        risk_level = "High Risk (Action Needed)"
        risk_badge = "danger"

    # Explainable Factors & Tailored Recommendations
    factors = generate_explainable_factors(student_data)
    recommendations = generate_growth_recommendations(student_data, prob)
    benchmarks = compute_peer_benchmarks(student_data)

    return {
        "status": status_label,
        "is_placed": is_placed,
        "probability": round(prob * 100, 2),
        "risk_level": risk_level,
        "risk_badge": risk_badge,
        "salary_lpa": pred_salary,
        "salary_range": f"{salary_min} - {salary_max} LPA",
        "salary_min": salary_min,
        "salary_max": salary_max,
        "factors": factors,
        "recommendations": recommendations,
        "benchmarks": benchmarks,
    }

# ----------------------------------------------------------------------------
# 6. Flask Web Application Routes
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    """
    Home Landing Page: Displays project overview, key metrics & ensemble highlights.
    """
    metrics = feature_info.get("best_metrics", {})
    summary = feature_info.get("dataset_summary", {})
    models_comp = feature_info.get("models_comparison", {})
    return render_template(
        "index.html", 
        metrics=metrics, 
        summary=summary, 
        models_comp=models_comp
    )

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Interactive Assessment Portal:
      - GET: Renders input form with default presets
      - POST: Validates input, runs inference, and displays results dashboard
    """
    presets = feature_info.get("presets", {})
    cat_options = feature_info.get("categorical_options", VALID_CATEGORIES)
    default_values = presets.get("Balanced Performer", {})

    result = None
    input_data = default_values
    error_msg = None

    if request.method == "POST":
        try:
            # 1. Parse and validate input data
            input_data, warnings = validate_and_parse_input(request.form)
            # 2. Run ensemble inference
            result = execute_inference(input_data)
        except Exception as e:
            error_msg = f"Prediction Error: {str(e)}"
            print(f"[ERROR] POST /predict: {error_msg}")

    return render_template(
        "prediction.html",
        presets=presets,
        cat_options=cat_options,
        input_data=input_data,
        result=result,
        error_msg=error_msg
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API Endpoint: Accepts JSON payload, returns inference results & factors.
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        if not data:
            return jsonify({
                "success": False, 
                "error": "Empty or malformed payload. Please provide valid student features."
            }), 400

        # Validate and parse input
        parsed_data, warnings = validate_and_parse_input(data)
        
        # Execute ML inference
        res = execute_inference(parsed_data)

        return jsonify({
            "success": True,
            "data": parsed_data,
            "prediction": res,
            "warnings": warnings if warnings else None
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route("/api/presets", methods=["GET"])
def api_presets():
    """
    REST API Endpoint: Returns predefined student profile presets.
    """
    return jsonify({
        "success": True,
        "presets": feature_info.get("presets", {})
    }), 200

@app.route("/about")
def about():
    """
    Model Architecture & Viva Insights Page:
    Displays ensemble comparison tables, permutation feature rankings & viva notes.
    """
    metrics = feature_info.get("best_metrics", {})
    summary = feature_info.get("dataset_summary", {})
    models_comp = feature_info.get("models_comparison", {})
    top_features = feature_info.get("top_features", [])
    return render_template(
        "about.html", 
        metrics=metrics, 
        summary=summary, 
        models_comp=models_comp, 
        top_features=top_features
    )

# ----------------------------------------------------------------------------
# 7. Server Execution Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Flask server on http://127.0.0.1:{port} ...")
    app.run(host="127.0.0.1", port=port, debug=True)
