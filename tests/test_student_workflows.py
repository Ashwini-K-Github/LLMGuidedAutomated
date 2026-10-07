"""Regression checks for the student datasets reported by users."""

import pandas as pd

from backend.cleaning.engine import apply_cleaning_recommendations
from backend.cleaning.profiler import profile_dataset
from backend.ml.objective_analyzer import analyze_objective
from backend.ml.question_generator import generate_questions
from backend.utils.dataio import read_tabular
from backend.utils.llm import _generate_mock_recommendations


DIRTY_STUDENTS = """Student_ID,Name,Gender,Age,Department,CGPA,Email,Phone,City,Admission_Date
101,Ashwini,Female,22,Data Analytics,9.1,ashwini@gmail.com,9876543210,Coimbatore,2025-06-10
102,Priya,F,21,Data analytics,8.8,priyagmail.com,9876543,Chennai,10/06/2025
103,Kavya,female,150,DATA ANALYTICS,9.3,kavya@gmail.com,9876543211,Chennai,2025/06/11
104,Divya,Male,,Computer Science,8.4,divya@gmail.com,9876543212,Bangalore,12-06-2025
105,Meena,M,22,Comp Sci,missing,meena@gmail.com,9876543213,Bengaluru,2025-06-13
106,Anu,FEMALE,23,Computer science,8.9,anu@gmail,9876543214,Bangalore,13/06/2025
107,Riya,Unknown,20,AI,7.8,riya@gmail.com,,Coimbatore,2025-06-14
108,Sneha,Female,21,Artificial Intelligence,9.5,sneha@gmail.com,9876543216,Coimbatore,14-06-2025
108,Sneha,Female,21,Artificial Intelligence,9.5,sneha@gmail.com,9876543216,Coimbatore,14-06-2025
109,Harini,None,19,AI,8.6,harini@gmail.com,9876543217,Chennai,2025-06-15
110,Nisha,Female,22,Data Analytics,9.2,nisha@gmail.com,9876543218,Chennai,2025-06-16
111,Kavi,male,0,Data Analytics,12,kavi@gmail.com,9876543219,Chennai,2025-06-17
112,Latha,Female,24,CSE,8.7,latha@gmail.com,Unknown,Coimbatore,17/06/2025
113,Deepa,F,,Computer Science,8.1,deepa@gmail.com,9876543220,Chennai,2025-06-18
114,Renu,female,21,Computer Science,8.5,renu@gmail,9876543221,Coimbatore,18-06-2025
"""


def _dirty_frame():
    frame, _ = read_tabular(DIRTY_STUDENTS.encode("utf-8"), "students.csv")
    return frame


def run(results):
    df = _dirty_frame()

    try:
        recommendations = _generate_mock_recommendations(df)
        results.ok("mixed numeric token does not crash semantic recommendations")
    except Exception as exc:
        results.fail("mixed numeric token does not crash semantic recommendations", str(exc))
        return

    cgpa = next((r for r in recommendations["columns"] if r["column"] == "CGPA"), None)
    # pandas versions differ: some preserve the literal token, while newer
    # parsers recognize it as NaN during CSV loading. Both must be handled.
    missing_detected = bool(
        cgpa is not None and (
            "missing" in cgpa["disguised_nulls"] or df["CGPA"].isna().any()
        )
    )
    results.check("'missing' CGPA is detected as absent", missing_detected, str(cgpa))
    results.check("mixed CGPA is safely cast to float",
                  cgpa is not None and cgpa["type_cast"] == "float", str(cgpa))
    results.check("CGPA imputation is based on parsed numeric values",
                  cgpa is not None and isinstance(cgpa["imputation_value"], float), str(cgpa))

    try:
        cleaned, _ = apply_cleaning_recommendations(df, recommendations)
        results.check("recommended cleaning converts CGPA to numeric",
                      pd.api.types.is_numeric_dtype(cleaned["CGPA"]), str(cleaned["CGPA"].dtype))
    except Exception as exc:
        results.fail("recommended cleaning converts CGPA to numeric", str(exc))

    profile = profile_dataset(df)
    analysis = analyze_objective(
        df, profile["columns"], "Can we predict whether a student will pass or fail?",
        "Mock (Default)",
    )
    results.check("missing Pass/Fail target is reported unavailable",
                  analysis["target_available"] is False and analysis["target"] is None,
                  str(analysis))
    results.check("no predictors are selected without a target",
                  analysis["features"] == [], str(analysis["features"]))

    questions = generate_questions(df, profile["columns"], "Mock (Default)")
    results.check("question generator does not invent a Pass/Fail target",
                  not any("pass or fail" in q.lower() for q in questions), str(questions))

    labelled = df.copy()
    labelled["Semester"] = [1, 2, 3] * 5
    labelled["Placement_Training"] = ["Yes", "No", "Yes"] * 5
    labelled["Placement_Status"] = ["Placed", "Not Placed", "Placed"] * 5
    labelled_profile = profile_dataset(labelled)

    # 1. Verify question generator suggests placement and does not suggest semester or pass/fail
    labelled_questions = generate_questions(labelled, labelled_profile["columns"], "Mock (Default)")
    results.check("question generator suggests placement prediction when placement data exists",
                  any("placed" in q.lower() or "placement" in q.lower() for q in labelled_questions),
                  str(labelled_questions))
    results.check("question generator does not suggest Semester as target",
                  not any("semester" in q.lower() for q in labelled_questions),
                  str(labelled_questions))
    results.check("question generator does not suggest Pass/Fail when not available",
                  not any("pass or fail" in q.lower() for q in labelled_questions),
                  str(labelled_questions))

    # 2. When user asks Pass/Fail on student dataset with Placement_Status and Semester:
    pass_fail_attempt = analyze_objective(
        labelled, labelled_profile["columns"], "Can we predict whether a student will pass or fail?",
        "Mock (Default)",
    )
    results.check("missing Pass/Fail target is reported unavailable",
                  pass_fail_attempt["target_available"] is False and pass_fail_attempt["target"] is None,
                  str(pass_fail_attempt))
    results.check("unrelated numeric column 'Semester' is NOT selected as target",
                  pass_fail_attempt["target"] != "Semester",
                  str(pass_fail_attempt["target"]))
    results.check("Pass/Fail unavailability message points user to available targets like Placement_Status",
                  "Placement_Status" in pass_fail_attempt["unavailable_reason"],
                  pass_fail_attempt["unavailable_reason"])

    # 3. When user asks to predict placement:
    placement = analyze_objective(
        labelled, labelled_profile["columns"], "Can we predict whether a student will be placed?",
        "Mock (Default)",
    )
    results.equals("placement objective uses the placement label",
                   placement["target"], "Placement_Status")
    results.equals("placement problem type is Classification",
                   placement["problem_type"], "Classification")
    results.check("identifier and PII are not proposed as placement predictors",
                  not {"Student_ID", "Name", "Email", "Phone", "Age", "Gender", "Admission_Date"}.intersection(placement["features"]),
                  str(placement["features"]))
    results.equals("exactly 5 candidate algorithms are recommended",
                   len(placement["candidate_algorithms"]), 5)

    # Verify Issue 2: Excluded features and reasons in objective analysis
    results.check("Student_ID is excluded before feature selection",
                  "Student_ID" in placement["excluded"], str(placement["excluded"]))
    results.check("Age is excluded before feature selection",
                  "Age" in placement["excluded"], str(placement["excluded"]))
    exc_dict = {item["feature"]: item["reason"] for item in placement.get("excluded_by_policy", [])}
    results.equals("Student_ID excluded as identifier",
                   exc_dict.get("Student_ID"), "Excluded as identifier")
    results.equals("Age excluded by project policy",
                   exc_dict.get("Age"), "Excluded by project policy")

    # 4. Honest model recommendation: test "No reliable model found" when baseline is not beaten
    from backend.ml.model_selector import select_best_model
    mock_evals = {
        "Logistic Regression": {"accuracy": 0.51, "balanced_accuracy": 0.50, "f1_macro": 0.50, "roc_auc": 0.51},
        "Random Forest": {"accuracy": 0.50, "balanced_accuracy": 0.49, "f1_macro": 0.49, "roc_auc": 0.50},
        "Decision Tree": {"accuracy": 0.48, "balanced_accuracy": 0.48, "f1_macro": 0.47, "roc_auc": 0.48},
        "SVM": {"accuracy": 0.50, "balanced_accuracy": 0.50, "f1_macro": 0.49, "roc_auc": 0.50},
        "Extra Trees": {"accuracy": 0.50, "balanced_accuracy": 0.50, "f1_macro": 0.49, "roc_auc": 0.50},
    }
    mock_cv = {
        "Logistic Regression": {"mean": 0.50, "std": 0.05, "metric": "f1_macro", "folds": 5},
        "Random Forest": {"mean": 0.49, "std": 0.04, "metric": "f1_macro", "folds": 5},
        "Decision Tree": {"mean": 0.48, "std": 0.06, "metric": "f1_macro", "folds": 5},
        "SVM": {"mean": 0.49, "std": 0.05, "metric": "f1_macro", "folds": 5},
        "Extra Trees": {"mean": 0.49, "std": 0.04, "metric": "f1_macro", "folds": 5},
    }
    mock_baseline = {
        "f1_macro": 0.50,
        "strategy": "stratified random guessing",
        "cv": {"mean": 0.50, "std": 0.02, "metric": "f1_macro", "folds": 5}
    }
    honest_rec = select_best_model(mock_evals, "Classification", "Result", cv_results=mock_cv, baseline_metrics=mock_baseline)
    results.check("model that fails to beat baseline reports beats_baseline False",
                  honest_rec["beats_baseline"] is False)
    results.equals("model that fails to beat baseline reports 'No reliable model found'",
                   honest_rec["recommended_model"], "No reliable model found")
    results.check("reason begins with 'No reliable model found'",
                  honest_rec["reason"].startswith("No reliable model found for predicting 'Result'"),
                  honest_rec["reason"])

    higher_score = select_best_model(
        {"Logistic Regression": {"f1_macro": 0.59}, "Random Forest": {"f1_macro": 0.60}},
        "Classification", "Result",
        cv_results={
            "Logistic Regression": {"mean": 0.590, "std": 0.050, "metric": "f1_macro", "folds": 5},
            "Random Forest": {"mean": 0.600, "std": 0.050, "metric": "f1_macro", "folds": 5},
        },
        baseline_metrics={"f1_macro": 0.40, "strategy": "stratified random guessing",
                          "cv": {"mean": 0.40, "std": 0.03, "metric": "f1_macro", "folds": 5}},
    )
    results.equals("highest CV mean remains the best model when ranges overlap",
                   higher_score["best_model"], "Random Forest")
