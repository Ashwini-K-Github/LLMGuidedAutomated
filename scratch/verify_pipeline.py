import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Create synthetic dataset
np.random.seed(42)
n_samples = 100

data = {
    "StudentID": [f"STU{1000 + i}" for i in range(n_samples)],
    "Name": [f"Student {i}" for i in range(n_samples)],
    "Email": [f"student_{i}@university.edu" for i in range(n_samples)],
    "Phone": [f"98765{str(i).zfill(5)}" for i in range(n_samples)],
    "Age": np.random.randint(18, 25, size=n_samples),
    "GPA": np.round(np.random.uniform(2.0, 4.0, size=n_samples), 2),
    "EnrollmentDate": pd.date_range(start="2023-09-01", periods=n_samples).strftime("%Y-%m-%d").tolist(),
    "Major": np.random.choice(["CS", "Major_Math", "Physics", "Chemistry"], size=n_samples),
    "Attendance": np.random.randint(50, 100, size=n_samples),
    "InternalMarks": np.random.randint(10, 30, size=n_samples),
    "AssignmentMarks": np.random.randint(15, 40, size=n_samples)
}

# SemesterMarks is computed linearly with some noise for Regression target
data["SemesterMarks"] = (
    data["Attendance"] * 0.3 + 
    data["InternalMarks"] * 1.2 + 
    data["AssignmentMarks"] * 0.8 + 
    np.random.normal(0, 3, size=n_samples)
).round(1)

# Pass_Fail represents Classification target
data["Pass_Fail"] = np.where(data["SemesterMarks"] >= 68, "Pass", "Fail")

df = pd.DataFrame(data)

# Importer mock profile types
col_types = {
    "StudentID": "Identifier",
    "Name": "Free Text",
    "Email": "Email",
    "Phone": "Phone Number",
    "Age": "Numeric",
    "GPA": "Numeric",
    "EnrollmentDate": "Date",
    "Major": "Categorical",
    "Attendance": "Numeric",
    "InternalMarks": "Numeric",
    "AssignmentMarks": "Numeric",
    "SemesterMarks": "Numeric",
    "Pass_Fail": "Categorical"
}

def run_test_pipeline(target_col, feature_cols, problem_type):
    print(f"\n=======================================================")
    print(f"Testing ML Pipeline: Target = {target_col} | Task = {problem_type}")
    print(f"=======================================================")
    
    from sklearn.model_selection import train_test_split
    from backend.ml.feature_engineering import FeatureEngineer
    from backend.ml.feature_selection import select_features
    from backend.ml.model_trainer import train_models
    from backend.ml.evaluator import evaluate_all_models
    from backend.ml.model_selector import select_best_model
    
    # 1. Split raw data (80/20)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # 2. Fit Feature Engineer
    engineer = FeatureEngineer(target_col, feature_cols, col_types)
    engineer.fit(train_df)
    
    X_train, y_train, feature_names = engineer.transform(train_df)
    X_test, y_test, _ = engineer.transform(test_df)
    
    print(f"[OK] Preprocessing completed. Feature count: {len(feature_names)}")
    print(f"Engineered features: {feature_names}")
    
    # 3. Feature Selection (scored one column per original feature)
    X_select, select_names, discrete_mask = engineer.transform_for_selection(train_df)
    selection = select_features(X_select, y_train, select_names, problem_type,
                                discrete_mask=discrete_mask)
    print(f"[OK] Feature selection method: {selection['method']}")
    print(f"Significance ranks:")
    for f, val in list(selection["importances"].items())[:5]:
        print(f"  - {f}: {val}")
        
    # 4. Train Models on the training split only (no re-splitting, no leakage)
    fitted_models, training_info = train_models(X_train, y_train, problem_type)
    print(f"[OK] Models trained successfully: {list(fitted_models.keys())}")

    # 5. Evaluate on the untouched test split
    evaluations = evaluate_all_models(fitted_models, X_test, y_test, problem_type,
                                      X_train=X_train, y_train=y_train,
                                      class_names=engineer.target_classes)
    print(f"[OK] Evaluations finished for {len(evaluations)} models.")
    
    # Show metric preview for first model
    first_model = list(evaluations.keys())[0]
    print(f"Sample metrics ({first_model}): {evaluations[first_model]}")
    
    # 6. Recommendation
    best = select_best_model(evaluations, problem_type, target_col)
    print(f"\n[BEST RECOMMENDED MODEL]")
    print(f"  Recommended Model: {best['best_model']}")
    print(f"  Justification: {best['reason']}")
    print(f"  Best metrics: {best['metrics']}")

if __name__ == "__main__":
    # Test Regression: predict SemesterMarks using numerical features, age, and major
    features_reg = ["Age", "GPA", "EnrollmentDate", "Major", "Attendance", "InternalMarks", "AssignmentMarks"]
    run_test_pipeline("SemesterMarks", features_reg, "Regression")
    
    # Test Classification: predict Pass_Fail
    features_clf = ["Age", "GPA", "Major", "Attendance", "InternalMarks", "AssignmentMarks"]
    run_test_pipeline("Pass_Fail", features_clf, "Classification")
    
    print("\n[SUCCESS] Pipeline verification runs completed successfully!")
