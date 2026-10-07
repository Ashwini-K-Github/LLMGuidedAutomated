"""Checks feature selection picks relevant attributes and that the model
recommendation is honest about what it has actually shown."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd


def _pipeline(df, target, features, col_types, problem_type):
    """Mirrors the /api/ml/train sequence so the tests exercise the real path."""
    from sklearn.model_selection import train_test_split
    from backend.ml.feature_engineering import FeatureEngineer
    from backend.ml.feature_selection import select_features
    from backend.ml.model_trainer import train_models, cross_validate_models, train_baselines
    from backend.ml.evaluator import evaluate_all_models
    from backend.ml.model_selector import select_best_model

    stratify = df[target].astype(str) if problem_type == "Classification" else None
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify)

    engineer = FeatureEngineer(target, features, col_types).fit(train_df)
    X_train, y_train, names = engineer.transform(train_df)
    X_test, y_test, _ = engineer.transform(test_df)

    X_sel, sel_names, discrete = engineer.transform_for_selection(train_df)
    selection = select_features(X_sel, y_train, sel_names, problem_type, discrete_mask=discrete)

    keep = engineer.model_columns_for(selection["selected"])
    if keep:
        X_train, X_test = X_train[:, keep], X_test[:, keep]

    models, info = train_models(X_train, y_train, problem_type, tune=False)
    evaluations = evaluate_all_models(models, X_test, y_test, problem_type,
                                      X_train=X_train, y_train=y_train,
                                      class_names=engineer.target_classes)
    cv = cross_validate_models(models, X_train, y_train, problem_type)

    # Mirrors the endpoint: every trivial strategy is scored and the strongest
    # becomes the bar the candidate models must clear.
    baselines = train_baselines(X_train, y_train, problem_type)
    baseline_evals = evaluate_all_models(baselines, X_test, y_test, problem_type,
                                         X_train=X_train, y_train=y_train,
                                         class_names=engineer.target_classes)
    baseline_cv = cross_validate_models(baselines, X_train, y_train, problem_type)

    def _bscore(name):
        if name in baseline_cv:
            return baseline_cv[name]["mean"]
        m = baseline_evals.get(name, {})
        v = m.get("f1_macro") if problem_type == "Classification" else m.get("r2_score")
        return v if v is not None else float("-inf")

    baseline_metrics = {}
    if baseline_evals:
        strongest = max(baseline_evals, key=_bscore)
        baseline_metrics = dict(baseline_evals[strongest])
        baseline_metrics["strategy"] = strongest
        baseline_metrics["cv"] = baseline_cv.get(strongest)
    recommendation = select_best_model(evaluations, problem_type, target,
                                       cv_results=cv, baseline_metrics=baseline_metrics,
                                       training_info=info)
    return selection, evaluations, recommendation, engineer


def run(results):
    r = np.random.default_rng(21)
    n = 800

    # 1. A non-monotonic categorical driver must survive selection.
    #    Alphabetical label codes 0..3 map to 40k/100k/100k/40k, so the Pearson
    #    correlation the old implementation used is ~0 by construction.
    pay = {"Analytics": 40_000, "Banking": 100_000, "Consulting": 100_000, "Design": 40_000}
    dept = r.choice(list(pay), n)
    exp = r.uniform(0, 20, n)
    salary = np.array([pay[d] for d in dept]) + exp * 6000 + r.normal(0, 2000, n)
    df = pd.DataFrame({"Department": dept, "YearsExperience": exp, "Salary": salary})
    types = {"Department": "Categorical", "YearsExperience": "Numeric", "Salary": "Numeric"}
    selection, _, rec, _ = _pipeline(df, "Salary", ["Department", "YearsExperience"], types, "Regression")
    results.check("non-monotonic categorical driver is selected",
                  "Department" in selection["selected"],
                  f"selected={selection['selected']}")
    results.check("recommended regression model beats the baseline", rec["beats_baseline"] is True)

    # 2. Pure noise must be discarded.
    real = r.normal(0, 1, n)
    cat = r.choice(["low", "mid", "high"], n)
    effect = {"low": 0, "mid": 25, "high": 60}
    target = real * 10 + np.array([effect[v] for v in cat]) + r.normal(0, 3, n)
    noisy = pd.DataFrame({"RealNum": real, "RealCat": cat, "Target": target})
    for i in range(5):
        noisy[f"Noise{i + 1}"] = r.normal(0, 1, n)
    noisy["NoiseCat"] = r.choice(list("abcdef"), n)
    ntypes = {c: ("Categorical" if c in ("RealCat", "NoiseCat") else "Numeric") for c in noisy.columns}
    sel, _, _, _ = _pipeline(noisy, "Target", [c for c in noisy.columns if c != "Target"],
                             ntypes, "Regression")
    results.check("both real signals retained",
                  {"RealNum", "RealCat"}.issubset(set(sel["selected"])), f"selected={sel['selected']}")
    results.check("no noise column retained",
                  not any(c.startswith("Noise") for c in sel["selected"]), f"selected={sel['selected']}")

    # 3. On an imbalanced target with no signal, the recommendation must say so.
    flat = pd.DataFrame({"F1": r.normal(0, 1, n), "F2": r.normal(0, 1, n),
                         "Status": np.where(r.random(n) < 0.05, "Churn", "Retained")})
    ftypes = {"F1": "Numeric", "F2": "Numeric", "Status": "Categorical"}
    _, evals, rec, engineer = _pipeline(flat, "Status", ["F1", "F2"], ftypes, "Classification")
    results.check("no-signal model is reported as not beating the baseline",
                  rec["beats_baseline"] is False, f"beats_baseline={rec['beats_baseline']}")
    results.check("the caveat appears in the justification",
                  "not" in rec["reason"].lower() and "baseline" in rec["reason"].lower())
    best = evals[rec["best_model"]]
    results.equals("positive class is the minority class", best.get("positive_class"), "Churn")
    results.check("balanced accuracy exposes the lack of signal",
                  best.get("balanced_accuracy") is not None and best["balanced_accuracy"] < 0.65,
                  f"balanced_accuracy={best.get('balanced_accuracy')}")

    # 4. A genuine classification signal must be recommended confidently.
    hours = r.uniform(0, 10, n)
    attendance = r.uniform(40, 100, n)
    outcome = np.where(hours * 3 + attendance * 0.4 + r.normal(0, 4, n) > 45, "Pass", "Fail")
    good = pd.DataFrame({"StudyHours": hours, "Attendance": attendance, "Result": outcome})
    gtypes = {"StudyHours": "Numeric", "Attendance": "Numeric", "Result": "Categorical"}
    _, evals, rec, _ = _pipeline(good, "Result", ["StudyHours", "Attendance"], gtypes, "Classification")
    results.equals("five classification algorithms are compared", len(evals), 5)
    required_metrics = {"accuracy", "precision", "recall", "f1_score", "roc_auc"}
    results.check("every classifier reports the requested comparison metrics",
                  all(required_metrics.issubset(metrics) for metrics in evals.values()),
                  str({name: sorted(metrics) for name, metrics in evals.items()}))
    results.check("real classification signal beats the baseline", rec["beats_baseline"] is True)
    results.check("cross-validated evidence is attached", rec.get("cv") is not None)
    results.check("best-model explanation covers CV, comparison, and baseline",
                  all(term in rec["reason"].lower()
                      for term in ("cross-validation", "baseline", "held-out")),
                  rec["reason"])

    # 5. Target leakage is flagged.
    leaky = good.copy()
    leaky["ResultFlag"] = (leaky["Result"] == "Pass").astype(int).astype(str)
    ltypes = dict(gtypes); ltypes["ResultFlag"] = "Categorical"
    sel, _, _, _ = _pipeline(leaky, "Result", ["StudyHours", "Attendance", "ResultFlag"],
                             ltypes, "Classification")
    flagged = [w["feature"] for w in sel["leakage_warnings"]]
    results.check("a feature derived from the target is flagged as leakage",
                  "ResultFlag" in flagged, f"flagged={flagged}")


def run_column_exclusion(results):
    """Feature columns must not be dropped by substring matches on their name."""
    from backend.ml.objective_analyzer import _is_excluded_column

    keep = [("Kidnapping", "Numeric"), ("Humidity", "Numeric"), ("Width", "Numeric"),
            ("Residual", "Numeric"), ("Incident_Count", "Numeric"),
            ("Video_Length", "Numeric"), ("Tournament", "Categorical"),
            ("Number_of_Rooms", "Numeric"), ("State", "Categorical")]
    for column, ctype in keep:
        results.check(f"'{column}' is kept as a feature",
                      _is_excluded_column(column, ctype) is False)

    drop = [("StudentID", "Identifier"), ("Employee_ID", "Numeric"),
            ("Full_Name", "Free Text"), ("Work_Email", "Email"),
            ("Contact_No", "Phone Number"), ("Account_Number", "Numeric"),
            ("Order_No", "Numeric"), ("Age", "Numeric"),
            ("Gender", "Categorical")]
    for column, ctype in drop:
        results.check(f"'{column}' is excluded as an identifier or personal data",
                      _is_excluded_column(column, ctype) is True)


def run_mi_robustness(results):
    """A high-cardinality categorical must not void the whole ranking.

    mutual_info_* raises "Found array with 0 sample(s)" when a column declared
    discrete has levels with fewer members than n_neighbors, which is normal for
    a column like State where most levels occur once. A single bulk call then
    failed and selection silently reverted to the univariate F-test.
    """
    from backend.ml.feature_engineering import FeatureEngineer
    from backend.ml.feature_selection import select_features

    r = np.random.default_rng(31)
    n = 60
    states = [f"State_{i}" for i in range(n)]          # every level occurs exactly once
    signal = r.normal(0, 1, n)
    df = pd.DataFrame({"State": states, "Signal": signal,
                       "Noise": r.normal(0, 1, n), "Target": signal * 5 + r.normal(0, 1, n)})
    types = {"State": "Categorical", "Signal": "Numeric", "Noise": "Numeric", "Target": "Numeric"}

    engineer = FeatureEngineer("Target", ["State", "Signal", "Noise"], types).fit(df)
    X_sel, names, discrete = engineer.transform_for_selection(df)
    selection = select_features(X_sel, engineer.transform_target(df), names,
                                "Regression", discrete_mask=discrete)

    results.check("mutual information is used, not the F-test fallback",
                  selection["method"].startswith("Mutual Information"),
                  selection["method"])
    results.check("the true signal outranks the noise",
                  selection["importances"].get("Signal", 0) > selection["importances"].get("Noise", 0),
                  str(selection["importances"]))
    results.check("the true signal is selected", "Signal" in selection["selected"],
                  str(selection["selected"]))


def run_feature_selection_reasons(results):
    """Features meeting MI threshold must be labelled 'mi_threshold' even if significant;
    features meeting only significance must be labelled 'significance';
    features meeting neither must be labelled 'not_selected'."""
    from backend.ml.feature_selection import select_features

    r = np.random.default_rng(123)
    n = 800
    y = r.choice([0, 1], n)
    # Feat 1: Strong relationship -> high MI and significant
    x_strong = y * 4.0 + r.normal(0, 0.5, n)
    # Feat 2: Weak linear correlation -> low MI (< 30% of Feat 1), but significant due to n
    x_weak_sig = y * 0.35 + r.normal(0, 1.0, n)
    # Feat 3: Pure noise -> neither passes
    x_noise = r.normal(0, 1.0, n)

    X = np.column_stack([x_strong, x_weak_sig, x_noise])
    names = ["Final_Exam_Score", "GPA", "Random_Noise"]

    selection = select_features(X, y, names, "Classification")
    details = selection["details"]

    # 1. Feature passing both MI and significance: primary label must be mi_threshold
    results.equals("Final_Exam_Score has mi_threshold path (not significance)",
                   details["Final_Exam_Score"]["selection_path"], "mi_threshold")
    results.check("Final_Exam_Score reason mentions MI threshold passed",
                  "Passed the MI threshold" in details["Final_Exam_Score"]["reason"],
                  details["Final_Exam_Score"]["reason"])
    results.check("Final_Exam_Score reason mentions FDR significance as well",
                  "statistically significant after FDR correction" in details["Final_Exam_Score"]["reason"],
                  details["Final_Exam_Score"]["reason"])
    results.check("Final_Exam_Score is selected",
                  "Final_Exam_Score" in selection["selected"])

    # 2. Feature failing MI threshold (< 30%) but passing FDR q < 0.05
    results.equals("GPA has significance path",
                   details["GPA"]["selection_path"], "significance")
    results.check("GPA reason explains FDR q < 0.05 despite MI below threshold",
                  "despite mutual information being below" in details["GPA"]["reason"],
                  details["GPA"]["reason"])
    results.check("GPA is selected",
                  "GPA" in selection["selected"])

    # 3. Feature failing both
    results.equals("Random_Noise has not_selected path",
                   details["Random_Noise"]["selection_path"], "not_selected")
    results.equals("Random_Noise reason states both conditions failed",
                   details["Random_Noise"]["reason"],
                   "Did not meet the MI threshold and not statistically significant after FDR correction.")
    results.check("Random_Noise is in removed list",
                  "Random_Noise" in selection["removed"])


def run_layout_guards(results):
    """The action buttons must stay inside the viewport for wide tables.

    .content-area is a flex item; without min-width:0 it refuses to shrink below
    the intrinsic width of the widest table, the page overflows horizontally,
    and because body sets overflow-x:hidden the right-aligned action bar is
    clipped out of reach. That is what made "Proceed to Semantic Analysis"
    invisible for datasets with more columns than the bundled samples.
    """
    import os
    css_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "style.css")
    with open(css_path, "r", encoding="utf-8") as fh:
        css = fh.read()

    def rule_body(selector):
        marker = selector + " {"
        if marker not in css:
            return ""
        start = css.index(marker)
        return css[start:css.index("}", start)]

    for selector in (".content-area", ".step-view", ".table-panel"):
        results.check(f"{selector} sets min-width:0",
                      "min-width: 0" in rule_body(selector),
                      "flex item would keep min-width:auto and overflow the page")
    results.check(".table-scroll-container is capped at the container width",
                  "max-width: 100%" in rule_body(".table-scroll-container"))


def run_problem_type(results):
    """The backend must not accept an impossible task description."""
    from backend.ml.objective_analyzer import resolve_problem_type

    continuous = pd.Series(np.random.default_rng(1).normal(100, 10, 300), name="Price")
    resolved, reason = resolve_problem_type(continuous, "Numeric", requested="Classification")
    results.equals("continuous target requested as classification becomes regression", resolved, "Regression")
    results.check("the override is explained", "continuous" in reason.lower(), reason)

    binary = pd.Series(["Yes", "No"] * 100, name="Churn")
    resolved, _ = resolve_problem_type(binary, "Categorical", requested="Regression")
    results.equals("categorical target stays classification", resolved, "Classification")

    coded = pd.Series([1, 2, 3, 4, 5] * 60, name="Rating")
    resolved, _ = resolve_problem_type(coded, "Numeric", requested="Classification")
    results.equals("few whole-number classes honour a classification request", resolved, "Classification")


def run_degenerate_inputs(results):
    """Small or awkward splits must produce honest output, never an exception."""
    from backend.ml.feature_engineering import FeatureEngineer
    from backend.ml.model_trainer import train_models
    from backend.ml.evaluator import evaluate_all_models
    from backend.ml.model_selector import select_best_model
    from sklearn.model_selection import train_test_split

    # Four target classes, a two-row test split, and a class seen only at test
    # time. Choosing the positive class from the training labels used to raise
    # "pos_label=0 is not a valid label".
    df = pd.DataFrame({
        "Price": [299.99, 19.99, None, 49.99, 15.50, 1200.00],
        "Quantity": [1, 2, 1, 999, 3, 1],
        "Category": ["Electronics", "Clothing", "Electronics", "Clothing", "Home", "Books"],
    })
    types = {"Price": "Numeric", "Quantity": "Numeric", "Category": "Categorical"}
    train_df, test_df = train_test_split(df, test_size=0.34, random_state=42)
    engineer = FeatureEngineer("Category", ["Price", "Quantity"], types).fit(train_df)
    X_train, y_train, _ = engineer.transform(train_df)
    X_test, y_test, _ = engineer.transform(test_df)

    models, info = train_models(X_train, y_train, "Classification", tune=False)
    try:
        evaluations = evaluate_all_models(models, X_test, y_test, "Classification",
                                          X_train=X_train, y_train=y_train,
                                          class_names=engineer.target_classes)
        results.check("tiny multi-class split evaluates without raising",
                      len(evaluations) > 0, f"{len(evaluations)} models evaluated")
        any_metrics = next(iter(evaluations.values()))
        results.check("a multi-class problem is not evaluated as binary",
                      any_metrics["is_binary"] is False)
    except Exception as exc:
        results.fail("tiny multi-class split evaluates without raising", f"{type(exc).__name__}: {exc}")
        return

    results.check("cross-validation is skipped and said to be skipped",
                  any("training rows" in n or "too small" in n for n in info["notes"]),
                  str(info["notes"]))

    empty = select_best_model({}, "Classification", "Category")
    results.check("an empty evaluation set yields no recommendation",
                  empty["best_model"] is None and empty["beats_baseline"] is False)


def run_no_leakage(results):
    """Preprocessing statistics must come only from the training partition."""
    from sklearn.model_selection import train_test_split
    from backend.ml.feature_engineering import FeatureEngineer

    r = np.random.default_rng(7)
    df = pd.DataFrame({"A": r.normal(0, 1, 400), "B": r.choice(["x", "y", "z"], 400),
                       "T": r.normal(0, 1, 400)})
    types = {"A": "Numeric", "B": "Categorical", "T": "Numeric"}
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    engineer = FeatureEngineer("T", ["A", "B"], types).fit(train_df)
    train_mean = engineer.scaling_params["A"]["mean"]
    results.check("scaling mean matches the training partition only",
                  abs(train_mean - train_df["A"].mean()) < 1e-9,
                  f"fitted={train_mean:.6f} train={train_df['A'].mean():.6f} full={df['A'].mean():.6f}")
    results.check("scaling mean differs from the full-dataset mean",
                  abs(train_mean - df["A"].mean()) > 1e-12)

    X_test, _, _ = engineer.transform(test_df)
    results.check("test matrix has the same width as the training matrix",
                  X_test.shape[1] == len(engineer.feature_names),
                  f"{X_test.shape[1]} vs {len(engineer.feature_names)}")

    # An unseen category must not collide with a real one
    unseen = pd.DataFrame({"A": [0.0], "B": ["brand_new_level"], "T": [0.0]})
    X_unseen, _, _ = engineer.transform(unseen)
    onehot_idx = [i for i, name in enumerate(engineer.feature_names) if name.startswith("B=")]
    results.check("unseen category does not activate a known category column",
                  all(X_unseen[0, i] == 0.0 for i in onehot_idx
                      if not engineer.feature_names[i].endswith("__other__")),
                  str([(engineer.feature_names[i], X_unseen[0, i]) for i in onehot_idx]))
