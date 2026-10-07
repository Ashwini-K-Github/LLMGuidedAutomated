import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.dummy import DummyClassifier, DummyRegressor

# Hyperparameter search is skipped above this many rows to keep the request
# responsive; the defaults are used instead and the response says so.
MAX_ROWS_FOR_TUNING = 5000
MIN_ROWS_FOR_TUNING = 30
DEFAULT_CV_FOLDS = 5


def select_candidate_algorithms(problem_type: str, X_train: np.ndarray = None,
                                y_train: np.ndarray = None, n_features: int = None) -> tuple:
    """
    Dynamically chooses the 5 most suitable ML algorithms based on the problem type
    and dataset characteristics (rows, features, class imbalance, multi-class, size).

    Returns (candidates_dict, algorithm_reasons)
    where candidates_dict is {name: (estimator, param_grid)}.
    """
    is_classification = str(problem_type).lower().startswith("class")
    n_samples = len(y_train) if y_train is not None else 100
    if n_features is None:
        n_features = X_train.shape[1] if X_train is not None and X_train.ndim > 1 else 5
    n_features = int(max(0, n_features))
    knn_k = min(5, max(1, n_samples - 1)) if n_samples > 1 else 1

    if is_classification:
        classes, counts = (
            np.unique(y_train, return_counts=True)
            if y_train is not None and len(y_train)
            else (np.array([0, 1]), np.array([50, 50]))
        )
        n_classes = len(classes)
        min_class_ratio = (counts.min() / n_samples) if n_samples > 0 else 0.5
        is_imbalanced = min_class_ratio < 0.20
        is_small = n_samples < 150
        is_high_dim = n_features > 15 or (n_samples > 0 and n_features > n_samples // 3)

        pool = {
            "Logistic Regression": (
                LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
                {"C": [0.1, 1.0, 10.0]},
                "Chosen as an interpretable linear baseline with balanced class weighting to prevent majority-class bias."
            ),
            "Decision Tree": (
                DecisionTreeClassifier(random_state=42, class_weight="balanced"),
                {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5]},
                "Chosen for transparent rule-based partitioning and invariance to monotonic feature scaling."
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=42, n_estimators=100, class_weight="balanced"),
                {"n_estimators": [100, 300], "max_depth": [None, 10]},
                "Chosen for robust ensemble averaging across non-linear interactions and resistance to overfitting."
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=42, n_estimators=100),
                {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                "Chosen for sequential boosting to minimize residual classification errors on structured tabular data."
            ),
            "SVM": (
                SVC(probability=True, random_state=42, class_weight="balanced"),
                {"C": [0.1, 1.0, 10.0]},
                "Chosen for maximum-margin separation with kernel projection, well-suited to small-to-medium sample sizes."
            ),
            "KNN": (
                KNeighborsClassifier(n_neighbors=knn_k),
                {"n_neighbors": [3, 5, 11] if n_samples >= 15 else [min(3, n_samples)], "weights": ["uniform", "distance"]},
                "Chosen as an instance-based non-parametric baseline sensitive to local proximity."
            ),
            "Gaussian Naive Bayes": (
                GaussianNB(),
                {},
                "Chosen as a fast probabilistic baseline with low variance, effective for smaller datasets."
            ),
            "Extra Trees": (
                ExtraTreesClassifier(random_state=42, n_estimators=100, class_weight="balanced"),
                {"n_estimators": [100, 200], "max_depth": [None, 10]},
                "Chosen as an extremely randomized forest ensemble to reduce variance on noisy or correlated features."
            ),
        }

        # Select exactly 5 most suitable algorithms
        if is_imbalanced:
            chosen = ["Logistic Regression", "Random Forest", "Decision Tree", "Extra Trees", "SVM"]
        elif is_small:
            chosen = ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "Gaussian Naive Bayes"]
        elif n_classes > 2:
            chosen = ["Logistic Regression", "Random Forest", "Decision Tree", "Gradient Boosting", "Gaussian Naive Bayes"]
        elif is_high_dim:
            chosen = ["Logistic Regression", "SVM", "Random Forest", "Extra Trees", "Decision Tree"]
        else:
            chosen = ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM"]

        candidates = {name: (pool[name][0], pool[name][1]) for name in chosen}
        reasons = {name: pool[name][2] for name in chosen}
        return candidates, reasons

    else:
        # Regression
        is_small = n_samples < 150
        is_high_dim = n_features > 10 or (n_samples > 0 and n_features > n_samples // 3)

        pool = {
            "Linear Regression": (
                LinearRegression(),
                {},
                "Chosen as an Ordinary Least Squares baseline with directly interpretable coefficient slopes."
            ),
            "Ridge Regression": (
                Ridge(random_state=42),
                {"alpha": [0.1, 1.0, 10.0]},
                "Chosen for L2 regularisation to stabilize collinear predictors and prevent parameter explosion."
            ),
            "Lasso Regression": (
                Lasso(random_state=42),
                {"alpha": [0.01, 0.1, 1.0]},
                "Chosen for L1 regularisation promoting feature sparsity by driving irrelevant weights to zero."
            ),
            "Decision Tree": (
                DecisionTreeRegressor(random_state=42),
                {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5]},
                "Chosen for step-wise rule splits without assuming linearity in target relationships."
            ),
            "Random Forest": (
                RandomForestRegressor(random_state=42, n_estimators=100),
                {"n_estimators": [100, 300], "max_depth": [None, 10]},
                "Chosen for high predictive stability by averaging de-correlated decision trees on tabular data."
            ),
            "Gradient Boosting": (
                GradientBoostingRegressor(random_state=42),
                {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
                "Chosen for sequential gradient boosting that iteratively minimizes prediction residuals."
            ),
            "SVR": (
                SVR(),
                {"C": [0.1, 1.0, 10.0]},
                "Chosen for margin-based regression with epsilon-insensitive loss, robust to outlier targets."
            ),
            "KNN": (
                KNeighborsRegressor(n_neighbors=knn_k),
                {"n_neighbors": [3, 5, 11] if n_samples >= 15 else [min(3, n_samples)]},
                "Chosen as a non-parametric instance-based regression baseline averaging local neighbors."
            ),
        }

        # Select exactly 5 most suitable algorithms
        if is_small:
            chosen = ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest", "SVR"]
        elif is_high_dim:
            chosen = ["Ridge Regression", "Lasso Regression", "Linear Regression", "Random Forest", "SVR"]
        elif n_samples > 1000:
            chosen = ["Random Forest", "Gradient Boosting", "Ridge Regression", "Decision Tree", "Linear Regression"]
        else:
            chosen = ["Linear Regression", "Ridge Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]

        candidates = {name: (pool[name][0], pool[name][1]) for name in chosen}
        reasons = {name: pool[name][2] for name in chosen}
        return candidates, reasons


def _candidate_models(problem_type: str, X_train: np.ndarray = None,
                      y_train: np.ndarray = None) -> dict:
    """Returns {name: (estimator, param_grid)} dynamically tailored to the dataset."""
    candidates, _ = select_candidate_algorithms(problem_type, X_train, y_train)
    return candidates


def make_cv(y_train: np.ndarray, problem_type: str, folds: int = DEFAULT_CV_FOLDS):
    """Builds a CV splitter that is valid for the data at hand, or None if too small."""
    n = len(y_train)
    if str(problem_type).lower().startswith("class"):
        _, counts = np.unique(y_train, return_counts=True)
        usable = int(min(folds, counts.min())) if len(counts) else 0
        if usable < 2 or n < 10:
            return None
        return StratifiedKFold(n_splits=usable, shuffle=True, random_state=42)
    usable = int(min(folds, max(2, n // 5))) if n < 25 else int(min(folds, n))
    if usable < 2 or n < 10:
        return None
    return KFold(n_splits=usable, shuffle=True, random_state=42)


def scoring_metric(problem_type: str) -> str:
    return "f1_macro" if str(problem_type).lower().startswith("class") else "r2"


def train_baselines(X_train: np.ndarray, y_train: np.ndarray, problem_type: str) -> dict:
    """Trivial reference models, all of them.

    Without a baseline there is no way to tell whether a reported accuracy of
    0.93 is a real result or just the majority-class rate.

    More than one strategy is needed. Macro-F1 rewards a model merely for
    attempting the minority class, so a balanced classifier scores above a
    most-frequent baseline even when its features are pure noise. A stratified
    random baseline also attempts the minority class at the same base rate, which
    is the honest bar for "is there any signal here".
    """
    if str(problem_type).lower().startswith("class"):
        candidates = {
            "most frequent class": DummyClassifier(strategy="most_frequent"),
            "stratified random guessing": DummyClassifier(strategy="stratified", random_state=42),
        }
    else:
        candidates = {
            "mean of the target": DummyRegressor(strategy="mean"),
            "median of the target": DummyRegressor(strategy="median"),
        }

    fitted = {}
    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            fitted[name] = model
        except Exception as exc:
            print(f"Could not fit baseline '{name}': {exc}")
    return fitted


def train_baseline(X_train: np.ndarray, y_train: np.ndarray, problem_type: str):
    """Backwards-compatible single baseline (the most conservative strategy)."""
    baselines = train_baselines(X_train, y_train, problem_type)
    return next(iter(baselines.values())) if baselines else None


def train_models(X_train: np.ndarray, y_train: np.ndarray, problem_type: str,
                 tune: bool = True) -> tuple:
    """
    Trains the candidate models on an already-split training partition.

    This used to take the full X and y and split internally. The caller had
    already produced a leak-free split, then stacked the two halves back together
    for this function to re-split - and because train_test_split permutes
    positions, the second split was NOT the same partition, so rows the
    preprocessor had been fitted on ended up in the evaluation set.

    Returns (fitted_models, training_info) where training_info records the
    hyperparameters chosen and whether tuning ran.
    """
    n_samples = len(y_train)
    cv = make_cv(y_train, problem_type)
    scoring = scoring_metric(problem_type)

    should_tune = bool(
        tune and cv is not None
        and MIN_ROWS_FOR_TUNING <= n_samples <= MAX_ROWS_FOR_TUNING
    )

    info = {
        "tuned": should_tune,
        "cv_folds": getattr(cv, "n_splits", None),
        "scoring": scoring,
        "best_params": {},
        "notes": [],
    }
    if not should_tune:
        if cv is None:
            info["notes"].append(
                f"Only {n_samples} training rows - cross-validation and tuning were skipped."
            )
        elif n_samples > MAX_ROWS_FOR_TUNING:
            info["notes"].append(
                f"{n_samples} training rows exceeds the {MAX_ROWS_FOR_TUNING} row tuning limit; "
                f"library defaults were used."
            )
        else:
            info["notes"].append("Dataset too small for a hyperparameter search; defaults were used.")

    if str(problem_type).lower().startswith("class"):
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            info["notes"].append("Only one class present in the training split; models cannot discriminate.")

    candidates, algorithm_reasons = select_candidate_algorithms(problem_type, X_train, y_train)
    info["selected_algorithms"] = list(candidates.keys())
    info["algorithm_reasons"] = algorithm_reasons
    if n_samples < 50:
        info["notes"].append(
            f"Small dataset warning: Only {n_samples} training rows available. "
            f"Cross-validation was adapted to {getattr(cv, 'n_splits', 'no')} folds; "
            "metrics may have higher variance."
        )

    fitted_models = {}
    for name, (estimator, grid) in candidates.items():
        try:
            if should_tune and grid:
                search = GridSearchCV(estimator, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
                search.fit(X_train, y_train)
                fitted_models[name] = search.best_estimator_
                info["best_params"][name] = search.best_params_
            else:
                estimator.fit(X_train, y_train)
                fitted_models[name] = estimator
                info["best_params"][name] = "library defaults"
        except Exception as e:
            print(f"Error training model '{name}': {str(e)}")
            info["notes"].append(f"{name} could not be trained: {e}")

    return fitted_models, info


def cross_validate_models(models: dict, X_train: np.ndarray, y_train: np.ndarray,
                          problem_type: str) -> dict:
    """Cross-validated score per model on the training partition.

    A single 80/20 split on a small dataset produces metric differences that are
    pure noise; selecting a "best model" on a third-decimal gap between two
    holdout scores is not defensible. The mean and standard deviation across
    folds give the selector something to compare honestly.
    """
    cv = make_cv(y_train, problem_type)
    scoring = scoring_metric(problem_type)
    results = {}
    if cv is None:
        return results

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            scores = np.asarray(scores, dtype=float)
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                continue
            results[name] = {
                "metric": scoring,
                "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
                "folds": int(len(scores)),
            }
        except Exception as e:
            print(f"Cross-validation failed for '{name}': {e}")
    return results
