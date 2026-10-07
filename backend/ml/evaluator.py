import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, balanced_accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

# Adjusted R2 is meaningless once the feature count approaches the sample count.
MIN_SAMPLES_PER_FEATURE_FOR_ADJ_R2 = 2


def _clean(value, digits=3):
    """Rounds to `digits`, or returns None for NaN/inf rather than a fake number."""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(num) or np.isinf(num):
        return None
    return round(num, digits)


def choose_positive_class(y_train: np.ndarray, y_test: np.ndarray) -> int:
    """Picks the minority class as the positive label for binary metrics.

    Binary precision/recall/F1 default to pos_label=1, and class 1 was whichever
    label sorted second alphabetically. On a churn dataset that made 'Retained'
    the positive class, so the reported recall described the majority class
    nobody is trying to detect.

    The label must exist in y_test, otherwise scikit-learn rejects it. Class
    frequencies are taken from the training labels where available, since the
    test split is the smaller and noisier sample.
    """
    test_labels = np.unique(y_test)
    if len(test_labels) == 0:
        return 1

    counts_by_label = {}
    reference = y_train if y_train is not None and len(y_train) else y_test
    labels, counts = np.unique(reference, return_counts=True)
    for label, count in zip(labels, counts):
        counts_by_label[label] = count

    # Only labels actually present in the test split are eligible.
    candidates = [l for l in test_labels]
    return int(min(candidates, key=lambda l: counts_by_label.get(l, 0)))


# FeatureEngineer encodes a target label it never saw during fit as -1, which
# happens when a rare class lands only in the test split. Naming it makes the
# per-class table readable instead of showing a bare "-1".
UNSEEN_CLASS_LABEL = "(class unseen in training)"


def _label_name(code, class_names):
    try:
        index = int(code)
    except (TypeError, ValueError):
        return str(code)
    if index < 0:
        return UNSEEN_CLASS_LABEL
    if class_names and index < len(class_names):
        return str(class_names[index])
    return str(code)


def evaluate_classification_model(model, X_test: np.ndarray, y_test: np.ndarray,
                                  y_train: np.ndarray = None, class_names: list = None) -> dict:
    """Classification metrics, reported per class as well as averaged."""
    y_pred = model.predict(X_test)

    y_probs = None
    if hasattr(model, "predict_proba"):
        try:
            y_probs = model.predict_proba(X_test)
        except Exception as e:
            print(f"Failed to get predict_proba: {e}")

    observed = np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)]))

    # The number of classes is a property of the problem, not of the test split.
    # A small test split can hold two of four classes; evaluating that as a
    # binary problem produces metrics that describe something else.
    if y_train is not None and len(y_train):
        all_labels = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test)]))
    else:
        all_labels = np.unique(y_test)
    is_binary = len(all_labels) == 2

    acc = _clean(accuracy_score(y_test, y_pred))
    balanced = _clean(balanced_accuracy_score(y_test, y_pred))

    # Macro averages treat every class equally, so a minority class cannot be
    # hidden behind a large majority class.
    prec_macro = _clean(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = _clean(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = _clean(f1_score(y_test, y_pred, average="macro", zero_division=0))

    result = {
        "accuracy": acc,
        "balanced_accuracy": balanced,
        "precision": prec_macro,
        "recall": rec_macro,
        "f1_score": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "is_binary": is_binary,
        "averaging": "macro",
    }

    # ROC-AUC. None means "could not be computed" - it used to default to 0.5 and
    # be reported as if it had been measured.
    roc_auc = None
    if y_probs is not None:
        try:
            if is_binary and y_probs.shape[1] == 2 and len(np.unique(y_test)) == 2:
                pos = choose_positive_class(y_train, y_test)
                classes = list(getattr(model, "classes_", [0, 1]))
                col = classes.index(pos) if pos in classes else 1
                roc_auc = _clean(roc_auc_score((np.asarray(y_test) == pos).astype(int), y_probs[:, col]))
            elif not is_binary and y_probs.shape[1] == len(np.unique(y_test)):
                roc_auc = _clean(roc_auc_score(y_test, y_probs, multi_class="ovr", average="macro"))
        except Exception as e:
            print(f"ROC-AUC could not be computed: {e}")
    result["roc_auc"] = roc_auc
    result["roc_auc_available"] = roc_auc is not None

    # Binary metrics for an explicitly named positive class
    if is_binary:
        pos = choose_positive_class(y_train, y_test)
        neg_candidates = [l for l in all_labels if l != pos]
        neg = neg_candidates[0] if neg_candidates else pos
        result["positive_class"] = _label_name(pos, class_names)
        try:
            result["precision_positive"] = _clean(precision_score(y_test, y_pred, pos_label=pos, zero_division=0))
            result["recall_positive"] = _clean(recall_score(y_test, y_pred, pos_label=pos, zero_division=0))
            result["f1_positive"] = _clean(f1_score(y_test, y_pred, pos_label=pos, zero_division=0))
        except Exception as e:
            print(f"Positive-class metrics unavailable: {e}")
            result["precision_positive"] = result["recall_positive"] = result["f1_positive"] = None

        cm_full = confusion_matrix(y_test, y_pred, labels=[neg, pos])
        tn, fp, fn, tp = cm_full.ravel() if cm_full.size == 4 else (0, 0, 0, 0)
        result["confusion_matrix"] = {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "positive_class": _label_name(pos, class_names),
            "negative_class": _label_name(neg, class_names),
        }
    else:
        cm = confusion_matrix(y_test, y_pred, labels=list(observed))
        result["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "labels": [_label_name(c, class_names) for c in observed],
        }

    # Per-class breakdown
    per_class = {}
    for code in np.unique(y_test):
        per_class[_label_name(code, class_names)] = {
            "support": int(np.sum(np.asarray(y_test) == code)),
            "precision": _clean(precision_score(y_test, y_pred, labels=[code], average="macro", zero_division=0)),
            "recall": _clean(recall_score(y_test, y_pred, labels=[code], average="macro", zero_division=0)),
            "f1": _clean(f1_score(y_test, y_pred, labels=[code], average="macro", zero_division=0)),
        }
    result["per_class"] = per_class
    return result


def evaluate_regression_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Regression metrics. Adjusted R2 is omitted when it cannot be trusted."""
    y_pred = model.predict(X_test)

    mae = _clean(mean_absolute_error(y_test, y_pred))
    mse = _clean(mean_squared_error(y_test, y_pred))
    rmse = _clean(np.sqrt(mean_squared_error(y_test, y_pred)))

    try:
        r2 = _clean(r2_score(y_test, y_pred))
    except Exception as e:
        print(f"R2 score calculation error on small test split: {e}")
        r2 = None

    n = len(y_test)
    p = X_test.shape[1] if X_test.ndim == 2 else 1
    adj_r2 = None
    adj_note = None
    if r2 is not None and n - p - 1 > 0 and n >= p * MIN_SAMPLES_PER_FEATURE_FOR_ADJ_R2:
        adj_r2 = _clean(1 - (1 - r2) * (n - 1) / (n - p - 1))
    else:
        adj_note = (
            f"Adjusted R2 is not reported: {n} test rows against {p} features is too few "
            f"for the correction to be meaningful."
        )

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2,
        "adjusted_r2": adj_r2,
        "adjusted_r2_note": adj_note,
        "test_rows": int(n),
        "n_features": int(p),
    }


def evaluate_all_models(models: dict, X_test: np.ndarray, y_test: np.ndarray, problem_type: str,
                        X_train: np.ndarray = None, y_train: np.ndarray = None,
                        class_names: list = None) -> dict:
    """Evaluates every fitted model on the held-out test partition."""
    evaluations = {}
    is_clf = str(problem_type).lower().startswith("class")

    for name, model in models.items():
        try:
            if is_clf:
                evaluations[name] = evaluate_classification_model(
                    model, X_test, y_test, y_train=y_train, class_names=class_names
                )
            else:
                evaluations[name] = evaluate_regression_model(model, X_test, y_test)
        except Exception as e:
            print(f"Error evaluating model '{name}': {str(e)}")

    return evaluations
