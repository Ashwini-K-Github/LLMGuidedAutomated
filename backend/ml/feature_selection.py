import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    f_regression, f_classif, mutual_info_regression, mutual_info_classif,
)


def _mutual_information(X: np.ndarray, y: np.ndarray, discrete_mask: list,
                        is_classification: bool) -> tuple:
    """Compute one MI score per source feature without allowing one noisy
    estimator draw to dominate the ranking.

    Classification MI is averaged over several deterministic seeds. This is
    important for small/medium samples because the k-nearest-neighbour MI
    estimator can otherwise give a random numeric column (for example an
    unrelated Semester column) an implausibly large one-run score. Each source
    feature is scored independently so a high-cardinality categorical cannot
    make the whole ranking fall back to a linear F-test.
    """
    n_features = X.shape[1]
    estimator = mutual_info_classif if is_classification else mutual_info_regression
    target = y.astype(int) if is_classification else y.astype(float)
    n_samples = max(1, X.shape[0])
    n_neighbors = min(5, max(2, n_samples - 1))
    seeds = (11, 22, 33, 44, 55) if is_classification else (42,)

    scores = np.zeros(n_features, dtype=float)
    degraded = []
    for i in range(n_features):
        column = X[:, i:i + 1]
        values = []
        for seed in seeds:
            scored = False
            for as_discrete in (bool(discrete_mask[i]), False):
                try:
                    value = float(
                        estimator(
                            column, target,
                            discrete_features=as_discrete,
                            n_neighbors=n_neighbors,
                            random_state=seed,
                        )[0]
                    )
                    if np.isfinite(value):
                        values.append(max(0.0, value))
                    scored = True
                    break
                except Exception:
                    continue
            if not scored and bool(discrete_mask[i]):
                degraded.append(i)
        scores[i] = float(np.mean(values)) if values else 0.0
    return scores, sorted(set(degraded))


# A feature is kept when it is statistically significant after FDR correction,
# or when it carries a meaningful share of the strongest feature's information.
DEFAULT_ALPHA = 0.05
RELATIVE_MI_FLOOR_REGRESSION = 0.10
RELATIVE_MI_FLOOR_CLASSIFICATION = 0.30
# Only guarantees we never train on an empty matrix; it is not a target count.
MIN_FEATURES_KEPT = 1
# Above this share of the target's own entropy / variance a feature is almost
# certainly a restatement of the target rather than a predictor of it.
LEAKAGE_THRESHOLD = 0.95


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """FDR-adjusted q-values. Controls false positives across many features."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    finite = np.where(np.isfinite(p), p, 1.0)
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _entropy(y: np.ndarray) -> float:
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def select_features(X_train: np.ndarray, y_train: np.ndarray, feature_names: list,
                    problem_type: str, discrete_mask: list = None,
                    alpha: float = DEFAULT_ALPHA,
                    min_keep: int = MIN_FEATURES_KEPT) -> dict:
    """
    Ranks and selects features on the training split.

    Mutual information is the primary criterion. The previous implementation used
    Pearson correlation for regression and an ANOVA F-test for classification,
    both computed on alphabetically label-encoded categoricals. Pearson only sees
    linear monotonic association, so a category with a strong non-monotonic
    effect scored near zero and could be discarded before any model saw it.
    Mutual information detects non-linear and categorical relationships and is
    invariant to how a discrete variable is coded.

    Selection is by statistical significance (Benjamini-Hochberg corrected
    p-values) rather than the old rule of "score >= 0.05 after dividing by the
    maximum", which made a feature's fate depend on how strong an unrelated
    feature happened to be.
    """
    n_features = X_train.shape[1] if X_train is not None and X_train.ndim == 2 else 0
    if n_features == 0 or feature_names is None or len(feature_names) == 0:
        return {
            "method": "None", "importances": {}, "details": {},
            "selected": [], "removed": [], "leakage_warnings": [],
            "criterion": "no features available",
        }

    is_classification = str(problem_type).lower().startswith("class")
    X = np.nan_to_num(np.asarray(X_train, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray(y_train)

    if discrete_mask is None:
        discrete_mask = [False] * n_features
    discrete_mask = list(discrete_mask)[:n_features]
    discrete_mask += [False] * (n_features - len(discrete_mask))

    # ---- primary criterion: mutual information --------------------------------
    method = ("Mutual Information (classification)" if is_classification
              else "Mutual Information (regression)")
    degraded_features = []
    try:
        mi, degraded_indices = _mutual_information(X, y, discrete_mask, is_classification)
        degraded_features = [feature_names[i] for i in degraded_indices if i < len(feature_names)]
    except Exception as exc:
        print(f"Mutual information failed ({exc}); falling back to univariate F-test.")
        method = "Univariate F-test (fallback)"
        try:
            scores, _ = (f_classif(X, y) if is_classification else f_regression(X, y))
            mi = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as exc2:
            print(f"F-test fallback also failed ({exc2}); using variance.")
            method = "Feature Variance (last-resort fallback)"
            mi = np.nan_to_num(np.var(X, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    mi = np.nan_to_num(np.asarray(mi, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    mi = np.clip(mi, 0.0, None)

    # ---- secondary criterion: significance ------------------------------------
    try:
        with np.errstate(all="ignore"):
            _, p_values = (f_classif(X, y) if is_classification else f_regression(X, y))
        p_values = np.nan_to_num(np.asarray(p_values, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    except Exception:
        p_values = np.ones(n_features, dtype=float)
    q_values = _benjamini_hochberg(p_values)

    max_mi = float(np.max(mi)) if n_features else 0.0
    normalized = (mi / max_mi) if max_mi > 0 else np.zeros(n_features)
    relative_floor = (RELATIVE_MI_FLOOR_CLASSIFICATION if is_classification
                      else RELATIVE_MI_FLOOR_REGRESSION)

    # ---- leakage detection -----------------------------------------------------
    leakage = []
    if is_classification:
        target_entropy = _entropy(y)
        if target_entropy > 0:
            for i, name in enumerate(feature_names):
                if mi[i] / target_entropy >= LEAKAGE_THRESHOLD:
                    leakage.append({
                        "feature": name,
                        "reason": f"carries {mi[i] / target_entropy:.0%} of the target's information - "
                                  f"check this is not a restatement of the target",
                    })
    else:
        for i, name in enumerate(feature_names):
            col = X[:, i]
            if np.std(col) > 0 and np.std(y) > 0:
                corr = abs(float(np.corrcoef(col, y.astype(float))[0, 1]))
                if corr >= 0.999:
                    leakage.append({
                        "feature": name,
                        "reason": f"correlates {corr:.4f} with the target - "
                                  f"check this is not derived from the target",
                    })

    # ---- selection -------------------------------------------------------------
    keep = set()
    for i, name in enumerate(feature_names):
        if mi[i] <= 0 and q_values[i] >= alpha:
            continue
        if q_values[i] < alpha or normalized[i] >= relative_floor:
            keep.add(name)

    # Never train on nothing: keep the strongest few even if nothing is significant.
    if len(keep) < min(min_keep, n_features):
        for i in np.argsort(mi)[::-1]:
            keep.add(feature_names[i])
            if len(keep) >= min(min_keep, n_features):
                break

    order = np.argsort(mi)[::-1]
    importances, details = {}, {}
    for i in order:
        name = feature_names[i]
        importances[name] = round(float(normalized[i]), 3)
        is_significant = bool(q_values[i] < alpha)
        meets_mi_floor = bool(normalized[i] >= relative_floor)

        if name in keep:
            if meets_mi_floor:
                # Case 1: MI threshold passed (primary rule). Even if also statistically
                # significant, label as MI Threshold so the user understands the primary reason.
                selection_path = "mi_threshold"
                if is_significant:
                    reason = "Passed the MI threshold. The feature is also statistically significant after FDR correction."
                else:
                    reason = f"Passed the MI threshold (carries >= {relative_floor:.0%} of the strongest feature's mutual information)."
            elif is_significant:
                # Case 2: MI threshold failed, but rescued by statistical significance.
                selection_path = "significance"
                q_formatted = f"{q_values[i]:.6g}" if q_values[i] < 0.001 else f"{q_values[i]:.4g}"
                reason = (
                    f"Selected because FDR q-value ({q_formatted}) < {alpha} despite "
                    f"mutual information being below the {relative_floor:.0%} threshold."
                )
            else:
                selection_path = "minimum_set"
                reason = "Retained to maintain the minimum required feature set."
        else:
            # Case 3: Neither condition passed.
            selection_path = "not_selected"
            reason = "Did not meet the MI threshold and not statistically significant after FDR correction."

        details[name] = {
            "mutual_information": round(float(mi[i]), 4),
            "normalized_score": round(float(normalized[i]), 3),
            "p_value": round(float(p_values[i]), 5),
            "q_value": round(float(q_values[i]), 6),
            "selected": name in keep,
            "selection_path": selection_path,
            "reason": reason,
            "meets_mi_floor": meets_mi_floor,
            "is_significant": is_significant,
        }

    selected = [feature_names[i] for i in order if feature_names[i] in keep]
    removed = [feature_names[i] for i in order if feature_names[i] not in keep]

    return {
        "method": method,
        "importances": importances,
        "details": details,
        "selected": selected,
        "removed": removed,
        "leakage_warnings": leakage,
        "unscored_features": degraded_features,
        "criterion": (
            f"kept when Benjamini-Hochberg q < {alpha}, or mutual information >= "
            f"{relative_floor:.0%} of the strongest feature"
        ),
    }
