# Complexity is used only as a deterministic tie-breaker when two models have
# exactly the same ranking score; it must never override a higher CV mean.
COMPLEXITY_RANK = {
    "Linear Regression": 0,
    "Logistic Regression": 0,
    "Gaussian Naive Bayes": 1,
    "Ridge Regression": 1,
    "Lasso Regression": 1,
    "Decision Tree": 2,
    "KNN": 3,
    "SVM": 4,
    "SVR": 4,
    "Random Forest": 5,
    "Extra Trees": 5,
    "Gradient Boosting": 6,
}

# A model has to clear the strongest trivial baseline by this margin *plus* its
# own cross-validation spread before it is described as useful. A flat margin on
# a single holdout split is itself noisy: on a 120-row test set an unusable model
# cleared 0.02 by luck while its fold-to-fold spread showed no real difference.
MEANINGFUL_MARGIN = 0.02


def _holdout_score(metrics: dict, is_classification: bool):
    if not metrics:
        return None
    value = metrics.get("f1_macro", metrics.get("f1_score")) if is_classification else metrics.get("r2_score")
    return None if value is None else float(value)


def select_best_model(evaluations: dict, problem_type: str, target_name: str,
                      cv_results: dict = None, baseline_metrics: dict = None,
                      training_info: dict = None) -> dict:
    """
    Picks the recommended model and explains the choice from the evidence.

    Selection uses the highest cross-validated score where available. Close
    alternatives are reported separately; overlap of mean +/- standard deviation
    is not treated as proof of statistical equivalence.

    If all models fail to meaningfully beat the baseline, reports 'No reliable model found'
    instead of recommending an uninformative model.
    """
    is_classification = str(problem_type).lower().startswith("class")
    cv_results = cv_results or {}
    training_info = training_info or {}
    warnings = []

    if not evaluations:
        return {
            "best_model": None,
            "recommended_model": "No reliable model found",
            "display_name": "No reliable model found",
            "is_reliable": False,
            "metrics": {},
            "cv": None,
            "baseline": baseline_metrics or {},
            "reason": "No model could be trained and evaluated on this dataset, so no recommendation "
                      "can be made. Check the target column and the selected features.",
            "beats_baseline": False,
            "tied_models": [],
            "warnings": ["No models were evaluated."],
        }

    # Rank on cross-validated score when we have one; otherwise the holdout score.
    ranking_basis = "cross-validated " + (
        cv_results[next(iter(cv_results))]["metric"] if cv_results else ""
    ) if cv_results else ("holdout F1 (macro)" if is_classification else "holdout R²")

    scores = {}
    for name, metrics in evaluations.items():
        if name in cv_results:
            scores[name] = cv_results[name]["mean"]
        else:
            fallback = _holdout_score(metrics, is_classification)
            if fallback is not None:
                scores[name] = fallback
    if not scores:
        scores = {name: 0.0 for name in evaluations}

    best_model_name = max(scores, key=lambda n: (scores[n], -COMPLEXITY_RANK.get(n, 99)))
    best_score = scores[best_model_name]
    best_std = cv_results.get(best_model_name, {}).get("std", 0.0)

    # Record close alternatives, but do NOT replace the numeric leader merely
    # because score ranges overlap. The highest mean CV score remains the winner.
    #
    # "Overlap" is a property of BOTH models' mean +/- std ranges, not just the
    # winner's spread: comparing only against best_std missed candidates whose
    # own fold-to-fold variance was what made their range touch the winner's
    # (e.g. Random Forest at 0.554 +/- 0.032 vs a winner at 0.591 +/- 0.036 -
    # the 0.037 gap clears best_std alone but not the combined spread).
    def _candidate_std(name):
        return cv_results.get(name, {}).get("std", 0.0) or 0.0

    tied = [n for n, s in scores.items() if n != best_model_name and
            abs(s - best_score) <= (best_std + _candidate_std(n))]

    best_metrics = evaluations.get(best_model_name, {})
    best_cv = cv_results.get(best_model_name)

    # Baseline comparison, on cross-validated scores where available so the
    # verdict does not hinge on one lucky split.
    baseline_metrics = baseline_metrics or {}
    baseline_cv = baseline_metrics.get("cv") or {}
    baseline_strategy = baseline_metrics.get("strategy", "trivial baseline")

    baseline_score = baseline_cv.get("mean")
    model_score = best_cv["mean"] if best_cv else None
    comparison_basis = "cross-validated"
    if baseline_score is None or model_score is None:
        baseline_score = _holdout_score(baseline_metrics, is_classification)
        model_score = _holdout_score(best_metrics, is_classification)
        comparison_basis = "held-out"

    beats_baseline = True
    delta = 0.0
    required = MEANINGFUL_MARGIN + (best_std or 0.0)
    metric_label = "F1 (macro)" if is_classification else "R²"

    if baseline_score is not None and model_score is not None:
        delta = model_score - baseline_score
        beats_baseline = delta >= required

    # Classification test details
    if is_classification:
        detail = (
            f"On the held-out test split: accuracy {best_metrics.get('accuracy')}, "
            f"balanced accuracy {best_metrics.get('balanced_accuracy')}, "
            f"macro F1 {best_metrics.get('f1_macro')}"
        )
        if best_metrics.get("roc_auc") is not None:
            detail += f", ROC-AUC {best_metrics.get('roc_auc')}"
        else:
            detail += ", ROC-AUC not available for this model"
        if best_metrics.get("positive_class"):
            detail += f" (positive class: '{best_metrics['positive_class']}')"
    else:
        detail = (
            f"On the held-out test split: R² {best_metrics.get('r2_score')}, "
            f"RMSE {best_metrics.get('rmse')}, MAE {best_metrics.get('mae')}"
        )
        if best_metrics.get("adjusted_r2") is not None:
            detail += f", adjusted R² {best_metrics.get('adjusted_r2')}"

    # Build justification evidence
    if best_cv:
        evidence = (
            f"Across {best_cv['folds']}-fold cross-validation on the training split it scored "
            f"{best_cv['mean']:.3f} +/- {best_cv['std']:.3f} {best_cv['metric']}"
        )
    else:
        evidence = (
            f"It scored {best_score:.3f} on the holdout split (cross-validation was not possible "
            f"on a dataset this small)"
        )
        warnings.append("Scores come from a single split; treat small differences as noise.")

    tuning_sentence = ""
    if training_info:
        if training_info.get("tuned"):
            params = training_info.get("best_params", {}).get(best_model_name)
            tuning_sentence = (
                f" Hyperparameters were selected by grid search over "
                f"{training_info.get('cv_folds')} folds"
                + (f" ({params})." if isinstance(params, dict) and params else ".")
            )
        else:
            tuning_sentence = " Hyperparameters were left at library defaults."
            for note in training_info.get("notes", []):
                warnings.append(note)

    # Suitability blurb from dynamic candidate algorithm reasons. When the
    # model is actually being recommended, lead with the evidence (it has the
    # highest cross-validated score) rather than the generic candidate-pool
    # description, which reads oddly once it's framed as merely "a baseline"
    # for the model that was just chosen.
    suitability_sentence = ""
    algo_reasons = training_info.get("algorithm_reasons", {})
    if best_model_name in algo_reasons:
        candidate_reason = algo_reasons[best_model_name]
        if beats_baseline:
            suitability_sentence = (
                f" Why suitable: {best_model_name} provides the highest {ranking_basis} "
                f"score among the candidates evaluated here. {candidate_reason}"
            )
        else:
            suitability_sentence = f" Why suitable: {candidate_reason}"

    if beats_baseline:
        baseline_sentence = (
            f" It beats the strongest trivial baseline ({baseline_strategy}) by "
            f"{delta:+.3f} {metric_label} on {comparison_basis} scores "
            f"({model_score:.3f} against {baseline_score:.3f})."
        )
        comp_sentence = ""
        if tied:
            comp_sentence = (
                f" {', '.join(tied)} are close to the winner and their cross-validation score "
                f"ranges overlap with it; this evaluation does not provide strong evidence "
                f"that they are better. {best_model_name} remains the highest-scoring candidate."
            )
        else:
            others = [n for n in evaluations if n != best_model_name]
            if others:
                comp_sentence = f" It outperformed alternative models ({', '.join(others)}) on cross-validation."

        reason = (
            f"{best_model_name} is recommended for predicting '{target_name}'. "
            f"{evidence}, ranked on {ranking_basis}. {detail}."
            f"{baseline_sentence}{comp_sentence}{suitability_sentence}{tuning_sentence}"
        )
    else:
        # Honest reporting: No reliable model found!
        baseline_sentence = (
            f" However it does NOT meaningfully beat the strongest trivial baseline "
            f"({baseline_strategy}): {model_score:.3f} against {baseline_score:.3f} "
            f"{metric_label} on {comparison_basis} scores, a gap of {delta:+.3f} against the "
            f"{required:.3f} needed to clear the fold-to-fold variation. On this data the "
            f"selected features carry little usable signal about '{target_name}', so this "
            f"recommendation should not be relied on."
        )
        warnings.append("No reliable model found: Candidate models do not meaningfully outperform the trivial baseline.")
        reason = (
            f"No reliable model found for predicting '{target_name}'. "
            f"{evidence}, ranked on {ranking_basis}. {detail}."
            f"{baseline_sentence} Cocoon will not recommend a model simply because it has the highest score among poor models."
            f"{tuning_sentence}"
        )

    return {
        "best_model": best_model_name,
        "recommended_model": best_model_name if beats_baseline else "No reliable model found",
        "display_name": f"{best_model_name} (Best Recommended Model)" if beats_baseline else "No reliable model found",
        "is_reliable": beats_baseline,
        "metrics": best_metrics,
        "cv": best_cv,
        "baseline": baseline_metrics or {},
        "ranking_basis": ranking_basis,
        "reason": reason,
        "beats_baseline": beats_baseline,
        "tied_models": tied,
        "warnings": warnings,
    }
