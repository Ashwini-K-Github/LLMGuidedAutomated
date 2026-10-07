import pandas as pd
import json
import re
from backend.utils.llm import call_llm_api
from backend.cleaning.profiler import tokenize_column_name


CLASSIFICATION_SEMANTIC_TYPES = ("Categorical", "Boolean", "Free Text", "Identifier")

# Matched as whole name tokens, never as substrings. Substring matching excluded
# any column whose name merely contained the letters "id" - Kidnapping, Humidity,
# Width, Residual, Incident_Count - or "name", such as Tournament. Those columns
# were dropped before feature selection ever saw them, which looks exactly like
# feature selection choosing the wrong attributes.
PII_TOKENS = {"name", "fullname", "firstname", "lastname", "surname", "email",
              "mail", "phone", "mobile", "contact", "ssn", "address", "aadhaar",
              "passport", "dob"}
PROTECTED_DEMOGRAPHIC_TOKENS = {"age", "gender", "sex"}
ID_TOKENS = {"id", "ids", "uuid", "guid", "pk", "key", "idx", "index", "roll",
             "regno", "serial", "sn", "ref", "identifier"}
ID_SUFFIXES = ("id", "no", "number", "code", "num", "key", "idx")

# Prediction intents whose labels must exist in the dataset.  These are kept
# deliberately narrow: Placement_Status is a valid placement target, but it is
# not evidence that a student passed an exam.
STRICT_TARGET_INTENTS = {
    "pass/fail": {
        "question_terms": ("pass", "fail"),
        "column_tokens": {"pass", "passed", "fail", "failed", "result", "outcome"},
        "value_tokens": {"pass", "passed", "fail", "failed"},
    },
    "placement": {
        "question_terms": ("placement", "placed"),
        "column_tokens": {"placement", "placed"},
        "value_tokens": {"placed", "unplaced"},
    },
}


def _is_excluded_column(col_name: str, semantic_type: str) -> bool:
    """True when a column is an identifier or personal data and should not be a feature."""
    if semantic_type == "Identifier":
        return True
    tokens = tokenize_column_name(col_name)
    if tokens & PII_TOKENS or tokens & ID_TOKENS or tokens & PROTECTED_DEMOGRAPHIC_TOKENS:
        return True
    glued = re.sub(r"[^a-z0-9]", "", str(col_name).lower())
    return any(glued.endswith(sfx) and len(glued) > len(sfx) + 2 for sfx in ID_SUFFIXES)


def exclusion_reason(col_name: str, semantic_type: str = None) -> str:
    """Human-readable reason a column was kept out of feature selection entirely.

    Used so the UI can show *why* a column such as Age or Student_ID never
    reached the feature-selection stage, instead of it simply disappearing -
    which otherwise looks indistinguishable from the column being dropped by
    accident.
    """
    tokens = tokenize_column_name(col_name)
    if tokens & PROTECTED_DEMOGRAPHIC_TOKENS:
        return "Excluded by project policy"
    if semantic_type == "Identifier" or (tokens & ID_TOKENS):
        return "Excluded as identifier"
    glued = re.sub(r"[^a-z0-9]", "", str(col_name).lower())
    if any(glued.endswith(sfx) and len(glued) > len(sfx) + 2 for sfx in ID_SUFFIXES):
        return "Excluded as identifier"
    if tokens & PII_TOKENS:
        return "Excluded as identifier"
    return "Excluded by project policy"



def _strict_target_intent(question: str):
    """Returns the explicitly requested label family, if the question has one."""
    words = set(re.findall(r"[a-z0-9]+", str(question).lower()))
    for intent, spec in STRICT_TARGET_INTENTS.items():
        if words.intersection(spec["question_terms"]):
            return intent
    return None


def _column_matches_intent(df: pd.DataFrame, column: str, intent: str) -> bool:
    """Checks column semantics and sampled values, never a generic word like status."""
    spec = STRICT_TARGET_INTENTS[intent]
    if tokenize_column_name(column).intersection(spec["column_tokens"]):
        return True
    values = df[column].dropna().astype(str).str.lower()
    value_words = set()
    for value in values.unique()[:100]:
        value_words.update(re.findall(r"[a-z]+", value))
    return bool(value_words.intersection(spec["value_tokens"]))


def _matching_intent_targets(df: pd.DataFrame, columns: list, intent: str) -> list:
    return [column for column in columns if _column_matches_intent(df, column, intent)]


def _choose_intent_target(question: str, candidates: list, df: pd.DataFrame = None):
    """Disambiguates related labels such as Placement_Training vs Placement_Status."""
    question_tokens = set(re.findall(r"[a-z0-9]+", str(question).lower()))

    def score_candidate(column):
        col_tokens = tokenize_column_name(column)
        score = 0
        # 1. Exact token match with question
        score += len(col_tokens.intersection(question_tokens)) * 10

        # 2. Substring / stem match (e.g. "placed" in question and "placement" in col_tokens)
        for q_tok in question_tokens:
            for c_tok in col_tokens:
                if len(q_tok) >= 4 and len(c_tok) >= 4:
                    if q_tok in c_tok or c_tok in q_tok or q_tok[:4] == c_tok[:4]:
                        score += 5

        # 3. Outcome / status indicators boost (e.g. status, result, outcome, target, decision)
        OUTCOME_TOKENS = {"status", "result", "outcome", "target", "decision", "label", "flag"}
        FEATURE_PENALTY_TOKENS = {"training", "attended", "course", "prep", "session", "program", "score", "hours"}
        if col_tokens & OUTCOME_TOKENS:
            score += 15
        if col_tokens & FEATURE_PENALTY_TOKENS:
            score -= 10

        # 4. If df is provided, check if column values match question words (e.g. "placed" in values)
        if df is not None and column in df.columns:
            try:
                sample_vals = df[column].dropna().astype(str).str.lower().unique()[:50]
                val_words = set()
                for v in sample_vals:
                    val_words.update(re.findall(r"[a-z0-9]+", v))
                if val_words & question_tokens:
                    score += 20
            except Exception:
                pass

        return score

    return max(
        candidates,
        key=lambda column: (
            score_candidate(column),
            -candidates.index(column),
        ),
    ) if candidates else None


def _find_available_targets(columns_profile: list) -> list:
    """Finds plausible prediction targets in the dataset (excluding IDs, PII, and sequence numbers like Semester)."""
    exclude_tokens = {
        "id", "email", "phone", "contact", "name", "ssn", "address", "gender", "sex", "age",
        "semester", "sem", "term", "year", "month", "day", "roll", "regno", "serial", "idx", "index"
    }
    candidates = []
    for col in columns_profile:
        name = col["name"]
        ctype = col.get("type", "")
        tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
        if ctype in ["Identifier", "Free Text", "Email", "Phone Number"] or (tokens & exclude_tokens):
            continue
        if ctype in ["Categorical", "Boolean"] or any(x in name.lower() for x in ["status", "placed", "placement", "result", "churn", "target"]):
            candidates.insert(0, name)
        elif ctype == "Numeric":
            candidates.append(name)
    return candidates


def _target_unavailable(columns: list, question: str, intent: str = None, available_targets: list = None) -> dict:
    requested = "Pass/Fail" if intent == "pass/fail" else (intent.replace("_", " ").title() if intent else "target")
    reason = f"The requested {requested} target is not available in this dataset. Please provide a suitable target column or choose an available prediction objective."
    if available_targets:
        targets_str = ", ".join(available_targets[:3])
        reason += f" Available prediction targets: {targets_str}."
    return {
        "target_available": False,
        "target": None,
        "features": [],
        "excluded": list(columns),
        "problem_type": None,
        "objective_summary": reason,
        "unavailable_reason": reason,
    }
# A numeric target with no more distinct values than this, and only whole
# numbers, is treated as a set of class labels when the caller asks for it.
MAX_DISCRETE_NUMERIC_CLASSES = 20


def resolve_problem_type(target_series, semantic_type: str = None, requested: str = None) -> tuple:
    """
    Decides Classification vs Regression from the target itself.

    Returns (problem_type, reason). The caller's request is honoured only when it
    is compatible with the data: asking for Classification on a continuous target
    used to be accepted, at which point every classifier raised
    "Unknown label type: continuous", all five models were skipped, and the UI
    displayed a best model of "None" with empty metrics.
    """
    requested_norm = (requested or "").strip().lower()
    values = target_series.dropna()
    n_unique = int(values.nunique())
    name = getattr(target_series, "name", "target")

    if n_unique <= 1:
        return "Classification", (
            f"Target '{name}' has {n_unique} distinct value(s); it carries no signal to model."
        )

    if semantic_type in CLASSIFICATION_SEMANTIC_TYPES:
        return "Classification", (
            f"Target '{name}' is {semantic_type} with {n_unique} distinct values, so this is a "
            f"classification task."
        )

    if pd.api.types.is_numeric_dtype(values):
        try:
            numeric = pd.to_numeric(values, errors="coerce").dropna()
            integer_like = bool(not numeric.empty and (numeric % 1 == 0).all())
        except Exception:
            integer_like = False

        if n_unique == 2:
            return "Classification", (
                f"Target '{name}' is numeric but takes only 2 distinct values, so it is treated as "
                f"a binary classification label."
            )
        if requested_norm.startswith("class") and integer_like and n_unique <= MAX_DISCRETE_NUMERIC_CLASSES:
            return "Classification", (
                f"Target '{name}' holds {n_unique} distinct whole numbers, so the requested "
                f"classification task is applied to them as class labels."
            )
        if requested_norm.startswith("class"):
            return "Regression", (
                f"Classification was requested, but target '{name}' is continuous with {n_unique} "
                f"distinct values. Regression is used instead - classifiers cannot fit a continuous "
                f"target."
            )
        return "Regression", (
            f"Target '{name}' is numeric and continuous with {n_unique} distinct values, so this is "
            f"a regression task."
        )

    return "Classification", (
        f"Target '{name}' is non-numeric with {n_unique} distinct values, so this is a "
        f"classification task."
    )

def analyze_objective(df: pd.DataFrame, columns_profile: list, question: str, provider: str, api_key: str = None) -> dict:
    """
    Analyzes the user's question/objective to identify:
    - Target column
    - Selected feature columns
    - Excluded columns
    - Problem type (Classification / Regression)
    """
    col_names = [col["name"] for col in columns_profile]
    col_types = {col["name"]: col["type"] for col in columns_profile}
    strict_intent = _strict_target_intent(question)
    strict_targets = (
        _matching_intent_targets(df, col_names, strict_intent) if strict_intent else []
    )
    strict_target = _choose_intent_target(question, strict_targets, df=df)
    allow_date_features = "forecast" in question.lower()

    available_targets = _find_available_targets(columns_profile)

    # Stop before asking an LLM or running feature selection.  A model cannot
    # learn a label that is not present, and guessing a nearby column changes the
    # user's question rather than answering it.
    if strict_intent and not strict_targets:
        return _target_unavailable(col_names, question, strict_intent, available_targets=available_targets)
    
    use_llm = provider.lower() != "mock (default)" and api_key is not None and len(api_key.strip()) > 0
    if not use_llm:
        if provider.lower() != "mock (default)":
            import os
            if "openai" in provider.lower() and os.getenv("OPENAI_API_KEY"):
                use_llm = True
                api_key = os.getenv("OPENAI_API_KEY")
            elif "anthropic" in provider.lower() and os.getenv("ANTHROPIC_API_KEY"):
                use_llm = True
                api_key = os.getenv("ANTHROPIC_API_KEY")

    if use_llm:
        try:
            system_prompt = (
                "You are a helpful data science assistant. Given a list of dataset columns and their types, "
                "analyze the user's objective/question to identify the single best target column for modeling, "
                "the list of feature columns to use for prediction (exclude identifiers and PII columns like Name, Email, Phone, StudentID unless absolutely necessary), "
                "and the machine learning problem type ('Classification' or 'Regression'). "
                "Return a strict JSON object with keys:\n"
                "- 'target': string (the exact column name)\n"
                "- 'features': list of strings (exact column names)\n"
                "- 'excluded': list of strings (exact column names)\n"
                "- 'problem_type': string ('Classification' or 'Regression')\n"
                "- 'objective_summary': string (brief summary of intent)"
            )
            
            columns_info = []
            for col in columns_profile:
                columns_info.append(f"- Name: {col['name']}, Type: {col['type']}, Unique values: {col['unique_count']}")
                
            user_prompt = (
                f"Dataset columns:\n" + "\n".join(columns_info) + "\n\n"
                f"User Objective: \"{question}\"\n\n"
                f"Provide target and features analysis in the requested JSON format."
            )
            
            # pyrefly: ignore [missing-import]
            from openai import OpenAI
            # pyrefly: ignore [missing-import]
            from anthropic import Anthropic
            
            res_dict = None
            if "openai" in provider.lower():
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                res_dict = json.loads(response.choices[0].message.content)
            elif "anthropic" in provider.lower():
                client = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.1
                )
                content = response.content[0].text
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
                if json_match:
                    content = json_match.group(1)
                res_dict = json.loads(content)
                
            if res_dict and "target" in res_dict and "features" in res_dict:
                # Sanity check columns exist
                target = res_dict["target"]
                features = res_dict["features"]
                
                # Make sure the target is valid, otherwise fallback
                if target in col_names and (
                    not strict_intent or target == strict_target
                ):
                    # Clean features to only include valid columns
                    valid_features = [
                        f for f in features
                        if f in col_names and f != target
                        and not _is_excluded_column(f, col_types.get(f))
                        and (col_types.get(f) != "Date" or allow_date_features)
                    ]
                    excluded = [c for c in col_names if c != target and c not in valid_features]
                    problem_type, _ = resolve_problem_type(
                        df[target], col_types.get(target), res_dict.get("problem_type")
                    )
                    
                    from backend.ml.model_trainer import select_candidate_algorithms
                    candidates_dict, algo_reasons = select_candidate_algorithms(problem_type, X_train=None, y_train=df[target].values, n_features=len(valid_features))
                    
                    excluded_by_policy = [
                        {"feature": c, "reason": exclusion_reason(c, col_types.get(c))}
                        for c in excluded
                    ]

                    return {
                        "target_available": True,
                        "target": target,
                        "features": valid_features,
                        "excluded": excluded,
                        "excluded_by_policy": excluded_by_policy,
                        "problem_type": problem_type,
                        "objective_summary": res_dict.get("objective_summary", f"Predict {target} using selected features."),
                        "candidate_algorithms": list(candidates_dict.keys()),
                        "algorithm_reasons": algo_reasons,
                    }
        except Exception as e:
            print(f"Error calling LLM in objective analysis: {str(e)}. Falling back to heuristics.")

    # Rule-Based Heuristic Parser
    # 1. Identify Target Column
    target = strict_target if strict_intent else None
    question_lower = question.lower()
    
    EXCLUDE_TARGET_TOKENS = {
        "id", "email", "phone", "contact", "name", "ssn", "address", "gender", "sex", "age",
        "semester", "sem", "term", "year", "month", "day", "roll", "regno", "serial", "idx", "index"
    }

    # Check for direct matches with non-excluded columns
    if not target:
        for col in col_names:
            col_lower = col.lower()
            tokens = set(re.findall(r"[a-z0-9]+", col_lower))
            if tokens & EXCLUDE_TARGET_TOKENS:
                continue
            if col_lower in question_lower or (len(col_lower) > 3 and col_lower.replace("_", " ") in question_lower):
                if col_types.get(col) != "Identifier":
                    target = col
                    break
                
    # If no target found, look for intent / keyword matches
    if not target:
        if any(w in question_lower for w in ["placement", "placed", "job_offer"]):
            candidates = [c for c in col_names if any(x in c.lower() for x in ["placement", "placed"])]
            if candidates:
                target = _choose_intent_target(question, candidates, df=df)
        elif any(w in question_lower for w in ["marks", "performance", "score", "gpa", "cgpa"]):
            candidates = [
                c for c in col_names
                if any(x in c.lower() for x in ["marks", "score", "grade", "gpa", "cgpa"])
                and not any(x in c.lower() for x in ["semester", "sem", "term", "year"])
            ]
            if candidates:
                target = candidates[0]
        elif any(w in question_lower for w in ["pass", "fail"]):
            candidates = strict_targets or [
                c for c in col_names
                if tokenize_column_name(c).intersection({"pass", "fail", "outcome", "result"})
            ]
            if candidates:
                target = candidates[0]
        elif any(w in question_lower for w in ["salary", "wage", "income", "pay"]):
            candidates = [c for c in col_names if any(x in c.lower() for x in ["salary", "wage", "income", "pay", "earn"])]
            if candidates:
                target = candidates[0]
        elif any(w in question_lower for w in ["active", "employment_status"]):
            candidates = [c for c in col_names if any(x in c.lower() for x in ["active", "employment_status"])]
            if candidates:
                target = candidates[0]
        elif any(w in question_lower for w in ["sales", "revenue", "spend", "price"]):
            candidates = [c for c in col_names if any(x in c.lower() for x in ["sales", "revenue", "spend", "amount", "price"])]
            if candidates:
                target = candidates[0]
        elif any(w in question_lower for w in ["churn", "attrition", "exit", "leave"]):
            candidates = [c for c in col_names if any(x in c.lower() for x in ["churn", "attrition", "exit", "leave"])]
            if candidates:
                target = candidates[0]

    # If no target found, do NOT arbitrarily guess an unrelated column like Semester!
    # Instead, inform the user clearly that the requested target cannot be predicted.
    if not target:
        q_words = re.findall(r"[a-z0-9]+", question_lower)
        predict_idx = -1
        for idx, w in enumerate(q_words):
            if w in ["predict", "classify", "forecast", "determine", "identify", "find"]:
                predict_idx = idx
                break
        unknown_intent = None
        if predict_idx != -1 and predict_idx + 1 < len(q_words):
            stop_words = {"whether", "a", "an", "the", "if", "student", "employee", "customer", "user", "will", "be", "is", "can", "we"}
            intent_words = [w for w in q_words[predict_idx + 1:] if w not in stop_words]
            if intent_words:
                unknown_intent = " ".join(intent_words[:2]).capitalize()
        if not unknown_intent:
            for strict_key in STRICT_TARGET_INTENTS:
                if any(t in question_lower for t in STRICT_TARGET_INTENTS[strict_key]["question_terms"]):
                    unknown_intent = strict_key
                    break
        return _target_unavailable(col_names, question, unknown_intent or "The requested target", available_targets=available_targets)

    # 2. Determine Problem Type from the target itself
    target_type = col_types.get(target, "Numeric")
    problem_type, _ = resolve_problem_type(df[target], target_type)
        
    # 3. Features & Excluded Columns Identification
    features = []
    excluded = []
    
    for col in col_names:
        if col == target:
            continue
            
        ctype = col_types[col]

        # Exclude identifiers and personal data, matched on whole name tokens.
        if _is_excluded_column(col, ctype) or (ctype == "Date" and not allow_date_features):
            excluded.append(col)
        else:
            features.append(col)
            
    from backend.ml.model_trainer import select_candidate_algorithms
    candidates_dict, algo_reasons = select_candidate_algorithms(problem_type, X_train=None, y_train=df[target].values, n_features=len(features))
    objective_summary = f"Predict '{target}' based on user query '{question}'."

    excluded_by_policy = [
        {"feature": col, "reason": exclusion_reason(col, col_types.get(col))}
        for col in excluded
    ]

    return {
        "target_available": True,
        "target": target,
        "features": features,
        "excluded": excluded,
        "excluded_by_policy": excluded_by_policy,
        "problem_type": problem_type,
        "objective_summary": objective_summary,
        "candidate_algorithms": list(candidates_dict.keys()),
        "algorithm_reasons": algo_reasons,
    }
