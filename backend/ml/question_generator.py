import pandas as pd
import json
import re
from backend.utils.llm import call_llm_api

def generate_questions(df: pd.DataFrame, columns_profile: list, provider: str, api_key: str = None) -> list:
    """
    Generates dataset-specific questions using either the LLM or high-fidelity rule heuristics.
    """
    col_names = [col["name"] for col in columns_profile]
    col_types = {col["name"]: col["type"] for col in columns_profile}
    
    # 1. Check if LLM should be used
    use_llm = provider.lower() != "mock (default)" and api_key is not None and len(api_key.strip()) > 0
    if not use_llm:
        if provider.lower() != "mock (default)":
            # Check env keys as fallback
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
                "You are an expert data scientist. Given a dataset summary with columns and their detected types, "
                "generate a JSON list containing 5 to 7 meaningful machine learning or statistical prediction questions. "
                "The questions must be highly specific to the columns. For example, if there is a 'SemesterMarks' column, "
                "ask 'Can we predict semester marks?'. Do not use generic template questions. "
                "Every prediction question must name or unambiguously refer to a target column that actually exists. "
                "Never suggest pass/fail, churn, placement, active status, or another outcome unless that outcome label "
                "is present in the supplied columns. Only return a valid JSON list of strings."
            )
            columns_info = []
            for col in columns_profile:
                columns_info.append(f"- Name: {col['name']}, Type: {col['type']}, Unique values: {col['unique_count']}")
            
            user_prompt = f"Dataset columns:\n" + "\n".join(columns_info) + "\n\nJSON List of Questions:"
            
            # Simple API call
            # pyrefly: ignore [missing-import]
            from openai import OpenAI
            # pyrefly: ignore [missing-import]
            from anthropic import Anthropic
            
            questions = []
            if "openai" in provider.lower():
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                res_data = json.loads(response.choices[0].message.content)
                if isinstance(res_data, dict) and "questions" in res_data:
                    questions = res_data["questions"]
                elif isinstance(res_data, list):
                    questions = res_data
            elif "anthropic" in provider.lower():
                client = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.3
                )
                content = response.content[0].text
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
                if json_match:
                    content = json_match.group(1)
                res_data = json.loads(content)
                if isinstance(res_data, dict) and "questions" in res_data:
                    questions = res_data["questions"]
                elif isinstance(res_data, list):
                    questions = res_data
            
            if isinstance(questions, list) and len(questions) > 0:
                has_pass_fail = any("pass" in c["name"].lower() or "fail" in c["name"].lower() for c in columns_profile)
                has_placement = any("placement" in c["name"].lower() or "placed" in c["name"].lower() for c in columns_profile)
                valid_llm_qs = []
                for q in questions:
                    q_str = str(q)
                    q_lower = q_str.lower()
                    if ("pass" in q_lower or "fail" in q_lower) and not has_pass_fail:
                        continue
                    if ("placement" in q_lower or "placed" in q_lower) and not has_placement:
                        continue
                    valid_llm_qs.append(q_str)
                if valid_llm_qs:
                    return valid_llm_qs[:7]
        except Exception as e:
            print(f"Error calling LLM for question generation: {str(e)}. Falling back to heuristics.")
            
    # Heuristic Rule-Based fallback generator
    questions = []
    
    # 1. Detect Domain
    domain = "general"
    all_text = " ".join([c.lower() for c in col_names])
    if any(x in all_text for x in ["employee", "salary", "joining_date", "hr", "manager", "tenure", "wage", "hire", "job_title", "staff"]):
        domain = "employee"
    elif any(x in all_text for x in ["student", "gpa", "attendance", "marks", "grade", "exam", "semester", "assignment", "pass", "fail", "education"]):
        domain = "student"
    elif any(x in all_text for x in ["sale", "revenue", "price", "quantity", "product", "transaction", "order", "invoice", "store"]):
        domain = "sales"
    elif any(x in all_text for x in ["customer", "churn", "income", "spend", "purchas", "user", "visitor"]):
        domain = "customer"
        
    # 2. Identify potential targets
    num_targets = []
    cat_targets = []
    date_cols = []
    
    EXCLUDE_TARGET_TOKENS = {
        "id", "email", "phone", "contact", "name", "ssn", "address", "gender", "sex", "age",
        "semester", "sem", "term", "year", "month", "day", "roll", "regno", "serial", "idx", "index"
    }
    
    for col in columns_profile:
        name = col["name"]
        ctype = col["type"]
        name_lower = name.lower()
        tokens = set(re.findall(r"[a-z0-9]+", name_lower))
        
        # Exclude identifiers, email, phone, personal data, and sequence grouping columns
        if ctype in ["Identifier", "Free Text", "Email", "Phone Number"] or (tokens & EXCLUDE_TARGET_TOKENS):
            continue
            
        if ctype == "Numeric":
            num_targets.append(name)
        elif ctype in ["Categorical", "Boolean"]:
            cat_targets.append(name)
        elif ctype == "Date":
            date_cols.append(name)
            
    # 3. Generate Domain-Specific Objectives
    if domain == "employee":
        salary_cols = [c for c in num_targets if any(x in c.lower() for x in ["salary", "wage", "income", "pay"])]
        salary_col = salary_cols[0] if salary_cols else (num_targets[0] if num_targets else None)
        
        attrition_cols = [c for c in cat_targets if any(x in c.lower() for x in ["attrition", "churn", "exit", "leave"])]
        active_cols = [c for c in cat_targets if any(x in c.lower() for x in ["active", "employment_status"])]
        status_col = attrition_cols[0] if attrition_cols else (active_cols[0] if active_cols else None)
        
        dept_cols = [c for c in col_names if "dept" in c.lower() or "department" in c.lower()]
        dept_col = dept_cols[0] if dept_cols else None
        
        join_cols = [c for c in date_cols if "join" in c.lower() or "hire" in c.lower() or "start" in c.lower() or "date" in c.lower()]
        join_col = join_cols[0] if join_cols else (date_cols[0] if date_cols else None)
        
        if salary_col:
            questions.append(f"Can we predict employee {salary_col.lower().replace('_', ' ')} based on the available employee information?")
            questions.append(f"What factors are most strongly associated with employee {salary_col.lower().replace('_', ' ')}?")
        if status_col:
            if any(x in status_col.lower() for x in ["attrition", "churn", "exit", "leave"]):
                questions.append("Can we predict employee attrition?")
            else:
                questions.append(f"Can we predict employee {status_col.lower().replace('_', ' ')}?")
        if dept_col:
            questions.append(f"How are employees distributed across departments?")
            if salary_col:
                questions.append(f"How does department relate to employee {salary_col.lower().replace('_', ' ')}?")
        if join_col:
            questions.append(f"Can we analyze employee joining trends over time?")
            
    elif domain == "student":
        # Keep student suggestions focused on genuine outcomes available in the
        # dataset. Training, semester, and demographic columns are predictors,
        # not suggested targets, unless explicitly named as an outcome.
        placement_cols = [
            c for c in cat_targets
            if any(x in c.lower() for x in ["placement_status", "placed", "campus_placement"])
            and "training" not in c.lower()
        ]
        placement_col = placement_cols[0] if placement_cols else None

        marks_cols = [
            c for c in num_targets
            if any(x in c.lower() for x in ["marks", "score", "grade", "gpa", "cgpa"])
        ]
        marks_col = marks_cols[0] if marks_cols else (num_targets[0] if num_targets else None)

        pass_cols = [
            c for c in cat_targets
            if ("pass" in c.lower() or "fail" in c.lower() or "exam_result" in c.lower())
        ]
        pass_col = pass_cols[0] if pass_cols else None

        attn_cols = [c for c in col_names if "attendance" in c.lower() or "attn" in c.lower()]
        attn_col = attn_cols[0] if attn_cols else None

        if placement_col:
            questions.append("Can we predict whether a student will be placed?")
            questions.append("Which student attributes are most useful for predicting placement?")
        if marks_col:
            pretty_target = marks_col.lower().replace("_", " ")
            questions.append(f"Can we predict students' {pretty_target} using the available academic attributes?")
            questions.append(f"Which attributes are most useful for predicting {pretty_target}?")
            if attn_col:
                questions.append(f"Can attendance help predict students' {pretty_target}?")
        if pass_col:
            questions.append("Can we predict whether a student will pass or fail?")

    elif domain == "sales":
        sales_cols = [c for c in num_targets if any(x in c.lower() for x in ["sales", "revenue", "spend", "amount", "price"])]
        sales_col = sales_cols[0] if sales_cols else (num_targets[0] if num_targets else None)
        
        date_col = date_cols[0] if date_cols else None
        
        region_cols = [c for c in col_names if "region" in c.lower() or "country" in c.lower() or "state" in c.lower()]
        region_col = region_cols[0] if region_cols else None
        product_cols = [c for c in col_names if "product" in c.lower() or "item" in c.lower() or "category" in c.lower()]
        product_col = product_cols[0] if product_cols else None
        
        if sales_col:
            questions.append(f"Can we predict {sales_col.lower().replace('_', ' ')} using the available attributes?")
            questions.append(f"Which factors have the strongest relationship with {sales_col.lower().replace('_', ' ')}?")
            if date_col:
                questions.append(f"Can we forecast future {sales_col.lower().replace('_', ' ')} based on historical data? (Forecasting)")
            if region_col:
                questions.append(f"How do sales vary across regions?")
            if product_col:
                questions.append(f"Which products generate the highest {sales_col.lower().replace('_', ' ')}?")
                
    elif domain == "customer":
        churn_cols = [c for c in cat_targets if any(x in c.lower() for x in ["churn", "attrition", "retention"])]
        churn_col = churn_cols[0] if churn_cols else None
        
        spend_cols = [c for c in num_targets if any(x in c.lower() for x in ["spend", "charge", "amount", "income", "balance"])]
        spend_col = spend_cols[0] if spend_cols else (num_targets[0] if num_targets else None)
        
        if churn_col:
            questions.append("Can we predict customer churn?")
            questions.append("What customer attributes are most predictive of churn risk?")
        if spend_col:
            questions.append(f"Can we predict customer {spend_col.lower().replace('_', ' ')} based on demographics?")
            questions.append(f"What factors are most strongly associated with customer {spend_col.lower().replace('_', ' ')}?")
            
    else: # General Domain
        for col in col_names:
            ctype = col_types.get(col)
            tokens = set(re.findall(r"[a-z0-9]+", col.lower()))
            if ctype in ["Identifier", "Free Text", "Email", "Phone Number"] or (tokens & EXCLUDE_TARGET_TOKENS):
                continue
                
            if ctype == "Numeric":
                questions.append(f"Can we predict the values of {col} based on other columns?")
                questions.append(f"What factors have the strongest relationship with {col}?")
                break
            elif ctype in ["Categorical", "Boolean"]:
                questions.append(f"Can we predict the category of {col} using available data?")
                break
                
    # Fallback padding to make sure we always have 5-7 questions based on actual candidate targets
    # Prefer meaningful prediction targets when padding: outcome-like categorical
    # columns first, then continuous numeric columns. Avoid arbitrary dimensions
    # such as Department/Gender when a measurable outcome is available.
    priority_cat_targets = [
        c for c in cat_targets
        if not (set(re.findall(r"[a-z0-9]+", c.lower())) & EXCLUDE_TARGET_TOKENS)
        and (
            bool(set(re.findall(r"[a-z0-9]+", c.lower())) &
                 {"status", "result", "outcome", "placed", "churn", "attrition", "target", "label"})
            or c.lower().replace(" ", "_") in {"placement_status", "campus_placement_status"}
        )
    ]
    valid_numeric_targets = [
        c for c in num_targets
        if not (set(re.findall(r"[a-z0-9]+", c.lower())) & EXCLUDE_TARGET_TOKENS)
    ]
    remaining_cat_targets = [
        c for c in cat_targets
        if not (set(re.findall(r"[a-z0-9]+", c.lower())) & EXCLUDE_TARGET_TOKENS)
        and c not in priority_cat_targets
        and not (domain == "student" and "training" in c.lower())
    ]
    all_valid_targets = priority_cat_targets + valid_numeric_targets + remaining_cat_targets
    # Student-domain questions are already crafted around concrete outcomes
    # (placement and academic measures). Do not pad them with arbitrary
    # categorical targets such as Department or Placement_Training.
    if domain != "student" and len(questions) < 5 and all_valid_targets:
        for t in all_valid_targets:
            ctype = col_types.get(t)
            if ctype in ["Categorical", "Boolean"]:
                q = f"Can we predict {t.replace('_', ' ')} based on the available attributes?"
            else:
                q = f"Can we predict {t.replace('_', ' ')} using the other columns?"
            if q not in questions:
                questions.append(q)
            q_factors = f"What factors most strongly influence {t.replace('_', ' ')}?"
            if q_factors not in questions:
                questions.append(q_factors)
            if len(questions) >= 7:
                break
                
    # Ensure unique and limit to 7
    seen = set()
    unique_questions = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)
            
    return unique_questions[:7]
