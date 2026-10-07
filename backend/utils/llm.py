import os
import json
import re
import pandas as pd
import numpy as np

def standardize_blood_group(val: str) -> str:
    if not isinstance(val, str):
        return val
    val_clean = val.strip().lower()
    
    # Normalize unicode minus, hyphen, space, and pos/neg words
    val_clean = val_clean.replace("−", "-").replace(" ", "")
    
    # Map A
    if val_clean in ["apositive", "apos", "a+", "a-pos", "a-positive"]:
        return "A+"
    if val_clean in ["anegative", "aneg", "a-", "a-neg", "a-negative"]:
        return "A-"
        
    # Map B
    if val_clean in ["bpositive", "bpos", "b+", "b-pos", "b-positive"]:
        return "B+"
    if val_clean in ["bnegative", "bneg", "b-", "b-neg", "b-negative"]:
        return "B-"
        
    # Map AB
    if val_clean in ["abpositive", "abpos", "ab+", "ab-pos", "ab-positive"]:
        return "AB+"
    if val_clean in ["abnegative", "abneg", "ab-", "ab-neg", "ab-negative"]:
        return "AB-"
        
    # Map O
    if val_clean in ["opositive", "opos", "o+", "o-pos", "o-positive"]:
        return "O+"
    if val_clean in ["onegative", "oneg", "o-", "o-neg", "o-negative"]:
        return "O-"
        
    return val


def call_llm_api(provider: str, api_key: str, system_prompt: str, user_prompt: str, df_context: pd.DataFrame = None) -> dict:
    """
    Interfaces with OpenAI, Anthropic, or a Local Mock to process cleaning requests.
    Returns a structured dictionary matching the recommendation schema.
    """
    p_lower = provider.lower()
    recommendations = {}
    
    # Use Mock if explicitly selected or if API keys are missing
    if "mock" in p_lower or not api_key:
        recommendations = _generate_mock_recommendations(df_context)
    else:
        try:
            if "openai" in p_lower:
                # pyrefly: ignore [missing-import]
                from openai import OpenAI
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
                recommendations = json.loads(response.choices[0].message.content)
                
            elif "anthropic" in p_lower:
                # pyrefly: ignore [missing-import]
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=4000,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                content = response.content[0].text
                json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
                if json_match:
                    content = json_match.group(1)
                recommendations = json.loads(content)
                
            else:
                recommendations = _generate_mock_recommendations(df_context)
                
        except Exception as e:
            print(f"Error calling LLM provider {provider}: {str(e)}")
            recommendations = _generate_mock_recommendations(df_context)
            recommendations["explanation"] = f"Failed to call {provider} API (Error: {str(e)}). Active fallback to simulated recommendations."

    # Post-process recommendations to enforce constraints based on dynamic semantic types
    if isinstance(recommendations, dict) and "columns" in recommendations and df_context is not None:
        from backend.cleaning.profiler import infer_semantic_type
        new_cols_recs = []
        for col_rec in recommendations["columns"]:
            col_name = col_rec.get("column")
            if col_name in df_context.columns:
                series = df_context[col_name]
                sem_type = infer_semantic_type(col_name, series)
                
                # Expose inferred type
                col_rec["semantic_type"] = sem_type
                
                if sem_type == "Identifier":
                    # Strictly no value mappings, no type casts, no outlier treatment, no imputation (unless dropping missing IDs)
                    col_rec["value_mappings"] = {}
                    col_rec["type_cast"] = None
                    col_rec["outlier_strategy"] = None
                    if col_rec.get("imputation_strategy") not in ["drop", None]:
                        col_rec["imputation_strategy"] = "drop" if series.isna().sum() > 0 else None
                    col_rec["explanation"] = f"Column '{col_name}' is a unique Identifier. Value standardizations and imputations are blocked."
                
                elif sem_type in ["Email", "Phone Number"]:
                    # No standardizations via LLM, no imputation, no outlier treatments
                    col_rec["value_mappings"] = {}
                    col_rec["outlier_strategy"] = None
                    col_rec["imputation_strategy"] = None
                    col_rec["imputation_value"] = None
                    col_rec["type_cast"] = "str" if sem_type == "Email" else None
                    col_rec["explanation"] = f"Column '{col_name}' contains sensitive {sem_type} strings. Allowed transformations are restricted to formatting normalizations and structure validation."
                    
                elif sem_type == "Free Text":
                    # No semantic value mapping, no imputation, no type cast
                    col_rec["value_mappings"] = {}
                    col_rec["outlier_strategy"] = None
                    col_rec["imputation_strategy"] = None
                    col_rec["imputation_value"] = None
                    col_rec["type_cast"] = None
                    col_rec["explanation"] = f"Column '{col_name}' is identified as unstructured Free Text. Standardizations and data imputation are disabled to protect narrative integrity."
                    
                elif sem_type == "Date":
                    # No value mappings, no imputation, only format casting to datetime
                    col_rec["value_mappings"] = {}
                    col_rec["outlier_strategy"] = None
                    col_rec["imputation_strategy"] = None
                    col_rec["imputation_value"] = None
                    col_rec["type_cast"] = "datetime"
                    col_rec["explanation"] = f"Column '{col_name}' is parsed as Date. Converting valid patterns to YYYY-MM-DD and protecting ambiguous layouts."
                    
                elif sem_type == "Numeric":
                    # Clean up: ensure no value mappings. Imputation and outliers allowed
                    col_rec["value_mappings"] = {}
                    if col_rec.get("type_cast") not in ["int", "float", None]:
                        col_rec["type_cast"] = None

                elif sem_type == "Boolean":
                    col_rec["value_mappings"] = {}
                    col_rec["type_cast"] = "bool"
                    
                elif sem_type == "Categorical":
                    # Clean up identity maps from mappings
                    mappings = col_rec.get("value_mappings", {})
                    if isinstance(mappings, dict):
                        col_rec["value_mappings"] = {k: v for k, v in mappings.items() if k != v}
                    
                    if col_name.lower().strip() in ["blood_group", "blood group", "bloodgroup"]:
                        from backend.cleaning.profiler import DISGUISED_PATTERNS
                        unique_vals = [str(x).strip() for x in series.dropna().unique() if str(x).strip().lower() not in DISGUISED_PATTERNS]
                        blood_mappings = {}
                        for v in unique_vals:
                            std_val = standardize_blood_group(v)
                            if std_val != v:
                                blood_mappings[v] = std_val
                        if not isinstance(col_rec.get("value_mappings"), dict):
                            col_rec["value_mappings"] = {}
                        col_rec["value_mappings"].update(blood_mappings)
                
                # Check for empty recommendations
                has_rec = (
                    col_rec.get("type_cast") or 
                    col_rec.get("disguised_nulls") or 
                    col_rec.get("value_mappings") or 
                    col_rec.get("imputation_strategy") or 
                    col_rec.get("outlier_strategy") or
                    sem_type in ["Phone Number", "Free Text", "Email", "Date"]
                )
                if not has_rec:
                    continue
                new_cols_recs.append(col_rec)
        recommendations["columns"] = new_cols_recs
        
    return recommendations

def _generate_mock_recommendations(df: pd.DataFrame) -> dict:
    """
    Generates high-fidelity, dynamic cleaning recommendations by analyzing 
    the column names and values of the DataFrame.
    """
    recommendations = {
        "columns": [],
        "drop_duplicates": False,
        "explanation": "Completed statistical profiling and semantic analysis. Generated optimal heuristics-based corrections."
    }
    
    if df is None:
        return recommendations
        
    # Check duplicate rows
    if df.duplicated().sum() > 0:
        recommendations["drop_duplicates"] = True
        
    from backend.cleaning.profiler import infer_semantic_type, DISGUISED_PATTERNS
    
    for col in df.columns:
        series = df[col]
        series_clean = series.dropna()
        sem_type = infer_semantic_type(col, series)
        
        # Disguised nulls detection
        disguised_nulls = []
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series.dtype):
            for v in series_clean.unique():
                if str(v).strip().lower() in DISGUISED_PATTERNS:
                    disguised_nulls.append(str(v))
                    
        rec = {
            "column": col,
            "semantic_type": sem_type,
            "type_cast": None,
            "disguised_nulls": disguised_nulls,
            "value_mappings": {},
            "imputation_strategy": None,
            "imputation_value": None,
            "outlier_strategy": None,
            "explanation": f"Statistical review of column '{col}' is complete. No issues detected."
        }
        
        # Standard safety: handle disguised nulls first
        if disguised_nulls:
            rec["explanation"] = f"Detected {len(disguised_nulls)} disguised missing values. Mapped to true Nulls."
            
        # 1. Identifier columns
        if sem_type == "Identifier":
            # Only recommend dropping missing values if there are any
            missing_cnt = series.isna().sum() + len([x for x in series_clean if str(x).strip().lower() in DISGUISED_PATTERNS])
            if missing_cnt > 0:
                rec["imputation_strategy"] = "drop"
                rec["explanation"] = f"Column '{col}' is a key identifier with {missing_cnt} missing values. Recommended dropping rows with missing IDs."
            else:
                rec["explanation"] = f"Column '{col}' is a key identifier. Checked for duplicates and pattern violations."
                
        # 2. Email columns
        elif sem_type == "Email":
            rec["type_cast"] = "str"
            rec["explanation"] = f"Column '{col}' detected as Email. Normalizing whitespace/casing and validating structures."
            
        # 3. Phone Number columns
        elif sem_type == "Phone Number":
            rec["explanation"] = f"Column '{col}' detected as Phone Number. Normalizing format spacing and validating digits."
            
        # 4. Date columns
        elif sem_type == "Date":
            rec["type_cast"] = "datetime"
            rec["explanation"] = f"Column '{col}' detected as Date. Converting valid dates to YYYY-MM-DD format."
            
        # 5. Free Text columns
        elif sem_type == "Free Text":
            rec["explanation"] = f"Column '{col}' detected as Free Text. Formatting normalizations applied."
            
        # 6. Numeric columns
        elif sem_type == "Numeric":
            # If all non-null values are integers (e.g. float values that are actually whole numbers like 20.0), cast to int
            non_null_vals = series_clean[~series_clean.apply(lambda x: str(x).strip().lower() in DISGUISED_PATTERNS)]
            # Mixed numeric columns commonly arrive as object dtype because a
            # token such as "missing" is embedded among valid numbers.  All
            # numeric statistics must use the coerced values, never the original
            # strings (which made Series.skew/mean raise on "missing").
            numeric_vals = pd.to_numeric(non_null_vals, errors="coerce").dropna()
            try:
                if not numeric_vals.empty and (numeric_vals % 1 == 0).all():
                    rec["type_cast"] = "int"
                elif not pd.api.types.is_numeric_dtype(series) and not numeric_vals.empty:
                    rec["type_cast"] = "float"
            except:
                pass
                
            # Missing Value Imputation
            total_missing = series.isna().sum() + len([x for x in series_clean if str(x).strip().lower() in DISGUISED_PATTERNS])
            if total_missing > 0 and not numeric_vals.empty:
                # Calculate skewness to decide between mean and median
                skewness = numeric_vals.skew()
                if not pd.isna(skewness) and abs(skewness) > 1.0:
                    rec["imputation_strategy"] = "median"
                    rec["imputation_value"] = float(numeric_vals.median())
                    rec["explanation"] = f"Numeric column with missing values. Recommended median imputation ({rec['imputation_value']:.2f}) due to skewness ({skewness:.2f})."
                else:
                    rec["imputation_strategy"] = "mean"
                    rec["imputation_value"] = float(numeric_vals.mean())
                    rec["explanation"] = f"Numeric column with missing values. Recommended mean imputation ({rec['imputation_value']:.2f}) due to symmetric distribution."
            
            # Outlier detection
            if not numeric_vals.empty:
                q1 = numeric_vals.quantile(0.25)
                q3 = numeric_vals.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = numeric_vals[(numeric_vals < lower) | (numeric_vals > upper)]
                if len(outliers) > 0:
                    rec["outlier_strategy"] = "clip"
                    rec["explanation"] = f"Detected {len(outliers)} numeric outliers beyond IQR bounds. Recommended clipping bounds."
                    
        # 7. Boolean columns
        elif sem_type == "Boolean":
            rec["type_cast"] = "bool"
            total_missing = series.isna().sum() + len([x for x in series_clean if str(x).strip().lower() in DISGUISED_PATTERNS])
            if total_missing > 0 and not series_clean.empty:
                mode_val = series_clean.mode().iloc[0]
                if str(mode_val).lower() in ["true", "yes", "y", "1", "t"]:
                    mode_bool = True
                else:
                    mode_bool = False
                rec["imputation_strategy"] = "mode"
                rec["imputation_value"] = mode_bool
                rec["explanation"] = f"Boolean column with missing values. Recommended mode imputation ({mode_bool})."
            else:
                rec["explanation"] = f"Boolean column detected. Casting to true boolean representation."
                
        # 8. Categorical columns
        elif sem_type == "Categorical":
            unique_vals = [str(x).strip() for x in series_clean.unique() if str(x).strip().lower() not in DISGUISED_PATTERNS]
            
            mappings = {}
            if col.lower().strip() in ["blood_group", "blood group", "bloodgroup"]:
                for v in unique_vals:
                    std_val = standardize_blood_group(v)
                    if std_val != v:
                        mappings[v] = std_val
            else:
                # 1. Resolve Acronyms first (e.g. CS -> Computer Science) only when full representation exists in the column
                phrases = [v for v in unique_vals if len(v.split()) > 1]
                acronym_candidates = [v for v in unique_vals if len(v.split()) == 1]
                
                for ac in acronym_candidates:
                    matched_phrase = None
                    for ph in phrases:
                        ac_clean = ac.upper().replace(".", "")
                        ph_words = [w.strip() for w in ph.split() if w.strip()]
                        first_letters = "".join(w[0].upper() for w in ph_words) if len(ph_words) >= 2 else ""
                        if first_letters and ac_clean == first_letters:
                            matched_phrase = ph
                            break
                    if matched_phrase:
                        mappings[ac] = matched_phrase
    
                # 2. Map inconsistent casings to the most frequent casing / formatted casing
                casing_groups = {}
                for v in unique_vals:
                    norm = v.lower()
                    if norm not in casing_groups:
                        casing_groups[norm] = []
                    casing_groups[norm].append(v)
                    
                for norm, variations in casing_groups.items():
                    if len(variations) > 1:
                        freq = series_clean[series_clean.astype(str).str.strip().isin(variations)].value_counts()
                        best_val = freq.index[0] if not freq.empty else variations[0].title()
                        for v in variations:
                            if v != best_val and v not in mappings:
                                mappings[v] = best_val
                
                # 3. Simple category standardization for common abbreviations (domain-independent)
                for v in unique_vals:
                    if v in mappings:
                        continue
                    norm = v.lower()
                    if norm in ["m", "male"]:
                        mappings[v] = "Male"
                    elif norm in ["f", "female"]:
                        mappings[v] = "Female"
                    elif norm in ["usa", "us", "u.s.a."]:
                        mappings[v] = "USA"
                    elif norm in ["uk", "united kingdom", "u.k."]:
                        mappings[v] = "UK"
            
            # Filter standard mappings to make sure we don't have identity maps
            mappings = {k: v for k, v in mappings.items() if k != v}
            
            if mappings:
                rec["value_mappings"] = mappings
                rec["explanation"] = f"Categorical column casing/abbreviation inconsistencies. Standardizing categories."
                
            # Missing Value Imputation
            total_missing = series.isna().sum() + len([x for x in series_clean if str(x).strip().lower() in DISGUISED_PATTERNS])
            if total_missing > 0 and not series_clean.empty:
                mode_val = str(series_clean.mode().iloc[0])
                rec["imputation_strategy"] = "mode"
                rec["imputation_value"] = mode_val
                if "inconsistencies" in rec["explanation"]:
                    rec["explanation"] += f" Imputing missing values with mode ('{mode_val}')."
                else:
                    rec["explanation"] = f"Categorical column with missing values. Recommended mode imputation ('{mode_val}')."
                    
        # Add to suggestions if any operation is proposed
        if rec["type_cast"] or rec["disguised_nulls"] or rec["value_mappings"] or rec["imputation_strategy"] or rec["outlier_strategy"] or sem_type in ["Phone Number", "Free Text", "Email", "Date"]:
            recommendations["columns"].append(rec)
            
    return recommendations
