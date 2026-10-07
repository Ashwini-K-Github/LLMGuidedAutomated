import pandas as pd
import numpy as np
import re

def clean_date_column(series: pd.Series) -> pd.Series:
    """
    Standardizes date values in a column dynamically.
    Checks column-wide formats for dayfirst / monthfirst signals to resolve individual ambiguity.
    """
    from backend.cleaning.profiler import DISGUISED_PATTERNS
    
    # 1. Inspect non-null values for formatting indicators
    has_day_first = False
    has_month_first = False
    
    for val in series.dropna():
        val_str = str(val).strip()
        # Look for patterns like DD-MM-YYYY or MM/DD/YYYY
        match = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', val_str)
        if match:
            g1, g2 = int(match.group(1)), int(match.group(2))
            if g1 > 12 and g2 <= 12:
                has_day_first = True
            elif g2 > 12 and g1 <= 12:
                has_month_first = True
                
    # 2. Extract unambiguous months and years to resolve ambiguity
    unambiguous_months = []
    unambiguous_years = []
    
    for val in series.dropna():
        val_str = str(val).strip()
        if val_str.lower() in DISGUISED_PATTERNS:
            continue
            
        if re.match(r'^\d{4}-\d{2}-\d{2}$', val_str):
            try:
                dt = pd.to_datetime(val_str, format='%Y-%m-%d')
                unambiguous_months.append(dt.month)
                unambiguous_years.append(dt.year)
            except:
                pass
        else:
            match = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', val_str)
            if match:
                g1, g2, g3 = int(match.group(1)), int(match.group(2)), int(match.group(3))
                if g1 > 12 and g2 <= 12:
                    unambiguous_months.append(g2)
                    unambiguous_years.append(g3)
                elif g2 > 12 and g1 <= 12:
                    unambiguous_months.append(g1)
                    unambiguous_years.append(g3)

    common_month = None
    common_year = None
    if unambiguous_months:
        common_month = pd.Series(unambiguous_months).mode().iloc[0]
    if unambiguous_years:
        common_year = pd.Series(unambiguous_years).mode().iloc[0]

    # 3. Parse values using the layout indicator and common components
    def parse_single_val(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if val_str.lower() in DISGUISED_PATTERNS:
            return np.nan
            
        # If already YYYY-MM-DD, verify it's a real parseable date
        if re.match(r'^\d{4}-\d{2}-\d{2}$', val_str):
            try:
                pd.to_datetime(val_str, format='%Y-%m-%d', errors='raise')
                return val_str
            except:
                return np.nan
            
        try:
            # Check individual value ambiguity (e.g. "01/02/2025")
            match = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', val_str)
            if match:
                g1, g2 = int(match.group(1)), int(match.group(2))
                if g1 <= 12 and g2 <= 12 and g1 != g2:
                    if has_day_first and not has_month_first:
                        parsed = pd.to_datetime(val_str, dayfirst=True, errors='raise')
                    elif has_month_first and not has_day_first:
                        parsed = pd.to_datetime(val_str, dayfirst=False, errors='raise')
                    else:
                        parsed_dayfirst = None
                        parsed_monthfirst = None
                        try:
                            parsed_dayfirst = pd.to_datetime(val_str, dayfirst=True, errors='raise')
                        except:
                            pass
                        try:
                            parsed_monthfirst = pd.to_datetime(val_str, dayfirst=False, errors='raise')
                        except:
                            pass
                            
                        if parsed_dayfirst is not None and parsed_monthfirst is not None:
                            match_dayfirst = (
                                (common_month is None or parsed_dayfirst.month == common_month) and
                                (common_year is None or parsed_dayfirst.year == common_year)
                            )
                            match_monthfirst = (
                                (common_month is None or parsed_monthfirst.month == common_month) and
                                (common_year is None or parsed_monthfirst.year == common_year)
                            )
                            if match_dayfirst and not match_monthfirst:
                                parsed = parsed_dayfirst
                            elif match_monthfirst and not match_dayfirst:
                                parsed = parsed_monthfirst
                            else:
                                parsed = parsed_monthfirst
                        elif parsed_dayfirst is not None:
                            parsed = parsed_dayfirst
                        elif parsed_monthfirst is not None:
                            parsed = parsed_monthfirst
                        else:
                            return np.nan
                else:
                    if g1 > 12:
                        parsed = pd.to_datetime(val_str, dayfirst=True, errors='raise')
                    else:
                        parsed = pd.to_datetime(val_str, dayfirst=False, errors='raise')
            else:
                # Other format (like YYYY/MM/DD)
                parsed = pd.to_datetime(val_str, errors='raise')
                
            return parsed.strftime('%Y-%m-%d')
        except:
            # Failed to parse: nullify it to resolve the issue
            return np.nan
            
    return series.apply(parse_single_val)

def apply_cleaning_recommendations(df: pd.DataFrame, recommendations: dict, active_cols: list = None) -> tuple:
    """
    Module 6: Automatic Data Cleaning
    Applies the selected recommendations to the DataFrame using Pandas.
    Returns: (cleaned_df, history_log)
    """
    if df is None:
        return None, []
        
    cleaned_df = df.copy()
    history = []
    
    # Filter columns to clean based on client selection
    cols_to_clean = recommendations.get("columns", [])
    if active_cols is not None:
        cols_to_clean = [c for c in cols_to_clean if c["column"] in active_cols]
        
    from backend.cleaning.profiler import infer_semantic_type, is_valid_phone, is_valid_email, DISGUISED_PATTERNS
    
    # 0. Apply Formatting Normalization by Semantic Type
    for col_rec in cols_to_clean:
        col = col_rec["column"]
        if col in cleaned_df.columns:
            # Infer semantic type dynamically or read from recommendation
            sem_type = col_rec.get("semantic_type") or infer_semantic_type(col, cleaned_df[col])
            
            # A. Email Casing and Spacing Normalization
            if sem_type == "Email" and (pd.api.types.is_object_dtype(cleaned_df[col]) or pd.api.types.is_string_dtype(cleaned_df[col].dtype)):
                def clean_email_val(val):
                    if pd.isna(val):
                        return np.nan
                    v_str = str(val).strip()
                    if v_str.lower() in DISGUISED_PATTERNS:
                        return np.nan
                    # Check if valid structure before lowercasing, otherwise return np.nan (remove cell)
                    if is_valid_email(v_str):
                        return v_str.lower()
                    return np.nan
                    
                cleaned_df[col] = cleaned_df[col].apply(clean_email_val)
                history.append({
                    "column": col,
                    "action": "Email Normalization",
                    "details": "Normalized valid email addresses to lowercase and stripped spaces; preserved invalid layouts.",
                    "code": f"cleaned_df['{col}'] = cleaned_df['{col}'].apply(lambda x: x.strip().lower() if is_valid_email(x) else x)"
                })
                
            # B. Phone Number Spacing and Formatting Normalization
            elif sem_type == "Phone Number":
                def clean_phone_val(val):
                    if pd.isna(val):
                        return np.nan
                    v_str = str(val).strip()
                    if v_str.lower() in DISGUISED_PATTERNS:
                        return np.nan
                        
                    # Remove common formatting characters (spaces, hyphens, parenthesis, dots)
                    cleaned = re.sub(r'[\s\-()\[\]\.]', '', v_str)
                    if cleaned.endswith(".0"):
                        cleaned = cleaned[:-2]
                    if "e+" in cleaned.lower():
                        try:
                            cleaned = str(int(float(v_str)))
                        except:
                            pass
                            
                    # If the number has 11 digits starting with 1 or +1, strip the country prefix
                    digits_only = re.sub(r'\D', '', cleaned)
                    if len(digits_only) == 11 and (cleaned.startswith("1") or cleaned.startswith("+1")):
                        if cleaned.startswith("+1"):
                            candidate = cleaned[2:]
                        else:
                            candidate = cleaned[1:]
                        if is_valid_phone(candidate):
                            cleaned = candidate
                            
                    # If it is valid according to is_valid_phone, return cleaned, otherwise return np.nan
                    if is_valid_phone(cleaned):
                        return cleaned
                    return np.nan
                    
                cleaned_df[col] = cleaned_df[col].apply(clean_phone_val)
                history.append({
                    "column": col,
                    "action": "Phone Normalization",
                    "details": "Trimmed formatting symbols from valid numbers, preserving leading zero or + symbols.",
                    "code": f"cleaned_df['{col}'] = cleaned_df['{col}'].apply(lambda x: clean_phone(x))"
                })
                
            # C. Date Layout Standardization
            elif sem_type == "Date":
                cleaned_df[col] = clean_date_column(cleaned_df[col])
                history.append({
                    "column": col,
                    "action": "Date Standardizing",
                    "details": "Parsed and standardized valid layouts to YYYY-MM-DD, protecting ambiguous dates.",
                    "code": f"cleaned_df['{col}'] = clean_date_column(cleaned_df['{col}'])"
                })
                # E. Age Domain & Outlier Validation (Cell-level Nullification)
            elif col.lower().strip() == "age":
                series = cleaned_df[col].dropna()
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outlier_strategy = col_rec.get("outlier_strategy")
                    
                    def clean_age_val(val):
                        if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                            return np.nan
                        try:
                            v = float(val)
                            # Age Domain constraint: must be between 0 and 100
                            if v < 0 or v > 100:
                                return np.nan
                            # IQR Outliers check
                            if v < lower_bound or v > upper_bound:
                                if outlier_strategy == "clip":
                                    return lower_bound if v < lower_bound else upper_bound
                                elif outlier_strategy == "keep":
                                    return val
                                else:
                                    return np.nan
                            return val
                        except:
                            return np.nan
                            
                    cleaned_df[col] = cleaned_df[col].apply(clean_age_val)
                    if outlier_strategy == "clip":
                        details = f"Clipped Age outliers to IQR bounds [{lower_bound:.1f}, {upper_bound:.1f}], nullified invalid domain values."
                        code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].apply(lambda x: np.nan if pd.isna(x) or float(x) < 0 or float(x) > 100 else ({lower_bound:.2f} if float(x) < {lower_bound:.2f} else ({upper_bound:.2f} if float(x) > {upper_bound:.2f} else x)))"
                    else:
                        details = f"Nullified Age values outside [0, 100] or beyond IQR bounds [{lower_bound:.1f}, {upper_bound:.1f}]."
                        code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].apply(lambda x: np.nan if pd.isna(x) or float(x) < 0 or float(x) > 100 or float(x) < {lower_bound:.2f} or float(x) > {upper_bound:.2f} else x)"
                        
                    history.append({
                        "column": col,
                        "action": "Age Domain & IQR Cleaning",
                        "details": details,
                        "code": code_str
                    })
            # D. Free Text Whitespace Stripping
            elif sem_type == "Free Text" and (pd.api.types.is_object_dtype(cleaned_df[col]) or pd.api.types.is_string_dtype(cleaned_df[col].dtype)):
                cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)
                history.append({
                    "column": col,
                    "action": "Text Normalization",
                    "details": "Normalized trailing and leading whitespaces for unstructured text values.",
                    "code": f"cleaned_df['{col}'] = cleaned_df['{col}'].astype(str).str.strip()"
                })

    # 1. Handle Disguised Nulls
    for col_rec in cols_to_clean:
        col = col_rec["column"]
        null_vals = col_rec.get("disguised_nulls", [])
        if null_vals and col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].replace(null_vals, np.nan)
            history.append({
                "column": col,
                "action": "Standardized Disguised Nulls",
                "details": f"Mapped disguised null string variations {null_vals} to true Nulls (NaN)",
                "code": f"cleaned_df['{col}'] = cleaned_df['{col}'].replace({null_vals}, np.nan)"
            })

    # 2. Value Mappings (Standardization maps for Categorical columns)
    for col_rec in cols_to_clean:
        col = col_rec["column"]
        mappings = col_rec.get("value_mappings", {})
        if mappings and col in cleaned_df.columns:
            sem_type = col_rec.get("semantic_type") or infer_semantic_type(col, cleaned_df[col])
            # Enforce safety: do NOT map Identifier, Email, Phone, Free Text
            if sem_type in ["Identifier", "Email", "Phone Number", "Free Text"]:
                continue
                
            actual_mappings = {k: v for k, v in mappings.items() if cleaned_df[col].isin([k]).any()}
            if actual_mappings:
                cleaned_df[col] = cleaned_df[col].replace(actual_mappings)
                history.append({
                    "column": col,
                    "action": "Semantic Value Mapping",
                    "details": f"Standardized inconsistent values: {actual_mappings}",
                    "code": f"cleaned_df['{col}'] = cleaned_df['{col}'].replace({dict(actual_mappings)})"
                })

    # 3. Type Casting
    for col_rec in cols_to_clean:
        col = col_rec["column"]
        target_type = col_rec.get("type_cast")
        if target_type and col in cleaned_df.columns:
            sem_type = col_rec.get("semantic_type") or infer_semantic_type(col, cleaned_df[col])
            # Safety: do NOT cast Identifier
            if sem_type == "Identifier":
                continue
                
            try:
                if target_type == "int":
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').round().astype('Int64')
                    code_str = f"cleaned_df['{col}'] = pd.to_numeric(cleaned_df['{col}']).round().astype('Int64')"
                elif target_type == "float":
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    code_str = f"cleaned_df['{col}'] = pd.to_numeric(cleaned_df['{col}'], errors='coerce')"
                elif target_type == "datetime":
                    cleaned_df[col] = clean_date_column(cleaned_df[col])
                    code_str = f"cleaned_df['{col}'] = clean_date_column(cleaned_df['{col}'])"
                elif target_type == "bool":
                    cleaned_df[col] = cleaned_df[col].map({
                        'Yes': True, 'Y': True, '1': True, 'true': True, 'True': True, 1: True, True: True,
                        'No': False, 'N': False, '0': False, 'false': False, 'False': False, 0: False, False: False
                    })
                    code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].map({{ 'Yes': True, 'No': False, '1': True, '0': False }})"
                elif target_type == "str":
                    cleaned_df[col] = cleaned_df[col].apply(lambda x: str(x) if pd.notna(x) and x is not None else np.nan)
                    code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].apply(lambda x: str(x) if pd.notna(x) else np.nan)"
                else:
                    continue
                    
                history.append({
                    "column": col,
                    "action": "Casted Data Type",
                    "details": f"Converted column values to '{target_type}' format",
                    "code": code_str
                })
            except Exception as e:
                history.append({
                    "column": col,
                    "action": "Casted Data Type (FAILED)",
                    "details": f"Failed type casting to {target_type}: {str(e)}",
                    "code": f"# Failed: {str(e)}"
                })

    # 4. Outliers Handling (Numeric columns only)
    for col_rec in cols_to_clean:
        col = col_rec["column"]
        outlier_strategy = col_rec.get("outlier_strategy")
        if outlier_strategy and col in cleaned_df.columns:
            if col.lower().strip() == "age":
                continue
            sem_type = col_rec.get("semantic_type") or infer_semantic_type(col, cleaned_df[col])
            # Safety: Outliers only allowed for Numeric columns
            if sem_type != "Numeric" or not pd.api.types.is_numeric_dtype(cleaned_df[col]):
                continue
                
            series = cleaned_df[col].dropna()
            if len(series) > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                if outlier_strategy == "clip":
                    if pd.api.types.is_integer_dtype(cleaned_df[col]):
                        lower_bound = int(np.round(lower_bound))
                        upper_bound = int(np.round(upper_bound))
                    
                    def clip_outlier(val):
                        if pd.isna(val):
                            return np.nan
                        try:
                            v = float(val)
                            if v < lower_bound:
                                return lower_bound
                            elif v > upper_bound:
                                return upper_bound
                            return val
                        except:
                            return val
                    cleaned_df[col] = cleaned_df[col].apply(clip_outlier)
                    details = f"Clipped outliers to IQR bounds [{lower_bound:.1f}, {upper_bound:.1f}]"
                    code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].apply(lambda x: {lower_bound:.2f} if pd.notna(x) and float(x) < {lower_bound:.2f} else ({upper_bound:.2f} if pd.notna(x) and float(x) > {upper_bound:.2f} else x))"
                elif outlier_strategy == "remove":
                    cleaned_df = cleaned_df[(cleaned_df[col] >= lower_bound) & (cleaned_df[col] <= upper_bound) | cleaned_df[col].isna()]
                    details = f"Dropped rows containing outliers beyond IQR bounds [{lower_bound:.1f}, {upper_bound:.1f}]"
                    code_str = f"cleaned_df = cleaned_df[(cleaned_df['{col}'] >= {lower_bound:.2f}) & (cleaned_df['{col}'] <= {upper_bound:.2f}) | cleaned_df['{col}'].isna()]"
                else:
                    continue
                    
                history.append({
                    "column": col,
                    "action": "Handled Outliers",
                    "details": details,
                    "code": code_str
                })

    # 5. Missing Value Imputation
    for col_rec in cols_to_clean:
        col = col_rec["column"]
        strategy = col_rec.get("imputation_strategy")
        imp_val = col_rec.get("imputation_value")
        
        if strategy and col in cleaned_df.columns:
            sem_type = col_rec.get("semantic_type") or infer_semantic_type(col, cleaned_df[col])
            # Safety: Do NOT impute Identifier (except drop), Email, Phone, Free Text
            if sem_type in ["Identifier", "Email", "Phone Number", "Free Text"]:
                if sem_type == "Identifier" and strategy == "drop":
                    cleaned_df = cleaned_df.dropna(subset=[col])
                    history.append({
                        "column": col,
                        "action": "Imputed Missing Values",
                        "details": f"Dropped rows with missing values in key Identifier '{col}'",
                        "code": f"cleaned_df = cleaned_df.dropna(subset=['{col}'])"
                    })
                continue
                
            if strategy == "median":
                val = float(cleaned_df[col].median()) if imp_val is None else imp_val
                if pd.api.types.is_integer_dtype(cleaned_df[col]):
                    val = int(np.round(val))
                cleaned_df[col] = cleaned_df[col].fillna(val)
                details = f"Imputed missing cells with column median ({val})"
                code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].fillna({val})"
            elif strategy == "mean":
                val = float(cleaned_df[col].mean()) if imp_val is None else imp_val
                if pd.api.types.is_integer_dtype(cleaned_df[col]):
                    val = int(np.round(val))
                cleaned_df[col] = cleaned_df[col].fillna(val)
                details = f"Imputed missing cells with column mean ({val:.2f})"
                code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].fillna({val:.4f})"
            elif strategy == "mode":
                mode_series = cleaned_df[col].mode()
                val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                val = imp_val if imp_val is not None else val
                if mappings and val in mappings:
                    val = mappings[val]
                val_repr = f"'{val}'" if isinstance(val, str) else str(val)
                cleaned_df[col] = cleaned_df[col].fillna(val)
                details = f"Imputed missing cells with column mode ({val_repr})"
                code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].fillna({val_repr})"
            elif strategy == "constant":
                val = imp_val if imp_val is not None else "Missing"
                if mappings and val in mappings:
                    val = mappings[val]
                val_repr = f"'{val}'" if isinstance(val, str) else str(val)
                cleaned_df[col] = cleaned_df[col].fillna(val)
                details = f"Imputed missing cells with constant: {val_repr}"
                code_str = f"cleaned_df['{col}'] = cleaned_df['{col}'].fillna({val_repr})"
            elif strategy == "drop":
                cleaned_df = cleaned_df.dropna(subset=[col])
                details = f"Dropped rows with missing values in column '{col}'"
                code_str = f"cleaned_df = cleaned_df.dropna(subset=['{col}'])"
            else:
                continue
                
            history.append({
                "column": col,
                "action": "Imputed Missing Values",
                "details": details,
                "code": code_str
            })

    # 5.5 Final Pass: Drop rows with remaining NaNs in cleaned columns to ensure 0 unresolved issues
    dropped_cols_details = []
    for col_rec in cols_to_clean:
        col = col_rec["column"]
        strategy = col_rec.get("imputation_strategy")
        if col in cleaned_df.columns:
            # If no active imputation strategy (or strategy is 'drop')
            if strategy is None or strategy == "drop":
                # If the column has remaining NaNs (missing values or nullified invalid/malformed values)
                na_count = cleaned_df[col].isna().sum()
                # If the column has disguised nulls remaining
                if pd.api.types.is_object_dtype(cleaned_df[col]) or pd.api.types.is_string_dtype(cleaned_df[col].dtype):
                    na_count += sum(1 for x in cleaned_df[col].dropna() if str(x).strip().lower() in DISGUISED_PATTERNS)
                
                if na_count > 0:
                    cleaned_df = cleaned_df.dropna(subset=[col])
                    # Filter out any remaining disguised patterns if object type
                    if pd.api.types.is_object_dtype(cleaned_df[col]) or pd.api.types.is_string_dtype(cleaned_df[col].dtype):
                        cleaned_df = cleaned_df[~cleaned_df[col].astype(str).str.strip().str.lower().isin(DISGUISED_PATTERNS)]
                    dropped_cols_details.append(f"Dropped {na_count} rows with unresolved/NaN values in '{col}'")
                
    if dropped_cols_details:
        history.append({
            "column": "ALL",
            "action": "Resolved Remaining NaNs/Malformed Cells",
            "details": "; ".join(dropped_cols_details),
            "code": "# Final pass: drop rows with remaining NaNs\nfor c in active_cols:\n    cleaned_df = cleaned_df.dropna(subset=[c])"
        })

    # 6. Deduplication
    if recommendations.get("drop_duplicates", False):
        dup_count = cleaned_df.duplicated().sum()
        if dup_count > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            history.append({
                "column": "ALL",
                "action": "Deduplicated Dataset",
                "details": f"Removed {dup_count} duplicate rows",
                "code": "cleaned_df = cleaned_df.drop_duplicates()"
            })
            
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df, history
