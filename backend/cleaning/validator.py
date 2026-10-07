import pandas as pd
import numpy as np
import re
from backend.cleaning.profiler import infer_semantic_type, is_valid_phone, is_valid_email, DISGUISED_PATTERNS

def analyze_data_quality_metrics(df: pd.DataFrame, is_cleaned: bool = False) -> dict:
    """
    Analyzes the dataset and counts occurrences of different data quality issues.
    """
    metrics = {
        "missing_values": 0,
        "duplicates": 0,
        "invalid_dates": 0,
        "email_issues": 0,
        "phone_issues": 0,
        "casing_issues": 0,
        "outliers": 0,
        "invalid_numeric": 0,
        "semantic_inconsistencies": 0
    }
    
    if df is None or len(df) == 0:
        return metrics
        
    num_rows = len(df)
    metrics["duplicates"] = int(df.duplicated().sum())
    
    for col in df.columns:
        series = df[col]
        sem_type = infer_semantic_type(col, series)
        
        # 1. Missing Values (NaNs + Disguised Nulls)
        nan_count = int(series.isna().sum())
        disguised_count = 0
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series.dtype):
            disguised_count = int(sum(1 for x in series.dropna() if str(x).strip().lower() in DISGUISED_PATTERNS))
        metrics["missing_values"] += (nan_count + disguised_count)
        
        series_clean = series.dropna()
        series_val = series_clean[series_clean.apply(lambda x: str(x).strip().lower() not in DISGUISED_PATTERNS)]
        
        # 2. Invalid Dates
        if sem_type == "Date":
            invalid_d = 0
            for val in series_val:
                val_str = str(val).strip()
                try:
                    # After cleaning, dates must strictly conform to YYYY-MM-DD
                    if is_cleaned:
                        if not re.match(r'^\d{4}-\d{2}-\d{2}$', val_str):
                            invalid_d += 1
                        else:
                            pd.to_datetime(val_str, format='%Y-%m-%d', errors='raise')
                    else:
                        pd.to_datetime(val_str, errors='raise')
                except:
                    invalid_d += 1
            metrics["invalid_dates"] += invalid_d
            
        # 3. Email Issues
        elif sem_type == "Email":
            invalid_em = 0
            for val in series_val:
                if not is_valid_email(val):
                    invalid_em += 1
            metrics["email_issues"] += invalid_em
            
        # 4. Phone Issues
        elif sem_type == "Phone Number":
            invalid_ph = 0
            for val in series_val:
                if not is_valid_phone(val):
                    invalid_ph += 1
            metrics["phone_issues"] += invalid_ph
            
        # 5. Casing Issues / Semantic Inconsistencies (for Categorical columns)
        elif sem_type == "Categorical":
            unique_vals = series_val.astype(str).unique()
            casing_map = {}
            for v in unique_vals:
                norm = v.strip().lower()
                if norm not in casing_map:
                    casing_map[norm] = []
                casing_map[norm].append(v)
            casing_count = sum(len(variations) - 1 for variations in casing_map.values() if len(variations) > 1)
            metrics["casing_issues"] += casing_count
            metrics["semantic_inconsistencies"] += casing_count
            
        # 6. Numeric Outliers / Invalid Numeric
        elif sem_type == "Numeric":
            invalid_num = 0
            parsed_vals = []
            is_age_col = col.lower().strip() == "age"
            for val in series_val:
                try:
                    num_val = float(val)
                    parsed_vals.append(num_val)
                    if is_age_col and (num_val < 0 or num_val > 100):
                        invalid_num += 1
                except ValueError:
                    invalid_num += 1
            metrics["invalid_numeric"] += invalid_num
            
            # Count outliers on valid parsed floats
            if len(parsed_vals) > 0:
                s_vals = pd.Series(parsed_vals)
                q1 = s_vals.quantile(0.25)
                q3 = s_vals.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers_count = ((s_vals < lower) | (s_vals > upper)).sum()
                metrics["outliers"] += int(outliers_count)
                
    return metrics

def validate_cleaning(df_original: pd.DataFrame, df_cleaned: pd.DataFrame) -> dict:
    """
    Module 7: Validate Clean Dataset
    Compares the original dataset stats with the cleaned dataset stats.
    """
    if df_original is None or df_cleaned is None:
        return {}
        
    from backend.cleaning.profiler import profile_dataset, infer_semantic_type
    
    orig_profile = profile_dataset(df_original)
    clean_profile = profile_dataset(df_cleaned)
    
    # Calculate detailed quality counts
    orig_quality = analyze_data_quality_metrics(df_original, is_cleaned=False)
    clean_quality = analyze_data_quality_metrics(df_cleaned, is_cleaned=True)
    
    # Standard counts for backward compatibility
    orig_missing = orig_quality["missing_values"]
    clean_missing = clean_quality["missing_values"]
    orig_outliers = orig_quality["outliers"]
    clean_outliers = clean_quality["outliers"]
    
    comparison = {
        "rows": {
            "before": orig_profile["num_rows"],
            "after": clean_profile["num_rows"],
            "diff": clean_profile["num_rows"] - orig_profile["num_rows"]
        },
        "columns": {
            "before": orig_profile["num_cols"],
            "after": clean_profile["num_cols"],
            "diff": clean_profile["num_cols"] - orig_profile["num_cols"]
        },
        "missing_values": {
            "before": orig_missing,
            "after": clean_missing,
            "resolved": orig_missing - clean_missing
        },
        "duplicates": {
            "before": orig_profile["duplicate_rows"],
            "after": clean_profile["duplicate_rows"],
            "resolved": orig_profile["duplicate_rows"] - clean_profile["duplicate_rows"]
        },
        "outliers": {
            "before": orig_outliers,
            "after": clean_outliers,
            "resolved": orig_outliers - clean_outliers
        }
    }
    
    # Detailed Before vs After audit metrics (the 9 metrics)
    comparison["detailed"] = {}
    for key in orig_quality.keys():
        b = orig_quality[key]
        a = clean_quality[key]
        comparison["detailed"][key] = {
            "before": b,
            "after": a,
            "resolved": max(0, b - a),
            "remaining": a
        }
        
    # Compute dynamic quality scores
    orig_total_cells = orig_profile["num_rows"] * orig_profile["num_cols"] if orig_profile["num_rows"] > 0 else 1
    orig_issues = sum(comparison["detailed"][k]["before"] for k in comparison["detailed"] if k != "duplicates") + (comparison["detailed"].get("duplicates", {}).get("before", 0) * orig_profile["num_cols"])
    before_score = max(10, min(100, round(100 * (1 - (orig_issues / orig_total_cells)))))
    
    clean_total_cells = clean_profile["num_rows"] * clean_profile["num_cols"] if clean_profile["num_rows"] > 0 else 1
    clean_issues = sum(comparison["detailed"][k]["after"] for k in comparison["detailed"] if k != "duplicates") + (comparison["detailed"].get("duplicates", {}).get("after", 0) * clean_profile["num_cols"])
    after_score = max(10, min(100, round(100 * (1 - (clean_issues / clean_total_cells)))))
    
    comparison["before_score"] = before_score
    comparison["after_score"] = after_score
    
    phone_cols = [c for c in df_original.columns if infer_semantic_type(c, df_original[c]) == "Phone Number"]
    
    before_valid = 0
    before_invalid = 0
    after_valid = 0
    after_invalid = 0
    
    for col in phone_cols:
        # Before
        before_valid += int(df_original[col].apply(is_valid_phone).sum())
        before_invalid += int(len(df_original) - df_original[col].apply(is_valid_phone).sum())
        
        # After
        if col in df_cleaned.columns:
            after_valid += int(df_cleaned[col].apply(is_valid_phone).sum())
            after_invalid += int(len(df_cleaned) - df_cleaned[col].apply(is_valid_phone).sum())
            
    comparison["phone_validation"] = {
        "before": {
            "valid": before_valid,
            "invalid": before_invalid
        },
        "after": {
            "valid": after_valid,
            "invalid": after_invalid
        },
        "has_phone": len(phone_cols) > 0
    }
    
    # Detailed audit of column validation and standardization
    column_audit = {}
    column_validation_report = {}
    for col in df_cleaned.columns:
        col_series = df_cleaned[col]
        # Infer semantic type based on original dataframe column (to keep semantic context stable)
        sem_type = infer_semantic_type(col, df_original[col]) if col in df_original.columns else infer_semantic_type(col, col_series)
        
        # 1. Missing Values (NaNs + Disguised Nulls)
        missing_count = int(col_series.isna().sum())
        if pd.api.types.is_object_dtype(col_series) or pd.api.types.is_string_dtype(col_series.dtype):
            missing_count += int(sum(1 for x in col_series.dropna() if str(x).strip().lower() in DISGUISED_PATTERNS))
            
        non_null_clean = col_series.dropna()
        non_null_clean_vals = non_null_clean[non_null_clean.apply(lambda x: str(x).strip().lower() not in DISGUISED_PATTERNS)]
        
        # 2. Valid vs Invalid counts
        valid_count = len(non_null_clean_vals)
        invalid_count = 0
        
        if sem_type == "Email":
            valid_count = int(sum(1 for x in non_null_clean_vals if is_valid_email(x)))
            invalid_count = len(non_null_clean_vals) - valid_count
        elif sem_type == "Phone Number":
            valid_count = int(sum(1 for x in non_null_clean_vals if is_valid_phone(x)))
            invalid_count = len(non_null_clean_vals) - valid_count
        elif sem_type == "Date":
            valid_dates = 0
            for val in non_null_clean_vals:
                val_str = str(val).strip()
                if re.match(r'^\d{4}-\d{2}-\d{2}$', val_str):
                    try:
                        pd.to_datetime(val_str, format='%Y-%m-%d', errors='raise')
                        valid_dates += 1
                    except:
                        pass
            valid_count = valid_dates
            invalid_count = len(non_null_clean_vals) - valid_count
        elif sem_type == "Numeric":
            valid_parsed = []
            invalid_parsed_count = 0
            is_age_col = col.lower().strip() == "age"
            for val in non_null_clean_vals:
                try:
                    num_val = float(val)
                    valid_parsed.append(num_val)
                    if is_age_col and (num_val < 0 or num_val > 100):
                        invalid_parsed_count += 1
                except:
                    invalid_parsed_count += 1
            outliers_count = 0
            if valid_parsed:
                s_vals = pd.Series(valid_parsed)
                q1 = s_vals.quantile(0.25)
                q3 = s_vals.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers_count = int(((s_vals < lower) | (s_vals > upper)).sum())
            valid_count = len(valid_parsed) - invalid_parsed_count
            invalid_count = invalid_parsed_count
            
        # 3. Standardized changes (compare df_original with df_cleaned)
        standardized_count = 0
        if col in df_original.columns:
            for idx in df_cleaned.index:
                if idx in df_original.index:
                    orig_val = df_original[col].loc[idx]
                    clean_val = df_cleaned[col].loc[idx]
                    if pd.notna(orig_val) and pd.notna(clean_val):
                        if str(orig_val) != str(clean_val):
                            standardized_count += 1
                    elif (pd.isna(orig_val) and pd.notna(clean_val)) or (pd.notna(orig_val) and pd.isna(clean_val)):
                        standardized_count += 1
                        
        # 4. Unresolved issues (remaining invalid + missing values)
        unresolved_count = invalid_count + missing_count
        
        column_validation_report[col] = {
            "semantic_type": sem_type,
            "valid": valid_count,
            "invalid": invalid_count,
            "standardized": standardized_count,
            "unresolved": unresolved_count
        }
        
        # Legacy column_audit support
        is_clean = (missing_count == 0) and (invalid_count == 0)
        column_audit[col] = {
            "missing_values": missing_count,
            "outliers": invalid_count if sem_type == "Numeric" else 0,
            "casing_inconsistencies": 0,
            "is_clean": is_clean
        }
        
    # 5. Age Validation Report (Outliers and Domain invalid values list)
    age_cols = [c for c in df_original.columns if c.lower().strip() == "age"]
    if age_cols:
        col = age_cols[0]
        orig_series = df_original[col].dropna()
        orig_vals = []
        for x in orig_series:
            try:
                if str(x).strip().lower() not in DISGUISED_PATTERNS:
                    orig_vals.append(float(x))
            except:
                pass
                
        iqr_outliers = []
        domain_invalid = []
        
        if orig_vals:
            s_vals = pd.Series(orig_vals)
            q1 = s_vals.quantile(0.25)
            q3 = s_vals.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            for val in orig_vals:
                is_outlier = val < lower or val > upper
                is_domain_inv = val < 0 or val > 100
                report_val = int(val) if val % 1 == 0 else val
                
                if is_outlier:
                    if report_val not in iqr_outliers:
                        iqr_outliers.append(report_val)
                if is_domain_inv:
                    if report_val not in domain_invalid:
                        domain_invalid.append(report_val)
                        
        comparison["age_validation"] = {
            "iqr_outliers": sorted(iqr_outliers),
            "domain_invalid": sorted(domain_invalid),
            "has_age": True
        }
        
    return {
        "comparison": comparison,
        "column_audit": column_audit,
        "column_validation_report": column_validation_report,
        "clean_profile": clean_profile,
        "clean_anomalies": clean_quality  # pass quality dict as anomalies context
    }
