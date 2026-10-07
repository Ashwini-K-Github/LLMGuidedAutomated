import pandas as pd
import numpy as np

from backend.cleaning.profiler import DISGUISED_PATTERNS, is_valid_phone, tokenize_column_name

def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Module 3: Statistical Analysis
    Detects missing values, duplicates, numeric outliers, and casing discrepancies.
    """
    if df is None:
        return {}

    anomalies = {
        "missing_values": {},
        "duplicates": [],
        "outliers": {},
        "casing_inconsistencies": {},
        "invalid_phones": {}
    }
    
    # 1. Missing Values (NaNs + Disguised Nulls)
    disguised_patterns = DISGUISED_PATTERNS
    for col in df.columns:
        null_indices = df[df[col].isna()].index.tolist()
        
        # Check for disguised nulls
        disguised_indices = []
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col].dtype):
            series_clean = df[col].dropna()
            for idx, val in series_clean.items():
                if str(val).strip().lower() in disguised_patterns:
                    disguised_indices.append(idx)
                    
        total_missing_indices = null_indices + disguised_indices
        
        if len(total_missing_indices) > 0:
            anomalies["missing_values"][col] = {
                "count": len(total_missing_indices),
                "nan_count": len(null_indices),
                "disguised_count": len(disguised_indices),
                "indices": total_missing_indices[:50],  # cap indices to avoid huge payloads
                "pct": round((len(total_missing_indices) / len(df)) * 100, 2)
            }
            
    # 2. Duplicate Rows
    dup_mask = df.duplicated(keep="first")
    dup_indices = df[dup_mask].index.tolist()
    anomalies["duplicates"] = dup_indices[:100]  # cap indices
    
    # 3. Numeric Outliers (using IQR)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            series = df[col].dropna()
            if len(series) > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_indices = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
                
                if len(outlier_indices) > 0:
                    outliers_list = []
                    for idx in outlier_indices[:20]: # show first 20 outliers
                        outliers_list.append({
                            "index": idx,
                            "value": float(df.loc[idx, col])
                        })
                        
                    anomalies["outliers"][col] = {
                        "count": len(outlier_indices),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "outliers": outliers_list
                    }
                    
    # 4. Casing / White-space inconsistencies in categorical columns
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col].dtype):
            series = df[col].dropna().astype(str)
            if len(series) > 0:
                unique_vals = series.unique()
                
                # Check for casing differences (e.g. "Male" vs "male")
                cleaned_to_originals = {}
                for val in unique_vals:
                    norm = val.strip().lower()
                    if norm not in cleaned_to_originals:
                        cleaned_to_originals[norm] = []
                    cleaned_to_originals[norm].append(val)
                
                inconsistencies = {}
                for norm, originals in cleaned_to_originals.items():
                    if len(originals) > 1:
                        inconsistencies[norm] = originals
                        
                if len(inconsistencies) > 0:
                    anomalies["casing_inconsistencies"][col] = inconsistencies
    phone_tokens = {"phone", "mobile", "tel", "telephone", "fax", "cell"}
    for col in df.columns:
        if tokenize_column_name(col) & phone_tokens:
            invalid_indices = []
            for idx, val in df[col].items():
                if not is_valid_phone(val):
                    invalid_indices.append(idx)
            if len(invalid_indices) > 0:
                anomalies["invalid_phones"][col] = {
                    "count": len(invalid_indices),
                    "indices": invalid_indices[:50]
                }
                
    return anomalies
