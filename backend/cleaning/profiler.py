import pandas as pd
import numpy as np
import re
import warnings

# Disguised null patterns used across the framework
DISGUISED_PATTERNS = ["n/a", "unknown", "none", "-", "null", "unknown value", "na", "?", "missing", "nan", "nil", ""]

def is_valid_phone(val) -> bool:
    if pd.isna(val):
        return False
    v_str = str(val).strip()
    if v_str.lower() in DISGUISED_PATTERNS:
        return False
    if v_str.endswith(".0"):
        v_str = v_str[:-2]
    if "e+" in v_str.lower():
        try:
            v_str = str(int(float(val)))
        except:
            pass
    # Remove permitted formatting characters (spaces, hyphens, parentheses, brackets, dots, pluses)
    cleaned = re.sub(r'[\s\-()\[\]\.\+]', '', v_str)
    return cleaned.isdigit() and len(cleaned) == 10

def is_valid_email(val) -> bool:
    if pd.isna(val):
        return False
    v_str = str(val).strip()
    if v_str.lower() in DISGUISED_PATTERNS:
        return False
    # Standard email regex pattern
    email_regex = r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
    return bool(re.match(email_regex, v_str))

# --- Column-name tokenisation -------------------------------------------------
# Identifier detection used to substring-match ("id" in "humidity"), which
# misclassified ordinary measurements. We now match whole name tokens instead.
_ID_TOKENS = {
    "id", "ids", "identifier", "uuid", "guid", "pk", "key", "idx", "index",
    "no", "num", "number", "code", "acct", "account", "sn", "serial",
    "ref", "reference", "roll", "regno", "barcode",
}
# Glued forms such as "StudentID" / "AccountNo" / "Barcode" that survive
# lower-casing without a separator.
_ID_SUFFIXES = ("id", "no", "code", "num", "key", "idx")

_EMAIL_TOKENS = {"email", "emails", "mail", "emailid", "e"}
_PHONE_TOKENS = {"phone", "phones", "mobile", "tel", "telephone", "fax", "cell",
                 "contact", "whatsapp", "msisdn"}
_DATE_TOKENS = {"date", "dates", "time", "timestamp", "datetime", "year", "month",
                "day", "dob", "birth", "birthday", "joining", "joined", "admission",
                "enrollment", "enrolment", "created", "updated", "modified",
                "expiry", "expires", "shipped", "ordered", "start", "end", "due"}
# Names that mark a column as a label rather than a measurement, so an integer
# code such as Pclass 1/2/3 or Grade 1..5 is treated as a category.
_CATEGORICAL_TOKENS = {"class", "type", "category", "categories", "group", "grade",
                       "level", "status", "flag", "gender", "sex", "region", "zone",
                       "band", "tier", "segment", "rating", "rank", "label", "state"}

_TEXT_TOKENS = {"comment", "comments", "review", "reviews", "description", "desc",
                "feedback", "note", "notes", "text", "msg", "message", "summary",
                "details", "detail", "reason", "remarks", "name", "address",
                "title", "bio", "about"}


def tokenize_column_name(col_name) -> set:
    """Splits a column name into lowercase word tokens.

    Handles snake_case, kebab-case, spaces and camelCase, so 'StudentID' yields
    {'student', 'id'} while 'Humidity' yields {'humidity'} and never {'id'}.
    """
    name = str(col_name)
    # Insert boundaries at camelCase transitions before splitting.
    spaced = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', name)
    spaced = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', spaced)
    parts = re.split(r'[^A-Za-z0-9]+', spaced)
    return {p.lower() for p in parts if p}


def _name_signals_identifier(col_name) -> bool:
    tokens = tokenize_column_name(col_name)
    if tokens & _ID_TOKENS:
        return True
    # Glued suffix form, e.g. "studentid", "accountno", "barcode".
    glued = re.sub(r'[^a-z0-9]', '', str(col_name).lower())
    for suffix in _ID_SUFFIXES:
        if glued.endswith(suffix) and len(glued) > len(suffix) + 2:
            return True
    return False


def infer_semantic_type(col_name, series: pd.Series) -> str:
    """
    Dynamically infers the semantic type of a column from the column name and the
    actual value characteristics (dtype, cardinality, uniqueness, patterns).

    Ordering matters: the structural checks (Boolean/Email/Date/Phone) run first,
    then the column is resolved as Numeric or as a string type. Identifier is
    deliberately the *narrowest* category - a float column is never an identifier,
    and a numeric column needs both a naming signal and near-total uniqueness.
    """
    col_name = str(col_name)
    c_low = col_name.lower().strip()
    tokens = tokenize_column_name(col_name)

    # Get non-null, non-disguised values
    series_clean = series.dropna()
    series_val = series_clean[series_clean.apply(lambda x: str(x).strip().lower() not in DISGUISED_PATTERNS)]
    val_len = len(series_val)

    if val_len == 0:
        # Fallback to pandas type mapping
        if pd.api.types.is_bool_dtype(series):
            return "Boolean"
        elif pd.api.types.is_datetime64_any_dtype(series):
            return "Date"
        elif pd.api.types.is_numeric_dtype(series):
            return "Numeric"
        return "Categorical"

    unique_count = series_val.nunique()
    unique_pct = (unique_count / val_len) * 100 if val_len > 0 else 0.0
    string_values = series_val.astype(str).str.strip()
    avg_len = string_values.str.len().mean()

    is_pd_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    is_pd_datetime = pd.api.types.is_datetime64_any_dtype(series)

    # 1. Boolean - unambiguous, so resolve it before anything else.
    boolean_values = {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}
    unique_lowered_vals = set(string_values.str.lower().unique())
    if unique_lowered_vals.issubset(boolean_values) and len(unique_lowered_vals) <= 2:
        return "Boolean"

    # A pandas datetime column needs no further evidence.
    if is_pd_datetime:
        return "Date"

    # How many values parse as plain numbers? Used to gate the string-pattern checks.
    numeric_matches = 0
    for val in string_values:
        try:
            float(val)
            numeric_matches += 1
        except ValueError:
            pass
    numeric_ratio = numeric_matches / val_len
    is_numeric = is_pd_numeric or numeric_ratio > 0.85

    # 2. Email - requires actual '@' evidence, so 'Mailing_Address' is not an Email.
    email_matches = string_values.apply(is_valid_email).sum()
    email_ratio = email_matches / val_len
    at_ratio = string_values.str.contains("@", regex=False).sum() / val_len
    is_email_name = bool(tokens & _EMAIL_TOKENS) and ("mail" in c_low)
    if not is_pd_numeric and (email_ratio > 0.6 or (is_email_name and at_ratio > 0.3)):
        return "Email"

    # 3. Date - skipped for numeric columns unless the name says otherwise, so a
    #    'Price' of 2023.5 is never read as a year.
    is_date_name = bool(tokens & _DATE_TOKENS)
    if not (is_numeric and not is_date_name):
        date_matches = 0
        sample_size = min(val_len, 30)
        sample_vals = string_values.sample(sample_size, random_state=42) if val_len > sample_size else string_values
        for val in sample_vals:
            if val.isdigit() and len(val) < 6:
                continue  # avoid short codes / integers being categorized as dates
            try:
                if any(s in val for s in ['-', '/', '.', ':']) or len(val) >= 8:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(val, errors='raise')
                    date_matches += 1
            except Exception:
                pass
        date_ratio = date_matches / sample_size if sample_size > 0 else 0.0
        if (is_date_name and date_ratio > 0.4) or date_ratio > 0.8:
            return "Date"

    id_name = _name_signals_identifier(col_name)

    # 4. Phone Number - a naming signal alone is not enough, otherwise a column
    #    like 'Contact_Person' would be treated as phone data and nullified.
    #    Conversely a 10-digit account number must not be read as a phone just
    #    because it happens to have ten digits, so an identifier name wins.
    phone_matches = string_values.apply(is_valid_phone).sum()
    phone_ratio = phone_matches / val_len
    digit_share = string_values.str.count(r'\d').sum() / max(string_values.str.len().sum(), 1)
    is_phone_name = bool(tokens & _PHONE_TOKENS)
    if (is_phone_name and digit_share > 0.5) or (phone_ratio > 0.6 and not id_name):
        return "Phone Number"

    # 5. Numeric vs Identifier.
    #    A numeric column is only an identifier when the *name* says so, the values
    #    are whole numbers, and they are essentially all distinct. Continuous
    #    measurements (Salary, GPA, Humidity, Latitude) therefore stay Numeric
    #    regardless of how unique they happen to be.
    if is_numeric:
        # 70% rather than 95% so a small dataset with a few repeated key values
        # (or duplicate rows) still resolves its integer key as an Identifier.
        if id_name and unique_pct >= 70.0:
            try:
                parsed = pd.to_numeric(string_values, errors='coerce').dropna()
                if not parsed.empty and (parsed % 1 == 0).all():
                    return "Identifier"
            except Exception:
                pass
        # A code-like or label-like name with few distinct values is a category,
        # not a measurement (e.g. Zip_Code, Ward_No, Pclass, Grade).
        is_label_name = bool(tokens & _CATEGORICAL_TOKENS) or any(
            re.sub(r'[^a-z0-9]', '', col_name.lower()).endswith(t) for t in ("class", "type", "grade", "status")
        )
        if (id_name or is_label_name) and unique_count <= 40 and unique_pct < 20.0:
            return "Categorical"
        return "Numeric"

    # 6. Identifier for non-numeric values (e.g. 'STU1001', UUIDs, order refs).
    has_whitespace = bool(string_values.str.contains(r'\s').any())
    has_digit = bool(string_values.str.contains(r'\d').any())
    if id_name and unique_pct >= 70.0 and not has_whitespace:
        return "Identifier"
    if unique_pct >= 95.0 and avg_len <= 32 and not has_whitespace and has_digit:
        return "Identifier"

    # 7. Free Text
    is_text_name = bool(tokens & _TEXT_TOKENS)
    words_count = string_values.str.split().str.len().mean()
    if is_text_name or (avg_len > 30 and words_count > 3 and unique_pct > 40):
        return "Free Text"

    # 8. Categorical - lower cardinality string classifications.
    if unique_count <= 40 or (unique_pct < 20.0 and unique_count <= 250):
        return "Categorical"

    # Fallback to Free Text for safety if cardinality is too high.
    return "Free Text"

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Module 2: Dataset Profiling
    Analyzes the structural and content characteristics of the dataset.
    """
    if df is None:
        return {}

    num_rows = len(df)
    num_cols = len(df.columns)
    
    # Calculate memory usage
    memory_usage_bytes = df.memory_usage(deep=True).sum()
    if memory_usage_bytes < 1024:
        memory_usage_str = f"{memory_usage_bytes} B"
    elif memory_usage_bytes < 1024 * 1024:
        memory_usage_str = f"{memory_usage_bytes / 1024:.2f} KB"
    else:
        memory_usage_str = f"{memory_usage_bytes / (1024 * 1024):.2f} MB"
        
    # Analyze columns
    columns_profile = []
    
    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        
        # Check for disguised nulls
        disguised_count = 0
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series.dtype):
            disguised_count = int(sum(1 for x in series.dropna() if str(x).strip().lower() in DISGUISED_PATTERNS))
            
        missing_pct = float(((missing_count + disguised_count) / num_rows) * 100) if num_rows > 0 else 0.0
        
        unique_vals = series.dropna().unique()
        num_unique = len(unique_vals)
        unique_pct = float((num_unique / num_rows) * 100) if num_rows > 0 else 0.0
        
        # Sample values
        sample_vals = [str(x) for x in unique_vals[:8]]
        
        # Dynamic Semantic Type Inference
        col_type = infer_semantic_type(col, series)
        
        col_data = {
            "name": col,
            "type": col_type,
            "missing_count": missing_count,
            "disguised_nulls_count": disguised_count,
            "missing_pct": round(missing_pct, 2),
            "unique_count": num_unique,
            "unique_pct": round(unique_pct, 2),
            "sample_values": sample_vals,
            "stats": {}
        }
        
        # Add email/phone validation indicators if semantic type matches
        if col_type == "Phone Number":
            valid_count = int(sum(1 for x in series if is_valid_phone(x)))
            invalid_count = len(series) - valid_count
            col_data["phone_validation"] = {
                "valid": valid_count,
                "invalid": invalid_count
            }
        elif col_type == "Email":
            valid_count = int(sum(1 for x in series if is_valid_email(x)))
            invalid_count = len(series) - valid_count
            col_data["email_validation"] = {
                "valid": valid_count,
                "invalid": invalid_count
            }
        
        # Gather statistical metrics
        if pd.api.types.is_numeric_dtype(series):
            col_data["stats"] = {
                "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
                "std": float(series.std()) if not pd.isna(series.std()) else None,
                "min": float(series.min()) if not pd.isna(series.min()) else None,
                "max": float(series.max()) if not pd.isna(series.max()) else None,
                "median": float(series.median()) if not pd.isna(series.median()) else None
            }
        else:
            mode_val = series.mode()
            top_val = str(mode_val.iloc[0]) if not mode_val.empty else None
            top_freq = int(series.value_counts().iloc[0]) if not series.empty and len(series.value_counts()) > 0 else 0
            col_data["stats"] = {
                "top": top_val,
                "frequency": top_freq
            }
            
        columns_profile.append(col_data)
        
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "memory_usage": memory_usage_str,
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns_profile
    }
