import pandas as pd
import json

def construct_semantic_prompt(df: pd.DataFrame, profile: dict, anomalies: dict) -> tuple:
    """
    Module 4: LLM Semantic Analysis Prompt Formulation
    Constructs the system and user prompts to guide the LLM in finding semantic quality problems.
    """
    
    # 1. Build a brief schema overview
    schema_overview = {}
    for col in profile.get("columns", []):
        schema_overview[col["name"]] = {
            "type": col["type"],
            "missing_pct": col["missing_pct"],
            "unique_count": col["unique_count"],
            "sample_values": col["sample_values"][:8] # sample unique values
        }
        
    system_prompt = """You are Cocoon, an advanced automated data cleaning assistant.
Your goal is to perform semantic analysis on a uploaded dataset and recommend cleaning actions.
You must analyze the dataset's columns, values, and statistical profile to detect the following error categories:
1. String Outliers: Spelling mistakes, abbreviations, inconsistent categorical values (e.g., 'NY' vs 'New York').
2. Pattern Outliers: Multiple date formats or pattern inconsistencies in a column.
3. Disguised Missing Values: Values like 'N/A', 'Unknown', 'None', '-', or 'NULL' stored as normal strings.
4. Column Type Detection: Columns stored as string/object that should be integer, float, datetime, or boolean.
5. Numeric Outliers: Determine if statistical outliers represent errors or valid extreme values.
6. Functional Dependency Violations: Inconsistent relationships (e.g., same ZIP code mapped to different states).
7. Duplicate Records: Redundant records that represent the same entity and should be deduplicated.
8. Column Uniqueness: Validation of columns that are expected to contain unique values (like IDs).

You MUST respond with a valid JSON object ONLY. Do not include markdown codeblocks or explanations outside the JSON. The JSON structure must match this schema:
{
  "columns": [
    {
      "column": "string (name of the column)",
      "type_cast": "string (one of: 'int', 'float', 'str', 'datetime', 'bool', or null)",
      "disguised_nulls": ["list of string values in this column that represent missing/null values, or empty list"],
      "value_mappings": {
         "raw_incorrect_value_1": "standardized_correct_value_1"
      },
      "imputation_strategy": "string (one of: 'mean', 'median', 'mode', 'constant', 'drop', or null)",
      "imputation_value": "any (the specific constant/imputation value to use if strategy is 'constant' or 'median'/'mean', or null)",
      "outlier_strategy": "string (one of: 'clip', 'remove', 'keep', or null)",
      "explanation": "string (short human-readable explanation of why this action is recommended)"
    }
  ],
  "drop_duplicates": boolean (true if duplicate records should be removed),
  "explanation": "string (general summary of the semantic analysis of this dataset)"
}
"""

    user_prompt = f"""Please analyze the following dataset profile and detected anomalies, and provide cleaning recommendations.

### DATASET STRUCTURE OVERVIEW
Total Rows: {profile.get("num_rows")}
Total Columns: {profile.get("num_cols")}
Total Duplicates: {profile.get("duplicate_rows")}

### COLUMNS AND DATA SAMPLES
{json.dumps(schema_overview, indent=2)}

### DETECTED ANOMALIES (STATISTICAL)
Missing Values Info:
{json.dumps(anomalies.get("missing_values"), indent=2)}

Numeric Outliers Detected:
{json.dumps(anomalies.get("outliers"), indent=2)}

Casing Inconsistencies Detected:
{json.dumps(anomalies.get("casing_inconsistencies"), indent=2)}

Provide your cleaning recommendations in the requested JSON format. Ensure value mappings target actual raw values present in the sample. If a column does not require cleaning, omit it from the 'columns' list.
"""

    return system_prompt, user_prompt
