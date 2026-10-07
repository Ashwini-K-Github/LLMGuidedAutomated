import pandas as pd
from backend.cleaning.semantic import construct_semantic_prompt
from backend.utils.llm import call_llm_api

def generate_recommendations(df: pd.DataFrame, profile: dict, anomalies: dict, provider: str, api_key: str) -> dict:
    """
    Module 5: LLM Cleaning Recommendation
    Generates recommendations by combining statistical anomalies and semantic profiling prompts.
    """
    if df is None:
        return {}

    # Formulate prompts
    system_prompt, user_prompt = construct_semantic_prompt(df, profile, anomalies)
    
    # Query the selected LLM provider (or run Mock Mode)
    recommendations = call_llm_api(provider, api_key, system_prompt, user_prompt, df)
    
    return recommendations, system_prompt, user_prompt
