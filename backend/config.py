import os

# In-memory storage for active session
# In production, this would use a database or session store, but for a local M.Sc. prototype, 
# storing in memory is lightweight and extremely fast.
SESSION = {
    "df": None,                # Original pandas DataFrame
    "cleaned_df": None,        # Cleaned pandas DataFrame
    "profiling": None,         # Profile metrics before cleaning
    "anomalies": None,         # Statistical anomalies list
    "suggestions": [],         # Suggestions list proposed by LLM
    "history": []              # Audit log of applied actions and code executed
}

# Supported LLM providers
LLM_PROVIDERS = ["Mock (Default)", "OpenAI", "Anthropic"]
DEFAULT_PROVIDER = "Mock (Default)"
