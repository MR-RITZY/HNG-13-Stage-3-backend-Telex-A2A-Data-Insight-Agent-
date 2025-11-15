from textwrap import dedent
from data_insight_agent.utils import ANALYTIC_INTENTS, ANALYTICAL_KEYWORDS
from data_insight_agent.ai_schema import AIParsedInstruction

def get_prompt(data: dict) -> str :
    return dedent(f"""
    You are a strict and smart AI assistant for data analysis. 
    Analyze the user request and dataset metadata, and output a single JSON  that conforms strictly and totally with schema specified
    that strictly matches the following keys:

    - intent (list of analytic tasks: summary, math, correlation, regression, anomaly, visualization, unknown)
    - operations (dict specifying actions per intent, e.g., math: ["sum", "mean", quantile: [0.40, 0.75]])
    - focus_columns (list of relevant columns)
    - group_by (list of columns for grouping)
    - drop (optional: "drop null" or "drop duplicates")
    - fill (optional: "ffill", "bfill", or value)
    - sort (optional: list of "ascending"/"descending")
    - filters (dict of column filters)
    - analysis_explanation (optional string explaining the analysis)
    - confidence (float between 0.0 and 1.0)

    --- Allowed analytic intents:
    {ANALYTIC_INTENTS}

    --- Allowed operations per intent:
    {ANALYTICAL_KEYWORDS}

     --- Schema which your response will be validated against:
    {AIParsedInstruction}

    --- Dataset metadata:
    {data}

    --- Example valid output:
    {{
        "intent": ["math"],
        "operations": {{"math": ["sum", "mean"]}},
        "focus_columns": ["sales", "profit"],
        "filters": {{"region": "West"}},
        "confidence": 0.88
    }}

    --- Rules:
    1. Output must be valid JSON with all required keys.
    2. Use only the allowed operations for each intent.
    3. For every intent you pick, pick their corresponding required operations for reasonable analysis and vice versa
    4. Populate optional fields with defaults if empty.
    5. If unsure, return: {{"intent": ["unknown"], "confidence": 0.0}}.
    6. Do not include explanations, markdown, or extra text.
    """)
