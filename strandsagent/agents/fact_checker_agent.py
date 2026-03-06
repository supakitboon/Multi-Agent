import io
import os

from botocore.config import Config
from strands import Agent, tool
from strands.models import BedrockModel

_BOTO_CONFIG = Config(read_timeout=300, connect_timeout=60)

from tools.csv_tools import _download_csv
from tools.memory_tools import _get_analysis

_SYSTEM_PROMPT = """You are a rigorous but kind data fact-checker helping a student learn.

You will receive:
- The student's claim
- A stored analysis summary of their dataset
- A preview of their raw CSV data (first 20 rows + shape info)

Your job:
1. Look at the analysis summary and CSV preview
2. Determine if the claim is CORRECT, WRONG, or AMBIGUOUS
3. Explain your reasoning with specific numbers from the data
4. If the claim involves a calculation, show what pandas code you would use
   and what the result is (based on the data you can see)
5. Be constructive, never dismissive — help the student learn"""


def _fact_check_claim(user_id: str, student_claim: str) -> str:
    """Plain helper: verify a student's claim against their dataset."""
    import pandas as pd

    # Pre-fetch everything before calling the LLM — no tool calls needed
    try:
        analysis = _get_analysis(user_id)
        print(f"[fact_check] Got analysis for '{user_id}': {len(analysis)} chars", flush=True)
    except Exception as e:
        print(f"[fact_check] ERROR retrieving analysis: {e}", flush=True)
        return f"Error retrieving analysis: {e}"

    try:
        csv_content = _download_csv(user_id)
        print(f"[fact_check] Got CSV for '{user_id}': {len(csv_content)} chars", flush=True)
    except Exception as e:
        print(f"[fact_check] ERROR downloading CSV: {e}", flush=True)
        return f"Error downloading CSV: {e}"

    # Build a concise data preview (full data is too large for the prompt)
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        preview_lines = [
            f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
            f"Columns: {', '.join(df.columns.tolist())}",
            "",
            "First 20 rows:",
            df.head(20).to_string(index=False),
            "",
            "Descriptive statistics:",
            df.describe(include="all").to_string(),
        ]
        data_preview = "\n".join(preview_lines)
    except Exception:
        data_preview = csv_content[:5000]

    try:
        agent = Agent(
            model=BedrockModel(
                model_id="us.anthropic.claude-sonnet-4-6",
                region_name=os.environ.get("AWS_REGION", "us-east-2"),
                boto_client_config=_BOTO_CONFIG,
            ),
            system_prompt=_SYSTEM_PROMPT,
            tools=[],
        )

        prompt = (
            f"Student '{user_id}' claims: \"{student_claim}\"\n\n"
            f"=== STORED ANALYSIS ===\n{analysis}\n\n"
            f"=== DATA PREVIEW ===\n{data_preview}\n\n"
            "Verify this claim. Show what pandas code would test it and "
            "explain the result educationally."
        )

        result = str(agent(prompt))
        print(f"[fact_check] LLM response: {len(result)} chars", flush=True)
        return result
    except Exception as e:
        print(f"[fact_check] ERROR in LLM call: {e}", flush=True)
        return f"Error during fact-check LLM call: {e}"


@tool
def fact_check_claim(user_id: str, student_claim: str) -> str:
    """
    Verify or refute a student's claim against their stored dataset.
    Pre-fetches all data, then uses a single LLM call to evaluate.

    Args:
        user_id: Unique identifier for the student session.
        student_claim: The statement the student made about their dataset.
    """
    return _fact_check_claim(user_id, student_claim)
