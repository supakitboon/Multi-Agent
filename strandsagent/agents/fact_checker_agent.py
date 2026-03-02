import os

from botocore.config import Config
from strands import Agent, tool
from strands.models import BedrockModel

_BOTO_CONFIG = Config(read_timeout=300, connect_timeout=60)

from tools.csv_tools import download_csv_from_s3
from tools.code_interpreter import run_analysis
from tools.memory_tools import get_analysis

_SYSTEM_PROMPT = """You are a rigorous but kind data fact-checker helping a student learn.

When a student makes a claim about their dataset:
1. Call get_analysis to retrieve the stored dataset summary
2. Call download_csv_from_s3 to fetch the raw CSV (decode bytes to string for analysis)
3. Write precise pandas code that directly tests the student's claim
4. Call run_analysis with the CSV content and your verification code
5. Interpret the result honestly:
   - If the claim is CORRECT: confirm it with the supporting numbers
   - If the claim is WRONG: explain what the data actually shows, and why the student might have been mistaken
   - If the claim is AMBIGUOUS: explain what assumptions would make it true or false

Always show your working — what code you ran and what it returned. This teaches students
how to verify claims themselves. Be constructive, never dismissive."""


@tool
def fact_check_claim(user_id: str, student_claim: str) -> str:
    """
    Verify or refute a student's claim against their stored dataset.
    Retrieves the CSV from S3, runs targeted pandas code, and explains the result.

    Args:
        user_id: Unique identifier for the student session.
        student_claim: The statement the student made about their dataset.
    """
    agent = Agent(
        model=BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-6",
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            boto_client_config=_BOTO_CONFIG,
        ),
        system_prompt=_SYSTEM_PROMPT,
        tools=[get_analysis, download_csv_from_s3, run_analysis],
    )

    prompt = (
        f"Student '{user_id}' claims: \"{student_claim}\"\n\n"
        "Verify this claim against their stored dataset. "
        "Show the pandas code you used and explain the result educationally."
    )

    return str(agent(prompt))
