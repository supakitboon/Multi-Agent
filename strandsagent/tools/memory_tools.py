import json
import os
from datetime import datetime, timezone

import boto3
from strands import tool

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
    return _client


@tool
def save_analysis(username: str, summary: dict) -> str:
    """Persist a dataset analysis summary to AgentCore Memory keyed by username.

    Args:
        username: The student's username used as the storage key.
        summary: The analysis results dict to store.
    """
    _get_client().create_event(
        memoryId=MEMORY_ID,
        event={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "dataset_analysis",
            "userId": username,
            "content": json.dumps(summary),
        },
    )
    return "Analysis saved successfully."


@tool
def get_analysis(username: str) -> str:
    """Retrieve the most recent dataset analysis summary for a username.
    Returns a message if no prior analysis exists.

    Args:
        username: The student's username used as the storage key.
    """
    response = _get_client().retrieve_memory_records(
        memoryId=MEMORY_ID,
        query=f"dataset analysis for username {username}",
    )
    records = response.get("memoryRecords", [])
    if not records:
        return "No prior analysis found for this user."

    latest = records[0]
    content = latest.get("content", "{}")
    return content
