import json
import os
from datetime import datetime, timezone

import boto3
from strands import tool

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
MEMORY_ID = os.environ.get("AGENTCORE_MEMORY_ID", "")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-agentcore", region_name=AWS_REGION)
    return _client


def _save_analysis(username: str, summary: dict) -> str:
    """Plain helper: persist analysis summary to AgentCore Memory."""
    _get_client().create_event(
        memoryId=MEMORY_ID,
        actorId=username,
        sessionId=f"{username}-analysis",
        eventTimestamp=datetime.now(timezone.utc),
        payload=[
            {
                "conversational": {
                    "content": {"text": json.dumps(summary)},
                    "role": "TOOL",
                }
            }
        ],
        metadata={
            "type": {"stringValue": "dataset_analysis"},
            "userId": {"stringValue": username},
        },
    )
    return "Analysis saved successfully."


def _get_analysis(username: str) -> str:
    """Plain helper: retrieve most recent analysis summary."""
    response = _get_client().retrieve_memory_records(
        memoryId=MEMORY_ID,
        namespace=username,
        searchCriteria={
            "searchQuery": f"dataset analysis for {username}",
        },
    )
    records = response.get("memoryRecords", [])
    if not records:
        return "No prior analysis found for this user."

    latest = records[0]
    content = latest.get("content", {})
    # Handle both old string format and new structured format
    if isinstance(content, str):
        return content
    text = content.get("text", "")
    return text if text else json.dumps(content)


def _save_plan(username: str, plan: str) -> str:
    """Plain helper: persist a project plan to AgentCore Memory."""
    _get_client().create_event(
        memoryId=MEMORY_ID,
        actorId=username,
        sessionId=f"{username}-plan",
        eventTimestamp=datetime.now(timezone.utc),
        payload=[
            {
                "conversational": {
                    "content": {"text": plan},
                    "role": "TOOL",
                }
            }
        ],
        metadata={
            "type": {"stringValue": "project_plan"},
            "userId": {"stringValue": username},
        },
    )
    return "Plan saved successfully."


def _delete_plan(username: str) -> str:
    """Plain helper: delete a student's project plan by saving an empty marker."""
    _save_plan(username, "")
    return "Plan deleted successfully."


def _get_plan(username: str) -> str:
    """Plain helper: retrieve most recent project plan."""
    response = _get_client().retrieve_memory_records(
        memoryId=MEMORY_ID,
        namespace=username,
        searchCriteria={
            "searchQuery": f"project plan for {username}",
        },
    )
    records = response.get("memoryRecords", [])
    if not records:
        return ""

    latest = records[0]
    content = latest.get("content", {})
    if isinstance(content, str):
        return content
    text = content.get("text", "")
    return text if text else json.dumps(content)


@tool
def save_analysis(username: str, summary: dict) -> str:
    """Persist a dataset analysis summary to AgentCore Memory keyed by username.

    Args:
        username: The student's username used as the storage key.
        summary: The analysis results dict to store.
    """
    return _save_analysis(username, summary)


@tool
def get_analysis(username: str) -> str:
    """Retrieve the most recent dataset analysis summary for a username.
    Returns a message if no prior analysis exists.

    Args:
        username: The student's username used as the storage key.
    """
    return _get_analysis(username)
