"""
S3-backed chat history persistence.

Stores each conversation as a JSON file at:
    chats/{username}/{chat_id}.json

Reuses the same S3 bucket and client pattern as csv_tools.py.
"""

import json
import os
from datetime import datetime, timezone

import boto3

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

_s3 = None


def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=AWS_REGION)
    return _s3


def _chat_key(username: str, chat_id: str) -> str:
    return f"chats/{username}/{chat_id}.json"


def save_chat(username, chat_id, title, agent_messages, chat_display, created_at=None):
    """Save (or update) a chat session to S3."""
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "chat_id": chat_id,
        "title": title,
        "created_at": created_at or now,
        "updated_at": now,
        "agent_messages": agent_messages,
        "chat_display": chat_display,
    }
    key = _chat_key(username, chat_id)
    _get_s3().put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(data, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return data


def load_chat(username, chat_id):
    """Load a single chat session from S3."""
    key = _chat_key(username, chat_id)
    resp = _get_s3().get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(resp["Body"].read().decode("utf-8"))


def list_chats(username):
    """Return list of {chat_id, title, updated_at} sorted newest first."""
    prefix = f"chats/{username}/"
    paginator = _get_s3().get_paginator("list_objects_v2")
    chats = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            try:
                resp = _get_s3().get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                data = json.loads(resp["Body"].read().decode("utf-8"))
                chats.append({
                    "chat_id": data["chat_id"],
                    "title": data.get("title", "Untitled"),
                    "updated_at": data.get("updated_at", ""),
                })
            except Exception:
                continue
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    return chats


def delete_chat(username, chat_id):
    """Delete a chat session from S3."""
    key = _chat_key(username, chat_id)
    _get_s3().delete_object(Bucket=S3_BUCKET, Key=key)
