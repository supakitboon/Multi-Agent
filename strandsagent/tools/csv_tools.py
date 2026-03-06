import os
import boto3
from botocore.exceptions import ClientError
from strands import tool

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")

_s3 = None


def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=AWS_REGION)
    return _s3


def _s3_key(user_id: str) -> str:
    return f"datasets/{user_id}/dataset.csv"


def _upload_csv(user_id: str, csv_content: str) -> str:
    """Plain helper: upload CSV text to S3. Returns the S3 key."""
    key = _s3_key(user_id)
    _get_s3().put_object(Bucket=S3_BUCKET, Key=key, Body=csv_content.encode("utf-8"))
    return key


def _download_csv(user_id: str) -> str:
    """Plain helper: download stored CSV text. Raises if not found."""
    key = _s3_key(user_id)
    try:
        response = _get_s3().get_object(Bucket=S3_BUCKET, Key=key)
        return response["Body"].read().decode("utf-8")
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise FileNotFoundError(
                f"No dataset found for user '{user_id}'. Please upload a CSV first."
            )
        raise


@tool
def upload_csv_to_s3(user_id: str, csv_content: str) -> str:
    """Upload CSV text to S3 for a given user. Returns the S3 key.

    Args:
        user_id: The student's username used as the storage key.
        csv_content: Raw CSV text content to store.
    """
    return _upload_csv(user_id, csv_content)


@tool
def download_csv_from_s3(user_id: str) -> str:
    """Download the stored CSV text for a given user. Raises if not found.

    Args:
        user_id: The student's username used as the storage key.
    """
    return _download_csv(user_id)


def dataset_exists(user_id: str) -> bool:
    """Return True if a CSV has been stored for this user."""
    key = _s3_key(user_id)
    try:
        _get_s3().head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
