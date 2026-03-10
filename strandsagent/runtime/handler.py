"""
AgentCore Runtime entry point.

Expected request event shape:
  {
    "username":    "alice",            # from website login — primary user key
    "inputText":   "what do you see?", # student's message
    "csvContent":  "...",              # (optional) raw CSV text on upload
    "csvBase64":   "...",              # (optional) base64-encoded CSV file
    "fileName":    "data.csv",        # (optional) original filename
    "messages":    [                   # (optional) prior conversation turns
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }

Also handles API Gateway proxy events where the CSV is sent as a
base64-encoded multipart/form-data or raw body via POST.

Fallback identity chain for username:
  event["username"]  →  event["identity"]["userId"]  →  "anonymous"

Response shape:
  {
    "statusCode": 200,
    "body": "{\"response\": \"...\", \"messages\": [...]}"
  }
  The frontend should store the returned "messages" array and send it back
  with the next request to maintain conversation continuity.

To run locally:
    python -m runtime.handler
"""

import base64
import csv
import io
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agents.tutor_agent import create_tutor
from agents.planner_agent import create_planner
from tools.csv_tools import dataset_exists, _upload_csv

# NOTE: we used to pre-warm the CodeInterpreter sandbox at import time
# to shave a few seconds off of the first analysis.  That meant the
# sandbox started as soon as the web server came up, even if nobody ever
# asked for an analysis.  To avoid unnecessary infrastructure usage we
# now start the sandbox lazily inside the tool itself (see
# tools/code_interpreter.get_warm_session).  The warmup() helper still
# exists for callers that really want an explicit background start, but
# the runtime no longer invokes it on startup.

MAX_CSV_SIZE = 10 * 1024 * 1024  # 10 MB


def _extract_csv_content(event: dict) -> str:
    """
    Extract CSV text from the event, handling multiple upload formats:
      1. "csvContent"  — raw CSV string (e.g. from Streamlit)
      2. "csvBase64"   — base64-encoded CSV bytes (e.g. from a JS frontend)
      3. API Gateway proxy body — base64 or plain body with CSV payload
    Returns the decoded CSV string, or "" if no CSV was provided.
    Raises ValueError on invalid / oversized files.
    """
    # 1. Raw string (current path — Streamlit, direct callers)
    raw = event.get("csvContent", "")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    raw = raw.strip()
    if raw:
        return _validate_csv(raw)

    # 2. Explicit base64 field (JS frontends, mobile apps)
    b64 = event.get("csvBase64", "")
    if b64:
        return _validate_csv(_decode_base64(b64))

    # 3. API Gateway proxy integration — body may hold the CSV
    body = event.get("body", "")
    if not body:
        return ""

    is_base64 = event.get("isBase64Encoded", False)
    content_type = (event.get("headers") or {}).get("content-type", "")

    # 3a. Multipart form-data (file upload via HTML form / fetch)
    if "multipart/form-data" in content_type:
        raw_body = _decode_base64(body) if is_base64 else body
        return _parse_multipart_csv(raw_body, content_type)

    # 3b. Plain CSV body (Content-Type: text/csv)
    if "text/csv" in content_type:
        decoded = _decode_base64(body) if is_base64 else body
        return _validate_csv(decoded)

    # 3c. JSON body (API posted JSON with csvContent/csvBase64 inside)
    if "application/json" in content_type or body.lstrip().startswith("{"):
        try:
            inner = json.loads(_decode_base64(body) if is_base64 else body)
            if isinstance(inner, dict):
                return _extract_csv_content(inner)  # recurse once
        except (json.JSONDecodeError, ValueError):
            pass

    return ""


def _decode_base64(data: str) -> str:
    """Decode a base64 string to UTF-8 text."""
    try:
        raw_bytes = base64.b64decode(data)
    except Exception as exc:
        raise ValueError(f"Invalid base64 data: {exc}")

    if len(raw_bytes) > MAX_CSV_SIZE:
        raise ValueError(
            f"CSV file too large ({len(raw_bytes) / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_CSV_SIZE / 1024 / 1024:.0f} MB."
        )

    # Try common encodings
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _parse_multipart_csv(body: str, content_type: str) -> str:
    """
    Extract the first CSV file part from a multipart/form-data body.
    Works with API Gateway proxy events.
    """
    # Extract boundary from content-type header
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part.split("=", 1)[1].strip('"')
            break
    if not boundary:
        raise ValueError("Multipart upload missing boundary in Content-Type header.")

    # Encode to bytes for proper splitting
    body_bytes = body.encode("utf-8") if isinstance(body, str) else body
    boundary_bytes = boundary.encode("utf-8")

    parts = body_bytes.split(b"--" + boundary_bytes)
    for part in parts:
        # Skip preamble and closing delimiter
        if not part.strip() or part.strip() == b"--":
            continue

        # Split headers from body at first double newline
        if b"\r\n\r\n" in part:
            header_section, file_data = part.split(b"\r\n\r\n", 1)
        elif b"\n\n" in part:
            header_section, file_data = part.split(b"\n\n", 1)
        else:
            continue

        header_text = header_section.decode("utf-8", errors="replace").lower()

        # Look for a part that has a .csv filename or text/csv content-type
        is_csv = (
            'filename="' in header_text and '.csv' in header_text
        ) or "text/csv" in header_text

        if is_csv:
            # Strip trailing boundary markers
            file_data = file_data.rstrip(b"\r\n-")
            decoded = file_data.decode("utf-8", errors="replace")
            return _validate_csv(decoded)

    raise ValueError("No CSV file found in multipart upload.")


def _validate_csv(text: str) -> str:
    """
    Basic validation that the text looks like CSV:
      - Not empty
      - Within size limit
      - Parseable by csv.reader (at least 2 rows with consistent columns)
    Returns the validated text.
    """
    if not text.strip():
        raise ValueError("Uploaded CSV file is empty.")

    if len(text.encode("utf-8")) > MAX_CSV_SIZE:
        size_mb = len(text.encode("utf-8")) / 1024 / 1024
        raise ValueError(
            f"CSV file too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_CSV_SIZE / 1024 / 1024:.0f} MB."
        )

    try:
        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if header is None or len(header) < 1:
            raise ValueError("CSV file has no header row.")
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("CSV file has a header but no data rows.")
    except csv.Error as exc:
        raise ValueError(f"File does not appear to be valid CSV: {exc}")

    return text


def handler(event: dict, context: object = None) -> dict:
    try:
        username = _extract_username(event)
        message = event.get("inputText", "").strip()
        prior_messages = event.get("messages", [])
        active_agent = event.get("activeAgent", "tutor")
        planner_messages = event.get("plannerMessages", [])

        # Extract & validate CSV from any supported upload format
        try:
            csv_content = _extract_csv_content(event)
        except ValueError as ve:
            return _error(400, str(ve))

        # ── Route to the currently active agent ──────────────────────────
        if active_agent == "planner":
            return _handle_planner(
                username, message, prior_messages, planner_messages
            )
        else:
            return _handle_tutor(
                username, message, prior_messages, planner_messages, csv_content
            )
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def _build_tutor_prompt(
    username: str, message: str, prior_messages: list, csv_content: str
) -> str:
    """Build the prompt sent to the tutor, injecting system context as needed."""
    if csv_content:
        _upload_csv(user_id=username, csv_content=csv_content)
        col_preview = ""
        try:
            header_line = csv_content.split("\n", 1)[0].strip()
            col_preview = f" Columns: {header_line}."
        except Exception:
            pass
        return (
            f"[SYSTEM: The student just uploaded a CSV dataset. "
            f"It has been saved to storage.{col_preview}]\n\n"
            f"Student says: {message}"
        )

    if not prior_messages and dataset_exists(username):
        reminder = (
            "This student already has a dataset stored from a previous session. "
            "Do NOT ask them to upload again — use recall_dataset to retrieve the analysis. "
            "Mention in your response that you still have their data and can continue."
        )
        safe_message = message or "Hello"
        return f"[SYSTEM: {reminder}]\n\nStudent says: {safe_message}"

    return message


def _handle_tutor(
    username: str,
    message: str,
    tutor_messages: list,
    planner_messages: list,
    csv_content: str,
) -> dict:
    """Run the tutor agent.  If it routes to the planner, signal the switch."""
    routing_state = {"switch_to": None, "planner_messages": planner_messages}

    tutor = create_tutor(
        username, prior_messages=tutor_messages, routing_state=routing_state
    )

    full_prompt = _build_tutor_prompt(username, message, tutor_messages, csv_content)
    response = tutor(full_prompt)

    new_active = (
        "planner" if routing_state["switch_to"] == "planner" else "tutor"
    )
    new_planner = routing_state.get("planner_messages", planner_messages)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "response": str(response),
            "messages": list(tutor.messages),
            "plannerMessages": new_planner,
            "activeAgent": new_active,
        }),
    }


def _handle_planner(
    username: str,
    message: str,
    tutor_messages: list,
    planner_messages: list,
) -> dict:
    """Run the planner agent.  If it routes back to the tutor, re-process."""
    routing_state = {"switch_to": None}

    planner = create_planner(
        username, prior_messages=planner_messages, routing_state=routing_state
    )
    response = planner(message)

    if routing_state["switch_to"] == "tutor":
        # Planner decided this isn't a planning question — re-route to tutor.
        # We pass csv_content="" because CSV uploads always go through the
        # tutor path (active_agent resets to "tutor" on new chat).
        tutor = create_tutor(username, prior_messages=tutor_messages)
        tutor_response = tutor(message)
        return {
            "statusCode": 200,
            "body": json.dumps({
                "response": str(tutor_response),
                "messages": list(tutor.messages),
                "plannerMessages": planner_messages,  # keep prior planner state
                "activeAgent": "tutor",
            }),
        }

    return {
        "statusCode": 200,
        "body": json.dumps({
            "response": str(response),
            "messages": tutor_messages,  # tutor messages unchanged
            "plannerMessages": list(planner.messages),
            "activeAgent": "planner",
        }),
    }


def _extract_username(event: dict) -> str:
    """
    Extract the student's username from the event.
    Priority: explicit "username" field → Cognito identity userId → "anonymous"
    """
    return (
        event.get("username")
        or event.get("identity", {}).get("userId")
        or "anonymous"
    )


def _error(status: int, message: str) -> dict:
    return {
        "statusCode": status,
        "body": json.dumps({"error": message}),
    }


# ---------------------------------------------------------------------------
# Local smoke test  (python -m runtime.handler)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["name", "age", "salary", "score"])
    writer.writerow(["Alice", 22, 55000, 88])
    writer.writerow(["Bob", 25, 72000, 72])
    writer.writerow(["Carol", 21, 48000, 95])
    writer.writerow(["Dave", 30, "", 61])   # intentional NaN in salary
    sample_csv = buf.getvalue()

    print("=== Turn 1: Upload CSV ===")
    result1 = handler({
        "username": "alice",
        "inputText": "Hi, I just uploaded my class dataset.",
        "csvContent": sample_csv,
    })
    body1 = json.loads(result1["body"])
    if "error" in body1:
        print("ERROR:", body1["error"])
        sys.exit(1)
    print(body1["response"])

    print("\n=== Turn 2: Vague question (should get options) ===")
    result2 = handler({
        "username": "alice",
        "inputText": "tell me about the data",
        "messages": body1.get("messages", []),
    })
    body2 = json.loads(result2["body"])
    if "error" in body2:
        print("ERROR:", body2["error"])
        sys.exit(1)
    print(body2["response"])

    print("\n=== Turn 3: Student makes a claim (fact-check) ===")
    result3 = handler({
        "username": "alice",
        "inputText": "I think Bob has the highest score",
        "messages": body2.get("messages", []),
    })
    body3 = json.loads(result3["body"])
    if "error" in body3:
        print("ERROR:", body3["error"])
        sys.exit(1)
    print(body3["response"])

    print("\n=== Turn 4: New session, no CSV re-upload ===")
    result4 = handler({
        "username": "alice",
        "inputText": "I'm back. Can we continue with my dataset?",
    })
    body4 = json.loads(result4["body"])
    if "error" in body4:
        print("ERROR:", body4["error"])
        sys.exit(1)
    print(body4["response"])
