"""
AgentCore Runtime entry point.

Expected request event shape:
  {
    "username":    "alice",            # from website login — primary user key
    "inputText":   "what do you see?", # student's message
    "csvContent":  "...",              # (optional) raw CSV text on upload
    "messages":    [                   # (optional) prior conversation turns
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }

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

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.tutor_agent import create_tutor  # noqa: E402


def handler(event: dict, context: object = None) -> dict:
    """AgentCore Runtime handler."""
    try:
        username = _extract_username(event)
        message = event.get("inputText", "").strip()
        csv_content = event.get("csvContent", "").strip()
        prior_messages = event.get("messages", [])

        if not message and not csv_content:
            return _error(400, "inputText or csvContent is required")

        tutor = create_tutor(username, prior_messages=prior_messages)

        if csv_content:
            full_prompt = f"[CSV_UPLOAD]\n{csv_content}"
            if message:
                full_prompt += f"\n\n[STUDENT_MESSAGE]\n{message}"
        else:
            full_prompt = message

        response = tutor(full_prompt)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "response": str(response),
                # Return updated conversation history so the frontend can
                # send it back next turn, preserving context across requests.
                "messages": list(tutor.messages),
            }),
        }

    except FileNotFoundError as e:
        return _error(400, str(e))
    except Exception as e:
        return _error(500, f"Internal error: {str(e)}")


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
