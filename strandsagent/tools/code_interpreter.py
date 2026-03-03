import os
import time
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from strands import tool

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


@tool
def run_analysis(csv_content: str, code: str) -> str:
    """
    Upload csv_content to an AgentCore Code Interpreter sandbox, execute
    the provided pandas code, and return the combined stdout output.

    The sandbox is started fresh per call and stopped when done.
    """
    t0 = time.time()
    print(f"[CodeInterpreter] Starting sandbox in {AWS_REGION}...", flush=True)
    client = CodeInterpreter(AWS_REGION)
    client.start()
    print(f"[CodeInterpreter] Sandbox ready ({time.time() - t0:.1f}s)", flush=True)
    try:
        print("[CodeInterpreter] Writing CSV to sandbox...", flush=True)
        client.invoke("writeFiles", {
            "content": [{"path": "dataset.csv", "text": csv_content}]
        })
        print(f"[CodeInterpreter] CSV written ({time.time() - t0:.1f}s)", flush=True)

        print("[CodeInterpreter] Executing analysis code...", flush=True)
        result = client.invoke("executeCode", {
            "code": code,
            "language": "python",
            "clearContext": False,
        })
        print(f"[CodeInterpreter] Code executed ({time.time() - t0:.1f}s)", flush=True)

        # The result is a list of output items; concatenate text outputs
        outputs = []
        for item in result.get("output", []):
            if item.get("type") == "text":
                outputs.append(item.get("text", ""))
        return "\n".join(outputs) if outputs else "(no output)"
    except Exception as e:
        print(f"[CodeInterpreter] ERROR: {e}", flush=True)
        raise
    finally:
        print("[CodeInterpreter] Stopping sandbox...", flush=True)
        client.stop()
        print(f"[CodeInterpreter] Done ({time.time() - t0:.1f}s)", flush=True)
