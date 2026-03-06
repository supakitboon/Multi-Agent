import atexit
import os
import threading
import time
from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter
from strands import tool

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")


class CodeInterpreterSession:
    """A reusable Code Interpreter sandbox session.

    Use as a context manager to keep the sandbox alive across multiple
    code executions (e.g. profile -> clean -> normalize).
    """

    def __init__(self):
        self._client = CodeInterpreter(AWS_REGION)
        self._started = False
        self._t0 = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        if not self._started:
            self._t0 = time.time()
            print(f"[CodeInterpreter] Starting sandbox in {AWS_REGION}...", flush=True)
            self._client.start()
            self._started = True
            print(f"[CodeInterpreter] Sandbox ready ({time.time() - self._t0:.1f}s)", flush=True)

    def stop(self):
        if self._started:
            print("[CodeInterpreter] Stopping sandbox...", flush=True)
            self._client.stop()
            self._started = False
            print(f"[CodeInterpreter] Done ({time.time() - self._t0:.1f}s)", flush=True)

    def upload_csv(self, csv_content: str, filename: str = "dataset.csv"):
        print(f"[CodeInterpreter] Writing {filename} to sandbox...", flush=True)
        self._client.invoke("writeFiles", {
            "content": [{"path": filename, "text": csv_content}]
        })
        print(f"[CodeInterpreter] {filename} written ({time.time() - self._t0:.1f}s)", flush=True)

    def run_code(self, code: str) -> str:
        print("[CodeInterpreter] Executing code...", flush=True)
        result = self._client.invoke("executeCode", {
            "code": code,
            "language": "python",
            "clearContext": False,
        })
        print(f"[CodeInterpreter] Code executed ({time.time() - self._t0:.1f}s)", flush=True)
        outputs = []
        for item in result.get("output", []):
            if item.get("type") == "text":
                outputs.append(item.get("text", ""))
        return "\n".join(outputs) if outputs else "(no output)"


# ---------------------------------------------------------------------------
# Warm singleton sandbox — started once in the background, reused across calls
# ---------------------------------------------------------------------------
_warm_session: CodeInterpreterSession | None = None
_warm_lock = threading.Lock()
_warm_ready = threading.Event()


def _start_warm_sandbox():
    """Start the singleton sandbox in a background thread."""
    global _warm_session
    session = CodeInterpreterSession()
    session.start()
    _warm_session = session
    _warm_ready.set()


def get_warm_session() -> CodeInterpreterSession:
    """Return the warm singleton sandbox, starting it if needed."""
    global _warm_session
    with _warm_lock:
        if _warm_session is None or not _warm_session._started:
            _warm_ready.clear()
            t = threading.Thread(target=_start_warm_sandbox, daemon=True)
            t.start()
    _warm_ready.wait()
    return _warm_session


def warmup():
    """Pre-warm the sandbox in the background (non-blocking).
    Call this at import time or during app startup."""
    with _warm_lock:
        if _warm_session is None:
            t = threading.Thread(target=_start_warm_sandbox, daemon=True)
            t.start()


def _cleanup():
    if _warm_session is not None and _warm_session._started:
        _warm_session.stop()


atexit.register(_cleanup)


@tool
def run_analysis(csv_content: str, code: str) -> str:
    """
    Upload csv_content to an AgentCore Code Interpreter sandbox, execute
    the provided pandas code, and return the combined stdout output.

    Uses a warm reusable sandbox to avoid cold-start latency.
    """
    session = get_warm_session()
    session.upload_csv(csv_content)
    return session.run_code(code)
