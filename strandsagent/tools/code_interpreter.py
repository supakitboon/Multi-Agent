import os
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
    client = CodeInterpreter(AWS_REGION)
    client.start()
    try:
        # Write the CSV into the sandbox
        client.invoke("writeFiles", {
            "content": [{"path": "dataset.csv", "text": csv_content}]
        })

        result = client.invoke("executeCode", {
            "code": code,
            "language": "python",
            "clearContext": False,
        })

        # The result is a list of output items; concatenate text outputs
        outputs = []
        for item in result.get("output", []):
            if item.get("type") == "text":
                outputs.append(item.get("text", ""))
        return "\n".join(outputs) if outputs else "(no output)"
    finally:
        client.stop()
