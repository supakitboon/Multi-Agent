import os
from dotenv import load_dotenv
from vertexai.preview import reasoning_engines
import vertexai

# 1. Load Env & Init
load_dotenv()
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

# 2. Import your agents
# We only need to import the root_agent; Python picks up the rest automatically.
from agenttest.agent import root_agent

print("🚀 Deploying Multi-Agent System...")

# 3. Deploy
# We deploy the 'root_agent', which carries the sub-agents inside it.
remote_agent = reasoning_engines.ReasoningEngine.create(
    root_agent,
    requirements=[
        "google-cloud-aiplatform[agent_engines,adk]",
        "python-dotenv",
    ],
    display_name="Travel-Multi-Agent-System",
    description="Orchestrator agent that manages weather and math sub-agents."
)

print(f"✅ Deployment Complete!")
print(f"Agent ID: {remote_agent.resource_name}")