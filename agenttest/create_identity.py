import vertexai
from vertexai import agent_engines
from vertexai import types

client = vertexai.Client( 
  project="adk-2025-459915", 
  location="us-central1"
)
remote_app = client.agent_engines.create( 
  config={"identity_type": types.IdentityType.AGENT_IDENTITY}
)
