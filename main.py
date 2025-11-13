# Import Library 
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.adk.tools import google_search
import asyncio 
import os
from dotenv import load_dotenv
### adk run // adk web // adk api_server 
#Load enviroment variables 
load_dotenv()
retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


root_agent = Agent(
    name = "personalized_assistant",
    model = Gemini(model = 'gemini-2.5-flash-lite',
                   retry_options = retry_config,
                   api_key  = os.getenv("GOOGLE_API_KEY")),
    description = 'A simple agent that can answer general question',
    instruction = " You are a helpful assistant. Use Google Search for current info or if unsure",
    tools =[google_search]

)

print("Created Ochestration Agent")
runner = InMemoryRunner(agent = root_agent)
print("Created Runner") # act as orchestrator
### It manages the conversation, sends our message to the agent and handles its reponse

async def main():
    response = await runner.run_debug(
    "What is Agent Development Kit from Google? What languages is the SDK available in?"
    )
 
if __name__ == "__main__":
    asyncio.run(main())


