from google.adk.agents import Agent
from dotenv import load_dotenv
import vertexai
import os


load_dotenv()
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

# --- Sub-Agent 1: Weather Specialist ---
def get_weather(city: str) -> str:
    """Fetches weather for a specific city."""
    # Mock data for demo
    return f"The weather in {city} is sunny and 25°C."

weather_agent = Agent(
    name="weather_specialist",
    model="gemini-1.5-flash",
    instruction="You are a meteorologist. Give short, concise weather updates.",
    tools=[get_weather]
)

# --- Sub-Agent 2: Math Specialist ---
def calculate_trip_cost(distance: float, mpg: float = 25.0, gas_price: float = 3.50) -> float:
    """Calculates trip cost based on distance."""
    return (distance / mpg) * gas_price

math_agent = Agent(
    name="math_specialist",
    model="gemini-1.5-flash",
    instruction="You are a calculator. specificy in trip costs.",
    tools=[calculate_trip_cost]
)

# --- The "Bridge" Functions ---
# These allow the Root Agent to "call" the sub-agents.
# We wrap the agent query in a simple function.

def ask_weather_agent(question: str) -> str:
    """Use this tool to ask the weather specialist a question."""
    # The root agent calls the sub-agent here
    response = weather_agent.query(message=question)
    return response.answer

def ask_math_agent(question: str) -> str:
    """Use this tool to perform calculations for travel costs."""
    response = math_agent.query(message=question)
    return response.answer

# --- Root Agent (The Orchestrator) ---
root_agent = Agent(
    name="travel_orchestrator",
    model="gemini-1.5-pro", # Smarter model for orchestration
    instruction="""
    You are a Travel Assistant. 
    - If the user asks about weather, delegate to the 'ask_weather_agent' tool.
    - If the user asks about costs, delegate to the 'ask_math_agent' tool.
    - Combine their answers to give a helpful travel recommendation.
    """,
    # The Root Agent sees the other agents as "Tools"
    tools=[ask_weather_agent, ask_math_agent] 
)