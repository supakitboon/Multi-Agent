import vertexai
from vertexai import agent_engines
from dotenv import load_dotenv
import os
import asyncio

# 1. Load environment
load_dotenv()
vertexai.init(
    project=os.environ["GOOGLE_CLOUD_PROJECT"],
    location=os.environ["GOOGLE_CLOUD_LOCATION"],
)

# 2. Define the async main function
async def main():
    # Get the most recently deployed agent
    # (We do this inside the function so we can exit easily if none are found)
    agents_list = list(agent_engines.list())
    
    if not agents_list:
        print("No agents found. Please deploy first.")
        return

    remote_agent = agents_list[0]  # Get the first (most recent) agent
    print(f"Connected to deployed agent: {remote_agent.resource_name}")

    print("Thinking...")
    # 3. Now 'async for' works because it is inside 'async def'
    try:
        async for item in remote_agent.async_stream_query(
            message="What is the weather in Tokyo?",
            user_id="user_42",
        ):
            print(item)
    except AttributeError:
        # Fallback: Some versions of the SDK use 'stream_query' (sync) or just 'query'
        print("\n 'async_stream_query' not found. Trying standard 'query'...")
        response = remote_agent.query(message="What is the weather in Tokyo?")
        print(response)

# 4. Run the async loop
if __name__ == "__main__":
    asyncio.run(main())