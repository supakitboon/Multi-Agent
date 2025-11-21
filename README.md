# Multi-Agent

## Prerequisites
- Access: You must be an Owner or have Vertex AI User permissions on the Google Cloud Project.
- Project ID: Ensure you have the Target Project ID ready (e.g., adk-2025-xxxxxx).

## System Tools & Authentication
These steps only need to be done once per machine.
1. Install Google Cloud SDK
We use the gcloud CLI for authentication.
```
brew install --cask google-cloud-sdk
```
2. Authenticate
We need to log in twice: once for the CLI tools, and once for the Python code (Application Default Credentials).
```
# 1. Login to the CLI tool
gcloud auth login
# 2. Login for Python Libraries 
gcloud auth application-default login
```
3. Configure Project
Set your default project and enable the required services (Vertex AI and Cloud Storage).

```
# Replace {PROJECT_ID} with your actual project ID
gcloud config set project {PROJECT_ID}

# Enable required APIs
gcloud services enable aiplatform.googleapis.com storage.googleapis.com
```

## Python Environment Setup
These steps should be done inside this project folder.
1. Create Virtual Environment
We use a virtual environment (.venv) to isolate dependencies and avoid conflicts with the system Python.
Vertex AI Agent Engine currently supports Python 3.9 - 3.13. Make sute that you use python in these version.
```
# Install python by Brew first  
brew install python@3.13
# Create the virtual environment using Homebrew's Python 3
/opt/homebrew/opt/python@3.13/bin/python3.13 -m venv .venv
```
2. Activate Environment
You must run this command every time you open a new terminal window to work on this project.
```
source .venv/bin/activate
```
Note : You should see (.venv) appear at the start of your command prompt

3. Install Dependencies
Once the environment is active, install the Vertex AI SDK with the required Agent Engine extras.
```
# Ensure pip is up to date
pip install --upgrade pip

# Install Vertex AI SDK (with extras) and ADK
pip install --upgrade "google-cloud-aiplatform[agent_engines,adk]" google-adk
```

## Create Agent
```
agenttest/
├── agent.py                  # The logic
├── requirements.txt          # The libraries
├── .env                      # The secrets/config
└── .agent_engine_config.json # The hardware specs
```
## command to create vertex engine 
```
adk deploy agent_engine --project=$PROJECT_ID --region=us-east4 agenttest --agent_engine_config_file=agenttest/.agent_engine_config.json

gcloud config set compute/region us-east4
```