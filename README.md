# 📊 Strands Agent — AI-Powered Data Tutoring System

An intelligent tutoring system that teaches students data analysis through Socratic dialogue. Powered by AWS Bedrock (Claude Sonnet 4.6), the agent guides students to discover insights from their own datasets rather than simply giving answers.

---

## 🧠 How It Works

Students upload a CSV file and chat with a Tutor Agent that:
- Analyzes their dataset automatically
- Asks Socratic questions to guide their thinking
- Fact-checks claims students make about their data
- Remembers previous sessions so learning can continue across logins

---

## 🏗️ Architecture Overview

The system is composed of four layers:

### 1. Frontend (Streamlit)
- `app.py` — handles login/session management, CSV upload, and the chat interface

### 2. Runtime Layer
- `handler.py` — parses incoming CSV (raw, base64, or multipart), restores conversation history, and routes requests to the Tutor Agent

### 3. Agent Layer (Strands Framework)
| Agent | Role | Model |
|---|---|---|
| **Tutor Agent** | Orchestrator — drives Socratic dialogue | Claude Sonnet 4.6 |
| **Data Analyst Agent** | Profiles and preprocesses CSV data locally via Pandas | — |
| **Fact Checker Agent** | Verifies student claims against actual data | Claude Sonnet 4.6 |

### 4. Tool Layer
| Tool | Responsibility |
|---|---|
| `csv_tools.py` | Upload / download / check existence of CSV in S3 |
| `memory_tools.py` | Save and retrieve dataset analysis summaries |
| `code_interpreter.py` | Spawn sandboxed Python execution sessions |
| `preprocessing_tools.py` | 8 data processing utilities (clean, profile, correlate, etc.) |

---

## ☁️ AWS Services

| Service | Usage |
|---|---|
| **AWS Bedrock** | LLM inference using `claude-sonnet-4-6` |
| **AWS S3** | Stores raw CSV files at `datasets/{user_id}/dataset.csv` (max 10MB) |
| **AWS AgentCore Memory** | Persists dataset analysis summaries per user (namespace: `{username}`) |
| **AWS AgentCore Code Interpreter** | Sandboxed environment for executing Pandas/NumPy/SciPy code |

---

## 🗺️ System Architecture Diagram

```mermaid
flowchart TB
    classDef frontend fill:#4A90D9,stroke:#2C5F8A,color:#fff,rx:8
    classDef runtime fill:#7B68EE,stroke:#4B3DB5,color:#fff
    classDef agent fill:#E8A838,stroke:#B87D1A,color:#fff
    classDef determin fill:#27AE60,stroke:#1A7A42,color:#fff
    classDef smart fill:#E74C3C,stroke:#A93226,color:#fff
    classDef tool fill:#5DADE2,stroke:#2E86C1,color:#fff
    classDef pandas fill:#A569BD,stroke:#7D3C98,color:#fff
    classDef aws fill:#FF9900,stroke:#CC7A00,color:#fff

    subgraph UI["🖥️  Frontend — Streamlit"]
        APP["app.py\nLogin · Session · CSV Upload · Chat"]
    end

    subgraph RUNTIME["⚙️  Runtime Layer"]
        HANDLER["handler.py\nParse CSV · Restore History · Route"]
    end

    subgraph AGENTS["🤖  Agent Layer — Strands Framework"]
        TUTOR["🎓 Tutor Agent\nOrchestrator · Claude Sonnet 4.6"]

        subgraph DATA_ANALYST["📊  Data Analyst"]
            DETERM["🔵 Deterministic Path\nPure Pandas · No LLM\nRuns ALL 6 steps"]
            SMART["🔴 Smart Path\nClaude Sonnet 4.6\nLLM picks steps"]
        end

        FACT["🔍 Fact Checker Agent\nClaude Sonnet 4.6"]
    end

    subgraph PANDAS_OPS["🐼  Pandas Analysis Steps"]
        direction LR
        P1["profile()"] --> P2["remove_duplicates()"] --> P3["clean_missing()"]
        P4["detect_outliers()"] --> P5["compute_correlations()"] --> P6["normalize()"]
    end

    subgraph TOOLS["🛠️  Tool Layer"]
        CSV_TOOLS["csv_tools.py\nupload · download · exists"]
        MEM_TOOLS["memory_tools.py\nsave · get analysis"]
        CODE_INTERP["code_interpreter.py\nCodeInterpreterSession"]
        PREPROC["preprocessing_tools.py\n8 processing tools"]
    end

    subgraph AWS["☁️  AWS Cloud Services"]
        BEDROCK["⚡ AWS Bedrock\nClaude Sonnet 4.6"]
        S3["🪣 AWS S3\ndatasets/{user_id}/dataset.csv"]
        AGENTCORE_MEM["🧠 AgentCore\nMemory Service"]
        AGENTCORE_CODE["💻 AgentCore\nCode Interpreter Sandbox"]
    end

    %% Main flow
    APP -->|"process_interaction()"| HANDLER
    HANDLER -->|"create_tutor()"| TUTOR

    %% Tutor delegates
    TUTOR -->|"run_analysis()"| DETERM
    TUTOR -->|"run_smart_analysis()"| SMART
    TUTOR -->|"check_claim()"| FACT
    TUTOR -->|"recall_dataset()"| MEM_TOOLS
    TUTOR -->|"has_dataset()"| CSV_TOOLS
    TUTOR -->|"LLM calls"| BEDROCK

    %% Deterministic path
    DETERM -->|"all 6 steps"| PANDAS_OPS
    DETERM -->|"save results"| MEM_TOOLS
    DETERM -->|"preprocess"| PREPROC

    %% Smart path
    SMART -->|"decide steps"| BEDROCK
    SMART -.->|"selective steps"| PANDAS_OPS
    SMART -->|"save results"| MEM_TOOLS
    SMART -->|"preprocess"| PREPROC

    %% Fact checker
    FACT -->|"fetch analysis"| MEM_TOOLS
    FACT -->|"fetch CSV"| CSV_TOOLS
    FACT -->|"verify claim"| BEDROCK

    %% Tools to AWS
    CSV_TOOLS -->|"put/get/head"| S3
    MEM_TOOLS -->|"create_event · retrieve"| AGENTCORE_MEM
    PREPROC -->|"execute python"| CODE_INTERP
    CODE_INTERP -->|"spawn sandbox"| AGENTCORE_CODE
    HANDLER -->|"upload CSV"| CSV_TOOLS

    %% Styles
    class APP frontend
    class HANDLER runtime
    class TUTOR,FACT agent
    class DETERM determin
    class SMART smart
    class CSV_TOOLS,MEM_TOOLS,CODE_INTERP,PREPROC tool
    class P1,P2,P3,P4,P5,P6 pandas
    class BEDROCK,S3,AGENTCORE_MEM,AGENTCORE_CODE aws
```

---

## 🔄 Key Flows

### CSV Upload & Analysis
1. Student uploads CSV via Streamlit
2. Handler uploads file to S3
3. Tutor Agent triggers Data Analyst
4. Data Analyst runs profiling and preprocessing via Code Interpreter sandbox
5. Analysis summary is saved to AgentCore Memory
6. Tutor generates a Socratic opening question

### Fact Checking
1. Student makes a claim (e.g. *"The average age is 35"*)
2. Tutor delegates to Fact Checker Agent
3. Fact Checker retrieves stored analysis from AgentCore Memory + raw CSV from S3
4. Claude verifies the claim and returns `CORRECT` / `WRONG` / `AMBIGUOUS` with reasoning
5. Tutor wraps the verdict in a Socratic follow-up response

### Returning User Session Restore
1. Student logs in without uploading a new CSV
2. Tutor checks S3 for an existing dataset
3. Tutor retrieves previous analysis from AgentCore Memory
4. Dialogue resumes from where it left off

---

## 📁 Project Structure

```
├── app.py                  # Streamlit frontend
├── handler.py              # Runtime routing layer
├── agents/
│   ├── tutor.py            # Tutor Agent (orchestrator)
│   ├── data_analyst.py     # Data Analyst Agent
│   └── fact_checker.py     # Fact Checker Agent
├── tools/
│   ├── csv_tools.py        # S3 CSV operations
│   ├── memory_tools.py     # AgentCore Memory operations
│   ├── code_interpreter.py # AgentCore Code Interpreter
│   └── preprocessing_tools.py  # 8 data processing tools
└── README.md
```

---

## ⚙️ Prerequisites

- Python 3.10+
- AWS account with access to:
  - AWS Bedrock (Claude Sonnet 4.6)
  - AWS S3
  - AWS AgentCore (Memory + Code Interpreter)
- Strands Agent framework installed

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-org/strands-agent.git
cd strands-agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` with your AWS credentials and config:
```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name
AGENTCORE_MEMORY_NAMESPACE=your_namespace
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🔐 Security Notes

- Each user's data is isolated in S3 under `datasets/{user_id}/` and in AgentCore Memory under their own namespace
- CSV files are capped at **10MB**
- Code execution runs in an **isolated AgentCore sandbox** — no access to the host environment

