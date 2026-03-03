Directory structure:
└── supakitboon-multi-agent/
    ├── README.md
    └── strandsagent/
        ├── app.py
        ├── requirements.txt
        ├── test.py
        ├── .env.example
        ├── agents/
        │   ├── __init__.py
        │   ├── data_analyst_agent.py
        │   ├── fact_checker_agent.py
        │   └── tutor_agent.py
        ├── runtime/
        │   ├── __init__.py
        │   └── handler.py
        └── tools/
            ├── __init__.py
            ├── code_interpreter.py
            ├── csv_tools.py
            └── memory_tools.py
