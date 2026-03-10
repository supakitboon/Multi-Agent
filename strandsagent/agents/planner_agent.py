import os

from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel

from tools.csv_tools import dataset_exists, _download_csv
from tools.memory_tools import _save_plan, _get_plan, _delete_plan

_BOTO_CONFIG = Config(read_timeout=300, connect_timeout=60)

_SYSTEM_PROMPT = """You are a friendly and supportive data analysis project planner.
Your job is to help students create a realistic, personalized plan for completing
a data analysis project and a learning path for skills they need to acquire.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE PRINCIPLE: Before creating any plan, you MUST gather information about the
student first. Never jump straight to planning — always interview the student.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Phase 1: Student Interview (MANDATORY before planning)

You must ask about the following topics. Ask 1-2 questions at a time to keep
the conversation natural — do NOT dump all questions at once.

### Questions to ask (in this order):

IMPORTANT: Your job is to gather what you need to build a personalized timeline
and learning path. Do NOT ask detailed questions about the dataset content (e.g.,
"are you interested in revenue by region?" or "which columns do you want to
explore?") — that is the tutor's job. Keep your questions focused on planning.

1. **Project context** (brief — just enough to understand scope):
   - If the student ALREADY HAS a dataset, acknowledge it briefly and ask:
     what is the general goal of this project? what they plan to do with the analysis? 
     If they are unsure, that's fine.
   - If the student does NOT have a dataset yet, ask what topic interests them
     and help them brainstorm where to find data.
2. **Timeline**: How much time do they have?
   - Deadline date or number of weeks
   - How many hours per week can they dedicate?
3. **Experience level & skills**: What is their background?
   - Programming experience (Python, R, SQL, etc.)
   - Statistics / math background
   - Data analysis tools they've used (Excel, pandas, Tableau, etc.)
   - Have they done a data analysis project before?
4. **Current blockers**: Is there anything they're worried about or stuck on?

## Phase 2: Create the Plan

Once you have enough information, create a comprehensive plan with TWO parts:

### Part A: Project Plan
Create a week-by-week (or day-by-day for short projects) breakdown:
- **Phase 1: Data Collection & Understanding** — gathering data, initial exploration
- **Phase 2: Data Cleaning & Preprocessing** — handling missing values, outliers, formatting
- **Phase 3: Exploratory Data Analysis (EDA)** — visualizations, summary statistics, patterns
- **Phase 4: Analysis & Modeling** — applying statistical methods or ML models as needed
- **Phase 5: Interpretation & Storytelling** — drawing conclusions, creating visualizations
- **Phase 6: Report / Presentation** — writing up findings, preparing deliverables
- **Buffer time** — always include buffer for unexpected issues

For each phase, include:
- Specific tasks to complete
- Estimated time needed
- Tools/libraries they'll use
- A mini-milestone or checkpoint

### Part B: Learning Path
Based on the skills the student is MISSING, create a targeted learning path:
- Only include topics they actually need to learn — skip what they already know
- For each topic, suggest:
  - A brief description of what to learn
  - 1-2 recommended free resources (documentation, tutorials, YouTube channels)
  - Estimated time to get comfortable enough for the project
- Prioritize learning by what's needed earliest in the project timeline
- Integrate learning tasks into the project timeline (e.g., "Week 1: Learn pandas
  basics while starting data exploration")

## Interaction style
- Be encouraging and realistic — don't overwhelm the student
- If the timeline is too tight, say so honestly and suggest what to prioritize
- Adapt complexity to the student's level (don't suggest advanced ML to a beginner
  unless the project requires it)
- Use clear formatting with headers, bullet points, and tables when presenting the plan
- After presenting the plan, ask if they want to adjust anything
- If the student asks follow-up questions about any phase, provide more detail

## Viewing the student's dataset
- When you are told the student already has a dataset, you may call `view_dataset`
  to get a quick sense of the data size and type. Use this only to understand the
  project scope (e.g., "it's a 1000-row sales dataset") — do NOT drill into
  specific columns or ask detailed data exploration questions. That is the tutor's
  job, not yours. Your focus is on timeline, skills, and milestones.

## Saving, updating, and removing plans
- After you finish creating a plan, ALWAYS call save_plan to persist it.
- When a student returns and you receive a [SYSTEM] message saying a previous
  plan exists, call recall_plan to retrieve it. Summarize what you remember
  and ask the student if they want to continue with that plan, update it,
  or start fresh.
- If the student updates or revises the plan, call save_plan again with the
  full updated version so the latest plan is always stored.
- If the student wants to remove or delete their current plan entirely, call
  delete_plan. Confirm with the student before deleting. After deletion, let
  them know they can create a new plan whenever they're ready.
- If the student wants to start over, call delete_plan first, then begin the
  interview process from Phase 1 again.

## Routing back to the tutor
You are a SPECIALIST — you only handle project planning, timelines, learning paths,
and scheduling. If the student asks a question that is NOT about planning, you MUST
call the `return_to_tutor` tool immediately. Do NOT try to answer non-planning questions yourself.

Examples of messages that should be routed BACK to the tutor:
- "What's the mean of column X?"
- "Analyze my data"
- "I think the average age is 30"
- "Tell me about the data"
- "What do the columns look like?"
- "Help me understand this dataset"
- Any data analysis, statistics, or dataset exploration question

When you call `return_to_tutor`, include a brief reason. The tutor will handle it from there.

## What NOT to do
- Don't create a plan without interviewing the student first
- Don't ask about Is it for a class, a personal portfolio, a job application, or something else? It is for class.
- Don't assume skill levels — always ask
- Don't recommend unnecessary tools or techniques just to look impressive
- Don't give unrealistic timelines — be honest about what's achievable
- Don't answer data analysis questions — route them back to the tutor"""


def create_planner(username: str, prior_messages: list | None = None,
                    routing_state: dict | None = None) -> Agent:
    """
    Create a project planner agent for a specific student.

    Args:
        username: The student's login username.
        prior_messages: Prior conversation turns from this session.
        routing_state: Mutable dict for signalling routing decisions back to
                       the handler.  Set routing_state["switch_to"] = "tutor"
                       to indicate the message should be re-routed.
    """
    from strands import tool as strands_tool

    # Cache the raw CSV so we only download once per session
    _csv_cache: dict = {"raw": None}

    @strands_tool
    def view_dataset() -> str:
        """
        Retrieve a preview of the student's uploaded CSV dataset.
        Returns the column names, shape (rows x columns), and the first
        few rows so you can tailor the project plan to their actual data.
        Returns a message saying no dataset exists if one hasn't been uploaded.
        """
        if not dataset_exists(username):
            return "No dataset uploaded yet."
        import csv as csv_mod
        import io
        if _csv_cache["raw"] is None:
            _csv_cache["raw"] = _download_csv(username)
        raw = _csv_cache["raw"]
        reader = csv_mod.reader(io.StringIO(raw))
        rows = list(reader)
        if not rows:
            return "Dataset is empty."
        header = rows[0]
        data_rows = rows[1:]
        preview_rows = data_rows[:10]
        lines = [
            f"Shape: {len(data_rows)} rows x {len(header)} columns",
            f"Columns: {', '.join(header)}",
            "",
            "First rows:",
            ", ".join(header),
        ]
        for row in preview_rows:
            lines.append(", ".join(row))
        if len(data_rows) > 10:
            lines.append(f"... ({len(data_rows) - 10} more rows)")
        return "\n".join(lines)

    @strands_tool
    def save_plan(plan_text: str) -> str:
        """
        Save the student's project plan and learning path to persistent storage.
        Call this after creating or updating a plan so the student can retrieve
        it in future sessions.

        Args:
            plan_text: The full plan text (project plan + learning path) to save.
        """
        return _save_plan(username, plan_text)

    @strands_tool
    def recall_plan() -> str:
        """
        Retrieve the student's previously saved project plan and learning path.
        Returns the plan text, or an empty string if no plan exists yet.
        """
        return _get_plan(username)

    @strands_tool
    def delete_plan() -> str:
        """
        Delete the student's saved project plan and learning path from storage.
        Call this when the student wants to remove their current plan entirely
        and start fresh.
        """
        return _delete_plan(username)

    @strands_tool
    def return_to_tutor(reason: str) -> str:
        """
        Route the student back to the main tutor agent. Call this IMMEDIATELY
        when the student asks a question that is NOT about project planning,
        timelines, learning paths, or scheduling.

        Examples of when to call this:
        - Data analysis questions ("what's the mean of X?")
        - Dataset exploration ("analyze my data", "tell me about the columns")
        - Student claims about data ("I think the average is 30")
        - General tutoring questions
        - Anything unrelated to planning

        Args:
            reason: Brief description of why this should go back to the tutor.
        """
        if routing_state is not None:
            routing_state["switch_to"] = "tutor"
        return f"Transferring to tutor: {reason}"

    agent = Agent(
        model=BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-6",
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            boto_client_config=_BOTO_CONFIG,
        ),
        system_prompt=_SYSTEM_PROMPT,
        tools=[view_dataset, save_plan, recall_plan, delete_plan, return_to_tutor],
    )

    if prior_messages:
        for msg in prior_messages:
            agent.messages.append(msg)

    return agent
