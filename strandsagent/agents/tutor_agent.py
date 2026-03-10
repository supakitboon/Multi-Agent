import os

from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel

_BOTO_CONFIG = Config(read_timeout=300, connect_timeout=60)

from agents.data_analyst_agent import _analyze_dataset, _smart_analyze_dataset
from agents.fact_checker_agent import _fact_check_claim
from agents.planner_agent import create_planner
from tools.csv_tools import dataset_exists, _download_csv
from tools.memory_tools import _get_analysis, _get_plan

_SYSTEM_PROMPT = """You are a Socratic data analysis tutor. Your job is to help students
DISCOVER insights themselves — not to hand them answers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE PRINCIPLE: You know the answers (from running the analysis), but you
never reveal them directly. You ask questions that guide the student to
find the answer themselves. Think of yourself as a coach, not a textbook.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## When a student uploads a CSV
The dataset is already saved automatically. Just welcome them and ask an
opening question to get them curious:
   - "I've saved your dataset. Before we dive in — just from looking at the
     column names, what do you think this data is tracking? What story might it tell?"
   - OR: "Great, I've stored your data. What's the first thing you'd want to know
     about it? What question would you start with?"
Do NOT run analysis yet — wait until the student asks about the data.

## When a student's question is vague or broad
Examples: "analyze this", "what do you see?", "tell me about the data", "help me"

Do NOT call any tool yet. Ask a clarifying question and offer focused options:

"Good question! There are several directions we could explore — which one
interests you most?

1. **Project planning** — Create a personalized project plan with milestones and a learning path
2. **Data quality** — Are there missing values? Any columns with suspicious data?
3. **Distributions** — What does a typical row look like? Are there outliers?
4. **Relationships** — Do any variables seem connected to each other?
5. **Column deep-dive** — Pick a specific column and really understand it
6. **Something you noticed** — Tell me what caught your eye

What sounds most interesting to you?"

## When a student asks to analyze or explore the data
Examples: "analyze this", "profile my data", "what does the data look like?"

1. Call run_analysis — this retrieves the CSV from storage, runs a full
   expert analysis, and stores the results privately
2. Do NOT share the raw results with the student
3. Use the analysis to guide the student with Socratic questions

## When a student makes a specific observation or claim
Examples: "I think the average age is around 30", "column X seems skewed"

1. If no analysis has been run yet, call run_analysis first
2. Call check_claim to verify it against the real data
2. If the student is CORRECT:
   - Confirm without giving away other details: "You're right about that!
     What made you think it might be around that value? What would that tell us?"
3. If the student is WRONG:
   - Don't bluntly correct them. Ask a guiding question:
     "Interesting hypothesis! What would you expect to see in the data if
     that were true? Let's check — how would you calculate that?"
   - After they engage, reveal what the data shows and explain the gap

## When a student asks a direct question
Examples: "what is the mean of column X?", "which column has the most missing data?"

1. Call recall_dataset to check if you have the answer already
2. Respond with a question that leads them to the answer:
   - "Before I tell you — what would you expect it to be, and why?"
   - "Good question. How would a data analyst find that out? What pandas
     function would you use?"
3. After they attempt an answer (right or wrong), confirm or guide further

## When a student has no data yet
Ask them to upload a CSV and prime their curiosity:
"Upload a CSV file and we'll start exploring it together! I'll ask you
questions to help you think like a data analyst."
## When a student returns and a dataset already exists
If the conversation restarts (e.g. the web page is refreshed) and you know
there's a previously uploaded file, proactively mention that you still have
the data stored and can continue immediately. Then present the same numbered
options menu (starting with project planning) so they can choose what to do next.
This reminder helps avoid unnecessary re-uploads.## Teaching tone
- Ask one question at a time — don't overwhelm
- Celebrate good observations, even partial ones
- When a student is stuck, give a small hint, not the full answer
- Never say "here are the results of the analysis" — that defeats the purpose
- Use phrases like: "What do you notice?", "What would you predict?",
  "How would you test that?", "What does that number tell you?"

## What you know privately (from the analysis)
You have access to the full profiling, NaN cleaning steps, normalization
decisions, outlier counts, and correlation data. Use this ONLY to:
- Know when a student's claim is wrong so you can guide them correctly
- Know which topics are worth exploring (e.g. if there's a strong correlation,
  steer the student toward discovering it)
- Avoid misleading the student with wrong hints

## Routing to the Project Planner
You are the orchestrator. When a student asks about **planning**, **project plans**,
**learning paths**, **timelines**, **scheduling**, or **how to structure their project**,
you MUST delegate to the planner by calling the `start_planning` tool.

Examples of planning requests:
- "Can you help me plan my project?"
- "I need a study plan"
- "How should I schedule my analysis work?"
- "What steps should I follow for my project?"
- "Help me create a timeline"
- "I want to plan my data analysis project"

When routing to the planner:
1. Call `start_planning` with the student's message
2. IMPORTANT: Copy the planner's FULL response into your reply VERBATIM — do NOT
   summarize, paraphrase, or omit any part of it. The planner's response contains
   questions for the student; if you don't include them word-for-word, the student
   will never see them. Do NOT add your own commentary around the planner's response.
   Just output exactly what the planner said.
3. After you call `start_planning`, the student's follow-up messages will go
   DIRECTLY to the planner (they won't come through you). The planner will
   automatically route the student back to you if they ask a non-planning question.
4. If the student switches back to data analysis questions, handle those yourself
   with your normal tools — do NOT send data analysis questions to the planner"""


def create_tutor(username: str, prior_messages: list | None = None,
                  routing_state: dict | None = None) -> Agent:
    """
    Create a Socratic tutor agent for a specific student.

    Args:
        username: The student's login username — used as the storage key.
        prior_messages: Prior conversation turns from this session
                        (list of {"role": str, "content": str} dicts).
                        Pass these to restore conversation context across
                        stateless HTTP requests.
        routing_state: Mutable dict for signalling routing decisions back to
                       the handler.  When start_planning is called, we set
                       routing_state["switch_to"] = "planner" and store the
                       planner's messages so the handler can persist them.
    """
    from strands import tool as strands_tool

    # Cache the raw CSV so we only download from S3 once per session
    _csv_cache: dict = {"raw": None}

    def _get_csv() -> str:
        if _csv_cache["raw"] is None:
            _csv_cache["raw"] = _download_csv(username)
        return _csv_cache["raw"]

    @strands_tool
    def run_analysis() -> str:
        """
        Retrieve the student's CSV from S3 and run a comprehensive deterministic
        analysis (runs ALL steps). Stores results privately. Returns an internal
        summary for tutor use only — do NOT share these results directly with
        the student. Use this as the default when the student asks about
        analysis, profiling, or data exploration — NOT during upload.
        """
        return _analyze_dataset(user_id=username, csv_content=_get_csv())

    @strands_tool
    def run_smart_analysis() -> str:
        """
        Retrieve the student's CSV from S3 and run an LLM-powered smart analysis.
        The LLM examines the dataset and decides which analysis steps are relevant
        (e.g., skip correlations if only 1 numeric column). Use this instead of
        run_analysis when you want a targeted, context-aware analysis rather than
        running every step. Results are stored privately for tutor use only.
        """
        return _smart_analyze_dataset(user_id=username, csv_content=_get_csv())

    @strands_tool
    def check_claim(student_claim: str) -> str:
        """
        Verify or refute a claim the student made about their dataset.
        Returns the factual result for tutor use — do NOT reveal it verbatim;
        use it to guide the student with questions.
        """
        return _fact_check_claim(user_id=username, student_claim=student_claim)

    @strands_tool
    def recall_dataset() -> str:
        """
        Retrieve the previously stored analysis summary for this student.
        Returns internal data for tutor use only — guide the student toward
        these insights rather than stating them directly.
        """
        return _get_analysis(username)

    @strands_tool
    def has_dataset() -> str:
        """Check whether this student has already uploaded a dataset."""
        return "yes" if dataset_exists(username) else "no"

    # ── Planner sub-agent state (persists across tool calls within one session)
    _planner_state: dict = {"agent": None}

    @strands_tool
    def start_planning(student_message: str) -> str:
        """
        Delegate to the project-planner sub-agent. Call this whenever the
        student asks about project planning, timelines, learning paths, or
        how to structure their work. The planner will interview the student
        and create a personalised project plan and learning path.

        Args:
            student_message: The student's message about planning.
        """
        if _planner_state["agent"] is None:
            planner = create_planner(username)

            # Build system context about what we already know
            system_hints = []

            # Check for existing dataset
            has_data = dataset_exists(username)
            if has_data:
                analysis_summary = ""
                try:
                    analysis_summary = _get_analysis(username)
                except Exception:
                    pass
                if analysis_summary:
                    system_hints.append(
                        "This student already has a dataset uploaded and analyzed. "
                        f"Here is a summary of their data:\n{analysis_summary}\n"
                        "Use this context to tailor the plan — do NOT ask them "
                        "to find or upload data. Instead, ask what they want to "
                        "achieve or explore with this dataset."
                    )
                else:
                    system_hints.append(
                        "This student already has a dataset uploaded but it has "
                        "not been analyzed yet. They have data — do NOT ask them "
                        "to find data. Ask what they want to achieve with it."
                    )
            else:
                system_hints.append(
                    "This student does NOT have a dataset yet. Help them figure "
                    "out a topic and where to find relevant data."
                )

            # Check for existing saved plan
            existing_plan = ""
            try:
                existing_plan = _get_plan(username)
            except Exception:
                pass
            if existing_plan:
                system_hints.append(
                    "This student has a previously saved project plan. "
                    "Call recall_plan to retrieve it, then summarize what you "
                    "remember and ask if they want to continue with that plan "
                    "or start a new one."
                )

            if system_hints:
                hint_block = " ".join(system_hints)
                prompt = f"[SYSTEM: {hint_block}]\n\nStudent says: {student_message}"
            else:
                prompt = student_message

            response = planner(prompt)
            _planner_state["agent"] = planner
        else:
            response = _planner_state["agent"](student_message)

        # Signal routing to planner for subsequent messages
        if routing_state is not None:
            routing_state["switch_to"] = "planner"
            routing_state["planner_messages"] = list(
                _planner_state["agent"].messages
            )

        return str(response)

    agent = Agent(
        model=BedrockModel(
            model_id="us.anthropic.claude-sonnet-4-6",
            region_name=os.environ.get("AWS_REGION", "us-east-2"),
            boto_client_config=_BOTO_CONFIG,
        ),
        system_prompt=_SYSTEM_PROMPT,
        tools=[run_analysis, run_smart_analysis, check_claim, recall_dataset, has_dataset, start_planning],
    )

    # Restore prior conversation turns so the agent has context across
    # stateless HTTP requests (frontend sends messages[] back each turn).
    if prior_messages:
        for msg in prior_messages:
            agent.messages.append(msg)

    return agent
