import json
import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Load .env from the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))
sys.path.insert(0, _HERE)

from runtime.handler import handler  # noqa: E402
from tools.csv_tools import dataset_exists  # check stored CSV

st.set_page_config(page_title="Data Analysis Tutor", page_icon="📊", layout="centered")

# ── Session state defaults ────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.update({
        "logged_in": False,
        "username": "",
        "agent_messages": [],
        "chat_display": [],
        "last_uploaded": None,
        # track whether the user has already been informed about a
        # previously-uploaded dataset during this browser session
        "dataset_notified": False,
    })

# ── Helper: Unified Message Processor ─────────────────────────────────────────
def process_interaction(user_input: str = "", csv_content: str = "", display_text: str = ""):
    """Handles the back-and-forth with the handler and updates UI state."""
    if display_text:
        st.session_state.chat_display.append({"role": "user", "content": display_text})

    event = {
        "username": st.session_state.username,
        "inputText": user_input,
        "messages": st.session_state.agent_messages,
    }
    if csv_content:
        event["csvContent"] = csv_content

    # Logic to call the backend
    try:
        result = handler(event)
        body = json.loads(result["body"])

        if "error" in body:
            st.session_state.chat_display.append({
                "role": "assistant", 
                "content": f"⚠️ {body['error']}"
            })
        else:
            # Sync the internal agent history and the UI display
            st.session_state.agent_messages = body.get("messages", [])
            st.session_state.chat_display.append({
                "role": "assistant", 
                "content": body["response"]
            })
    except Exception as e:
        st.error(f"Communication Error: {str(e)}")

# ── Login Page ────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.title("📊 Data Analysis Tutor")
    st.markdown("A Socratic learning assistant for data discovery.")
    st.divider()

    with st.form("login"):
        username = st.text_input("Username", placeholder="Enter your username")
        if st.form_submit_button("Start Learning", use_container_width=True):
            if username.strip():
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.rerun()
            else:
                st.error("Please enter a username.")

# ── Chat Page ─────────────────────────────────────────────────────────────────
else:
    # Sidebar/Header Area
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("📊 Data Analysis Tutor")
        st.caption(f"Logged in as **{st.session_state.username}**")
    with col2:
        if st.button("Logout"):
            for key in ["logged_in", "username", "agent_messages", "chat_display", "last_uploaded"]:
                st.session_state.pop(key, None)
            st.rerun()

    # File Uploader - Logic improved to prevent "Tool Sticking"
    # Let the user know if we already have a stored dataset for them.
    if st.session_state.logged_in and dataset_exists(st.session_state.username):
        st.info("✅ You already have a dataset stored — no need to upload again unless you want to replace it.")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])
    
    # if the page has just loaded/refreshed and we already have a dataset
    # for this user, proactively tell the agent so it can remind the student
    # (only do this once per session).  We don't send any CSV content here –
    # the handler will detect the stored dataset and the agent will reply.
    if (not st.session_state.dataset_notified
            and st.session_state.logged_in
            and dataset_exists(st.session_state.username)):
        st.session_state.dataset_notified = True
        with st.spinner("Checking stored dataset…"):
            process_interaction(user_input="", display_text="")

    if uploaded is not None:
        # Check if this is a NEW file upload
        if uploaded.name != st.session_state.get("last_uploaded"):
            # Update state immediately to prevent double-processing
            st.session_state.last_uploaded = uploaded.name
            csv_text = uploaded.getvalue().decode("utf-8")
            
            with st.spinner("Analyzing dataset..."):
                process_interaction(
                    user_input="I just uploaded my dataset.",
                    csv_content=csv_text,
                    display_text=f"📁 Uploaded **{uploaded.name}**"
                )
            # No manual st.rerun() here - let the natural flow update the chat below

    st.divider()

    # Display Chat History
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about your data..."):
        with st.spinner("Thinking..."):
            process_interaction(user_input=prompt, display_text=prompt)
        st.rerun()