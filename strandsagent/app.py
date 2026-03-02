import json
import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Load .env from the same directory as this file, regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))

sys.path.insert(0, _HERE)

from runtime.handler import handler  # noqa: E402

st.set_page_config(page_title="Data Analysis Tutor", page_icon="📊", layout="centered")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "logged_in": False,
    "username": "",
    "agent_messages": [],
    "chat_display": [],
    "last_uploaded": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper ────────────────────────────────────────────────────────────────────
def send_message(user_input: str = "", csv_content: str = "", display_text: str = ""):
    if display_text:
        st.session_state.chat_display.append({"role": "user", "content": display_text})

    event = {
        "username": st.session_state.username,
        "inputText": user_input,
        "messages": st.session_state.agent_messages,
    }
    if csv_content:
        event["csvContent"] = csv_content

    result = handler(event)
    body = json.loads(result["body"])

    if "error" in body:
        st.session_state.chat_display.append({
            "role": "assistant",
            "content": f"⚠️ {body['error']}",
        })
    else:
        st.session_state.agent_messages = body.get("messages", [])
        st.session_state.chat_display.append({
            "role": "assistant",
            "content": body["response"],
        })


# ── Login page ────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.title("📊 Data Analysis Tutor")
    st.markdown("A Socratic learning assistant that helps you discover insights in your data — through questions, not answers.")
    st.divider()

    with st.form("login"):
        username = st.text_input("Username", placeholder="Enter your username to begin")
        if st.form_submit_button("Start Learning", use_container_width=True):
            if username.strip():
                st.session_state.logged_in = True
                st.session_state.username = username.strip()
                st.rerun()
            else:
                st.error("Please enter a username.")

# ── Chat page ─────────────────────────────────────────────────────────────────
else:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("📊 Data Analysis Tutor")
        st.caption(f"Logged in as **{st.session_state.username}**")
    with col2:
        st.write("")
        st.write("")
        if st.button("Logout", use_container_width=True):
            for key in ["logged_in", "username", "agent_messages", "chat_display", "last_uploaded"]:
                del st.session_state[key]
            st.rerun()

    uploaded = st.file_uploader("Upload a CSV dataset to explore", type=["csv"])
    if uploaded and uploaded.name != st.session_state.last_uploaded:
        st.session_state.last_uploaded = uploaded.name
        csv_text = uploaded.read().decode("utf-8")
        with st.spinner("Analyzing your dataset..."):
            send_message(
                user_input="I just uploaded my dataset.",
                csv_content=csv_text,
                display_text=f"📁 Uploaded **{uploaded.name}**",
            )
        st.rerun()

    st.divider()

    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your data..."):
        with st.spinner("Thinking..."):
            send_message(user_input=prompt, display_text=prompt)
        st.rerun()
