import json
import os
import sys
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

# Load .env from the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))
sys.path.insert(0, _HERE)

from runtime.handler import handler  # noqa: E402
from tools.csv_tools import dataset_exists  # check stored CSV
from tools.chat_storage import save_chat, load_chat, list_chats  # noqa: E402

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
        # ── Chat history ──
        "current_chat_id": None,
        "chat_created_at": None,
        "chat_list": [],
        "chat_list_loaded": False,
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
        return

    # Auto-save after every successful response
    _auto_save()


# ── Chat History Helpers ──────────────────────────────────────────────────────
def _generate_chat_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _derive_title(chat_display):
    """Extract title from first user message."""
    for msg in chat_display:
        if msg["role"] == "user" and msg["content"].strip():
            text = msg["content"].strip()
            if text.startswith("\U0001f4c1 Uploaded"):
                continue
            return text[:60] + ("..." if len(text) > 60 else "")
    return "New conversation"


def _auto_save():
    """Persist the current chat to S3 if there is content."""
    ss = st.session_state
    if not ss.get("current_chat_id") or not ss.get("chat_display"):
        return
    try:
        save_chat(
            username=ss.username,
            chat_id=ss.current_chat_id,
            title=_derive_title(ss.chat_display),
            agent_messages=ss.agent_messages,
            chat_display=ss.chat_display,
            created_at=ss.chat_created_at,
        )
        ss.chat_list_loaded = False  # refresh sidebar on next rerun
    except Exception as e:
        print(f"[chat_storage] Auto-save failed: {e}", flush=True)


def _start_new_chat():
    """Reset state for a brand new conversation."""
    ss = st.session_state
    ss.current_chat_id = _generate_chat_id()
    ss.chat_created_at = datetime.now(timezone.utc).isoformat()
    ss.agent_messages = []
    ss.chat_display = []
    ss.last_uploaded = None
    ss.dataset_notified = False
    ss.chat_list_loaded = False


def _load_existing_chat(chat_id):
    """Load a chat from S3 into session state."""
    ss = st.session_state
    data = load_chat(ss.username, chat_id)
    ss.current_chat_id = data["chat_id"]
    ss.chat_created_at = data.get("created_at")
    ss.agent_messages = data.get("agent_messages", [])
    ss.chat_display = data.get("chat_display", [])
    ss.last_uploaded = None
    ss.dataset_notified = True  # don't re-notify about dataset


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
    # Ensure a chat session exists
    if not st.session_state.current_chat_id:
        _start_new_chat()

    # ── Sidebar: Chat History ──
    with st.sidebar:
        st.title("Chat History")

        if st.button("+ New Chat", use_container_width=True, type="primary"):
            _auto_save()
            _start_new_chat()
            st.rerun()

        st.divider()

        # Load chat list from S3 (once per session or after save)
        if not st.session_state.chat_list_loaded:
            st.session_state.chat_list = list_chats(st.session_state.username)
            st.session_state.chat_list_loaded = True

        for chat_meta in st.session_state.chat_list:
            cid = chat_meta["chat_id"]
            label = chat_meta["title"]
            is_active = cid == st.session_state.current_chat_id
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"chat_{cid}", use_container_width=True, type=btn_type):
                if not is_active:
                    _auto_save()
                    _load_existing_chat(cid)
                    st.rerun()

    # ── Main Area ──
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("📊 Data Analysis Tutor")
        st.caption(f"Logged in as **{st.session_state.username}**")
    with col2:
        if st.button("Logout"):
            _auto_save()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
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

    # ── Pixel cat walking animation ──
    # Uses st.components.v1.html so JavaScript actually runs (st.markdown strips scripts)
    import streamlit.components.v1 as components
    components.html("""
    <canvas id="pixelCat" width="32" height="32" style="
        position:fixed; bottom:6px; left:-64px; z-index:9999;
        pointer-events:none; image-rendering:pixelated;
    "></canvas>
    <script>
    (function() {
        const canvas = document.getElementById('pixelCat');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const S = 2;
        canvas.width = 16 * S;
        canvas.height = 16 * S;
        canvas.style.width = (16 * S * 2) + 'px';
        canvas.style.height = (16 * S * 2) + 'px';

        // 0=transparent, 1=#222(outline), 2=#6090d0(blue body), 3=#90b8e8(light blue),
        // 4=#fff(white eyes), 5=#e04040(red eye), 6=#aaa(silver/gray), 7=#ffcc00(antenna)
        const P = ['transparent','#222','#6090d0','#90b8e8','#fff','#e04040','#aaa','#ffcc00'];

        const F = [
          // Frame 0: robot walk A (left leg forward)
          [[0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,1,2,2,2,2,2,1,0,0,0,0,0],[0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0],[0,0,0,1,2,4,5,2,4,5,2,1,0,0,0,0],[0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0],[0,0,0,1,2,2,6,6,6,2,2,1,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,1,6,1,2,2,2,1,6,1,0,0,0,0],[0,0,0,0,0,1,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,0,1,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,0,1,1,2,1,1,0,0,0,0,0,0],[0,0,0,0,1,2,1,0,0,1,2,1,0,0,0,0],[0,0,0,1,2,1,0,0,0,0,1,2,1,0,0,0],[0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0]],
          // Frame 1: robot walk B (legs together)
          [[0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,1,2,2,2,2,2,1,0,0,0,0,0],[0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0],[0,0,0,1,2,4,5,2,4,5,2,1,0,0,0,0],[0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0],[0,0,0,1,2,2,6,6,6,2,2,1,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,1,6,1,2,2,2,1,6,1,0,0,0,0],[0,0,0,0,0,1,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,0,1,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,0,1,1,2,1,1,0,0,0,0,0,0],[0,0,0,0,0,1,2,0,1,2,1,0,0,0,0,0],[0,0,0,0,0,1,2,0,1,2,1,0,0,0,0,0],[0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0]],
          // Frame 2: robot walk C (right leg forward)
          [[0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,1,2,2,2,2,2,1,0,0,0,0,0],[0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0],[0,0,0,1,2,4,5,2,4,5,2,1,0,0,0,0],[0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0],[0,0,0,1,2,2,6,6,6,2,2,1,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,1,6,1,2,2,2,1,6,1,0,0,0,0],[0,0,0,0,0,1,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,0,1,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,0,1,1,2,1,1,0,0,0,0,0,0],[0,0,0,0,1,2,1,0,0,1,2,1,0,0,0,0],[0,0,0,0,0,1,2,1,1,2,1,0,0,0,0,0],[0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0]],
          // Frame 3: robot walk D (legs together, opposite bob)
          [[0,0,0,0,0,0,7,7,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0],[0,0,0,0,1,2,2,2,2,2,1,0,0,0,0,0],[0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0],[0,0,0,1,2,4,5,2,4,5,2,1,0,0,0,0],[0,0,0,1,2,2,2,2,2,2,2,1,0,0,0,0],[0,0,0,1,2,2,6,6,6,2,2,1,0,0,0,0],[0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],[0,0,0,1,6,1,2,2,2,1,6,1,0,0,0,0],[0,0,0,0,0,1,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,0,1,2,2,2,1,0,0,0,0,0,0],[0,0,0,0,0,1,1,2,1,1,0,0,0,0,0,0],[0,0,0,0,0,1,2,0,1,2,1,0,0,0,0,0],[0,0,0,0,0,1,2,0,1,2,1,0,0,0,0,0],[0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
        ];

        let frame = 0, x = -64, dir = 1, tick = 0;
        const speed = 1.2;

        function draw(f) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const d = F[f];
            for (let y = 0; y < 16; y++)
                for (let px = 0; px < 16; px++) {
                    const c = d[y][dir === 1 ? px : 15 - px];
                    if (c === 0) continue;
                    ctx.fillStyle = P[c];
                    ctx.fillRect(px * S, y * S, S, S);
                }
        }

        // Move canvas to top-level document body so it escapes the iframe
        const topDoc = window.parent.document;
        const el = topDoc.getElementById('pixel-cat-live');
        if (el) el.remove();
        const wrap = topDoc.createElement('div');
        wrap.id = 'pixel-cat-live';
        wrap.style.cssText = 'position:fixed;bottom:0;left:0;width:100%;height:60px;pointer-events:none;z-index:9999;overflow:hidden;';
        const c2 = topDoc.createElement('canvas');
        c2.width = 16 * S; c2.height = 16 * S;
        c2.style.cssText = 'position:absolute;bottom:6px;image-rendering:pixelated;image-rendering:crisp-edges;width:'+(16*S*2)+'px;height:'+(16*S*2)+'px;';
        wrap.appendChild(c2);
        topDoc.body.appendChild(wrap);
        const ctx2 = c2.getContext('2d');

        function draw2(f) {
            ctx2.clearRect(0, 0, c2.width, c2.height);
            const d = F[f];
            for (let y = 0; y < 16; y++)
                for (let px = 0; px < 16; px++) {
                    const c = d[y][dir === 1 ? px : 15 - px];
                    if (c === 0) continue;
                    ctx2.fillStyle = P[c];
                    ctx2.fillRect(px * S, y * S, S, S);
                }
        }

        const W = topDoc.documentElement.clientWidth || 1200;
        function animate() {
            tick++;
            if (tick % 8 === 0) frame = (frame + 1) % 4;
            x += speed * dir;
            if (x > W + 80) dir = -1;
            if (x < -80) dir = 1;
            c2.style.left = x + 'px';
            draw2(frame);
            requestAnimationFrame(animate);
        }
        animate();
    })();
    </script>
    """, height=0)

    # Display Chat History
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about your data..."):
        with st.spinner("Thinking..."):
            process_interaction(user_input=prompt, display_text=prompt)
        st.rerun()