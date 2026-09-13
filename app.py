import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from csv_ingest import ask_question

# Page Configuration
st.set_page_config(
    page_title="ESI FAQ Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# Sidebar Controls & Theme Selection
with st.sidebar:
    st.markdown("### 🎓 ESI FAQ Assistant")
    st.write(
        "Ask questions about ESI (École nationale Supérieure d'Informatique) — "
        "admissions, the concours, programs, student life, and more."
    )
    st.markdown("---")
    
    st.markdown("**Theme Mode**")
    theme_choice = st.radio(
        "Choose Theme",
        options=["Dark 🌙", "Light ☀️"],
        index=0 if st.session_state.theme == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.theme = "Dark" if "Dark" in theme_choice else "Light"

    st.markdown("---")
    st.markdown("**Languages Supported**")
    st.write("English · Français · العربية")
    
    st.markdown("---")
    st.markdown("**Knowledge Base Note**")
    st.caption(
        "Answers are grounded strictly in the official ESI knowledge base "
        "(FAQ entries + reference documents). Off-topic queries are declined."
    )
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.rerun()

# Dynamic Theme CSS Injection
if st.session_state.theme == "Dark":
    css = """
    <style>
        .stApp {
            background-color: #0b0b0d;
            color: #d8d3ca;
        }
        section[data-testid="stSidebar"] {
            background-color: #16161a;
            border-right: 1px solid #2a2a30;
        }
        h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #f4efe6 !important;
        }
        p, span, label, li {
            color: #d8d3ca;
        }
        [data-testid="stChatMessage"] {
            background-color: #16161a;
            border: 1px solid #2a2a30;
            border-radius: 12px;
            padding: 12px 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .stChatInput textarea {
            background-color: #16161a !important;
            color: #f4efe6 !important;
            border: 1px solid #2a2a30 !important;
        }
        .badge {
            display: inline-block;
            background: rgba(224, 32, 43, 0.15);
            border: 1px solid rgba(224, 32, 43, 0.4);
            color: #ff4d4d;
            border-radius: 50px;
            padding: 4px 14px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 12px;
        }
    </style>
    """
else:
    css = """
    <style>
        .stApp {
            background-color: #f8f9fa;
            color: #2d3748;
        }
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #1a202c !important;
        }
        p, span, label, li {
            color: #2d3748;
        }
        [data-testid="stChatMessage"] {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 16px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        }
        .stChatInput textarea {
            background-color: #ffffff !important;
            color: #1a202c !important;
            border: 1px solid #cbd5e1 !important;
        }
        .badge {
            display: inline-block;
            background: rgba(224, 32, 43, 0.08);
            border: 1px solid rgba(224, 32, 43, 0.25);
            color: #c53030;
            border-radius: 50px;
            padding: 4px 14px;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 12px;
        }
    </style>
    """

st.markdown(css, unsafe_allow_html=True)

# Main UI Header
st.markdown('<span class="badge">ESI · École nationale Supérieure d\'Informatique</span>', unsafe_allow_html=True)
st.title("Ask me anything about ESI")
st.caption("Grounded answers sourced directly from the official ESI FAQ knowledge base.")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Quick Suggested Prompts (shown when conversation history is empty)
if not st.session_state.messages:
    st.markdown("#### 💡 Frequently Asked Questions")
    col1, col2 = st.columns(2)
    prompt_selected = None
    with col1:
        if st.button("What is the ESI concours?", use_container_width=True):
            prompt_selected = "What is the ESI concours?"
        if st.button("What are the admission requirements?", use_container_width=True):
            prompt_selected = "What are the admission requirements?"
    with col2:
        if st.button("Tell me about student clubs at ESI", use_container_width=True):
            prompt_selected = "Tell me about student clubs at ESI"
        if st.button("Where is ESI located?", use_container_width=True):
            prompt_selected = "Where is ESI located?"

    if prompt_selected:
        user_input = prompt_selected
    else:
        user_input = st.chat_input("e.g. What is the concours?")
else:
    user_input = st.chat_input("e.g. What is the concours?")

# Query Processing & LangChain Execution
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the ESI knowledge base..."):
            try:
                answer = ask_question(user_input, chat_history=st.session_state.lc_history)
            except Exception as e:
                answer = f"Something went wrong while retrieving the answer: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.lc_history.append(HumanMessage(content=user_input))
    st.session_state.lc_history.append(AIMessage(content=answer))