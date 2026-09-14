import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from csv_ingest import ask_question

st.set_page_config(
    page_title="ESI FAQ Assistant",
    page_icon="🎓",
    layout="centered",
)

COMMON_QUESTIONS = [
    "What is ESI?",
    "What are the ESI clubs?",
    "How does the concours to enter 1CS work?",
    "What specializations does ESI offer?",
    "What is the girls' residence like?",
    "What is the difference between note eliminatoire and rachat?",
]

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


DARK_CSS = """
<style>
    .stApp { background-color: #0b0b0d; }
    section[data-testid="stSidebar"] {
        background-color: #16161a;
        border-right: 1px solid #2a2a30;
    }
    h1, h2, h3 { color: #f4efe6 !important; }
    p, span, label, li { color: #d8d3ca; }
    [data-testid="stChatMessage"] {
        background-color: #16161a;
        border: 1px solid #2a2a30;
        border-radius: 14px;
        padding: 10px 14px;
    }
    .stChatInput textarea {
        background-color: #16161a !important;
        color: #f4efe6 !important;
    }
    .badge {
        display: inline-block;
        background: rgba(179, 18, 28, .15);
        border: 1px solid rgba(224, 32, 43, .4);
        color: #e0202b;
        border-radius: 50px;
        padding: 4px 14px;
        font-size: .8rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .stButton button {
        background-color: #1c1c22;
        color: #f4efe6;
        border: 1px solid #2a2a30;
        border-radius: 10px;
        text-align: left;
    }
    .stButton button:hover {
        border-color: #e0202b;
        color: #f4efe6;
    }
</style>
"""

LIGHT_CSS = """
<style>
    .stApp { background-color: #fafaf8; }
    section[data-testid="stSidebar"] {
        background-color: #f0efe9;
        border-right: 1px solid #dcdad2;
    }
    h1, h2, h3 { color: #1a1a1a !important; }
    p, span, label, li { color: #2e2e2e; }
    [data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #e0ded6;
        border-radius: 14px;
        padding: 10px 14px;
    }
    .stChatInput textarea {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }
    .badge {
        display: inline-block;
        background: rgba(179, 18, 28, .08);
        border: 1px solid rgba(224, 32, 43, .3);
        color: #b3121c;
        border-radius: 50px;
        padding: 4px 14px;
        font-size: .8rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .stButton button {
        background-color: #ffffff;
        color: #1a1a1a;
        border: 1px solid #dcdad2;
        border-radius: 10px;
        text-align: left;
    }
    .stButton button:hover {
        border-color: #b3121c;
        color: #1a1a1a;
    }
</style>
"""

st.markdown(DARK_CSS if st.session_state.theme == "dark" else LIGHT_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🎓 ESI FAQ Assistant")
    st.write(
        "Ask questions about ESI (École nationale Supérieure d'Informatique) — "
        "admissions, the concours, programs, student life, and more."
    )

    st.markdown("---")
    theme_label = "☀️ Switch to light mode" if st.session_state.theme == "dark" else "🌙 Switch to dark mode"
    if st.button(theme_label, use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown("---")
    st.markdown("**Common questions**")
    for question in COMMON_QUESTIONS:
        if st.button(question, use_container_width=True, key=f"common_{question}"):
            st.session_state.pending_question = question

    st.markdown("---")
    st.markdown("**Languages supported**")
    st.write("English · Français · العربية")

    st.markdown("---")
    st.markdown("**Note**")
    st.write(
        "Answers are grounded only in the ESI knowledge base "
        "(FAQ entries + reference documents). Off-topic questions will be declined."
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_history = []
        st.rerun()

st.markdown('<span class="badge">ESI · École nationale Supérieure d\'Informatique</span>', unsafe_allow_html=True)
st.title("Ask me anything about ESI")
st.caption("Grounded answers from the official ESI FAQ knowledge base.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("e.g. What is the concours?")

question_to_ask = None
if st.session_state.pending_question:
    question_to_ask = st.session_state.pending_question
    st.session_state.pending_question = None
elif user_input:
    question_to_ask = user_input

if question_to_ask:
    st.session_state.messages.append({"role": "user", "content": question_to_ask})
    with st.chat_message("user"):
        st.markdown(question_to_ask)

    with st.chat_message("assistant"):
        with st.spinner("Searching the ESI knowledge base..."):
            try:
                answer = ask_question(question_to_ask, chat_history=st.session_state.lc_history)
            except Exception as e:
                answer = f"Something went wrong while answering: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.lc_history.append(HumanMessage(content=question_to_ask))
    st.session_state.lc_history.append(AIMessage(content=answer))