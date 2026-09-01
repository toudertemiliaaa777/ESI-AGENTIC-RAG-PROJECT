import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from csv_ingest import ask_question

st.set_page_config(
    page_title="ESI FAQ Assistant",
    page_icon="🎓",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b0b0d;
        }
        section[data-testid="stSidebar"] {
            background-color: #16161a;
            border-right: 1px solid #2a2a30;
        }
        h1, h2, h3 {
            color: #f4efe6 !important;
        }
        p, span, label, li {
            color: #d8d3ca;
        }
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
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 🎓 ESI FAQ Assistant")
    st.write(
        "Ask questions about ESI (École nationale Supérieure d'Informatique) — "
        "admissions, the concours, programs, student life, and more."
    )
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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "lc_history" not in st.session_state:
    st.session_state.lc_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("e.g. What is the concours?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching the ESI knowledge base..."):
            try:
                answer = ask_question(user_input, chat_history=st.session_state.lc_history)
            except Exception as e:
                answer = f"Something went wrong while answering: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.lc_history.append(HumanMessage(content=user_input))
    st.session_state.lc_history.append(AIMessage(content=answer))