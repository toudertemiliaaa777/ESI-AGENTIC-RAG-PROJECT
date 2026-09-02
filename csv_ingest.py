from pathlib import Path
import os

from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document

load_dotenv()

CSV_PATH = Path(__file__).resolve().parent / "ESI_FAQ_updated_v3-2.csv"
ALIA_PATH = Path(__file__).resolve().parent / "ALIA.txt"

df = pd.read_csv(CSV_PATH)
from langchain_community.document_loaders import TextLoader

loader = TextLoader(ALIA_PATH, encoding="utf-8")
alia_docs = loader.load()

for doc in alia_docs:
    doc.metadata.update({
        "doc_type": "reference",
        "category": "ESI Reference (Cité El Alia)",
        "source_type": "community information",
        "source": "ALIA.txt",
        "disclaimer": "",
        "last_updated": "",
        "question": "",  # no natural Q&A pairing for this document
    })

def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_metadata(row):
    question = clean_value(row["question"]) if "question" in row.index else ""
    answer = clean_value(row["answer"]) if "answer" in row.index else ""

    return {
        "doc_type": "faq",
        "category": clean_value(row["category"]) if "category" in row.index else "",
        "source_type": clean_value(row["source_type"]) if "source_type" in row.index else "",
        "source": clean_value(row["source"]) if "source" in row.index else "",
        "disclaimer": clean_value(row["disclaimer"]) if "disclaimer" in row.index else "",
        "last_updated": clean_value(row["last_updated"]) if "last_updated" in row.index else "",
        "question": question,
        "answer_length": len(answer),
        "has_disclaimer": bool(clean_value(row["disclaimer"])) if "disclaimer" in row.index else False,
    }


documents = [
    Document(
        page_content=f"Question: {clean_value(row['question'])}\nAnswer: {clean_value(row['answer'])}",
        metadata=build_metadata(row),
    )
    for _, row in df.iterrows()
]

csv_docs = documents

documents = csv_docs + alia_docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)

chunks = splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vector_store = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     collection_name="esi_faq",
#     persist_directory="./chroma_db",
# )

COLLECTION_NAME = "esi_faq"
PERSIST_DIR = "./chroma_db"

def get_vector_store():
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    if vector_store._collection.count() == 0:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )
    return vector_store

vector_store = get_vector_store()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)




from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

def format_docs(docs):
    formatted = []
    for doc in docs:
        formatted.append(
            f"""
Content:
{doc.page_content}

Source:
{doc.metadata.get("source") or "Unknown"}

Source type:
{doc.metadata.get("source_type") or "Unspecified"}
"""
        )
    return "\n\n---\n\n".join(formatted)

SYSTEM_PROMPT = """\
You are a strict ESI FAQ assistant.
Your domain is ONLY ESI (École nationale Supérieure d’Informatique) information, student life, admissions, programs, research, clubs, and related institutional knowledge.

Rules:
1. Answer only questions related to ESI, its programs, admissions, student life, research, clubs, contacts, or official/non-official information provided in the context.
2. Use ONLY the context provided below. Do not use outside knowledge, assumptions, or general knowledge.
3. If the user asks something unrelated to ESI or the context does not contain enough information to answer reliably, respond exactly with:
   "I can only answer ESI-related questions based on the provided knowledge base."
   If the question IS ESI-related and the context contains information that partially or specifically
   answers it (even if narrower than the question, e.g. concours statistics for a general "what is the
   concours" question), synthesize the best answer you can from that information instead of refusing,
   and note which specific aspect(s) it covers.
4. Do not be chatty, casual, or speculative. Keep answers concise, factual, and direct.
5. If the answer is based on an official ESI source, say "Official source".
6. If the answer is based on a non-official or student/community source, say "Non-official source".
7. Always include the source in your answer, using the format:
   Source: <source>
   Source type: <Official source / Non-official source>
8. If the context contains a disclaimer, include it in the answer when relevant.
9. If the answer is uncertain, clearly say that the information should be verified from the official ESI source.
10. Detect the language the user asked in (English, French, or Arabic) and respond in that same language,
    translating the context content as needed without changing any facts, numbers, or names.
11. Use the chat history only to understand what the user is referring to (e.g. follow-up questions).
    Never let earlier turns override rule 2 — every fact still has to come from the context below.
11. Use the chat history only to understand what the user is referring to. Never let earlier turns
    override rule 2 — every fact still has to come from the context below.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])

# llm = ChatGoogleGenerativeAI(
#     model="gemini-flash-latest",
#     temperature=0,
# )
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm
    | StrOutputParser()
)


def ask_question(question: str, chat_history: list = None) -> str:
    chat_history = chat_history or []
    return chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })


if __name__ == "__main__":
    test_query = "What is ESI?"
    print("Building RAG chain and testing with one query...")
    print(f"Q: {test_query}")
    print("A:", ask_question(test_query))