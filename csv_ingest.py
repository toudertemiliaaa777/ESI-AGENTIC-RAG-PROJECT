import os
import hashlib
import shutil
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ESI_FAQ_updated_v3-2.csv"
ALIA_PATH = BASE_DIR / "ALIA.txt"
PDF_PATH = BASE_DIR / "Guide_nouvel_etudiant_ESI.pdf"
COLLECTION_NAME = "esi_rag"
PERSIST_DIR = Path("./chroma_db")
HASH_FILE = PERSIST_DIR / "source_hash.txt"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INGEST_VERSION = "v1-csv-alia-pdf"

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

Context:
{context}
"""


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

def load_pdf_documents(pdf_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    for idx, doc in enumerate(pages):
        doc.metadata.update({
            "doc_type": "reference",
            "category": "ESI New Student Guide",
            "source_type": "Official ESI information",
            "source": "Guide_nouvel_etudiant_ESI.pdf",
            "disclaimer": "",
            "last_updated": "",
            "question": "",
            "page": idx + 1,
        })
    return pages

def load_csv_documents(csv_path: Path) -> list[Document]:
    df = pd.read_csv(csv_path)
    return [
        Document(
            page_content=f"Question: {clean_value(row['question'])}\nAnswer: {clean_value(row['answer'])}",
            metadata=build_metadata(row),
        )
        for _, row in df.iterrows()
    ]


def load_alia_documents(alia_path: Path) -> list[Document]:
    loader = TextLoader(alia_path, encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({
            "doc_type": "reference",
            "category": "ESI Reference (Cité El Alia)",
            "source_type": "community information",
            "source": "ALIA.txt",
            "disclaimer": "",
            "last_updated": "",
            "question": "",
        })
    return docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)


def get_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_source_signature() -> str:
    hasher = hashlib.sha256()
    for path in (CSV_PATH, ALIA_PATH, PDF_PATH):
        hasher.update(get_file_hash(path).encode("utf-8"))
    hasher.update(EMBED_MODEL.encode("utf-8"))
    hasher.update(INGEST_VERSION.encode("utf-8"))
    return hasher.hexdigest()


def get_vector_store(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> Chroma:
    current_signature = get_source_signature()

    if PERSIST_DIR.exists() and HASH_FILE.exists():
        saved_signature = HASH_FILE.read_text(encoding="utf-8").strip()
        if saved_signature == current_signature:
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=str(PERSIST_DIR),
            )
        shutil.rmtree(PERSIST_DIR)

    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    HASH_FILE.write_text(current_signature, encoding="utf-8")
    return vector_store


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


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])


def build_rag_chain():
    csv_docs = load_csv_documents(CSV_PATH)
    alia_docs = load_alia_documents(ALIA_PATH)
    pdf_docs = load_pdf_documents(PDF_PATH)
    documents = csv_docs + alia_docs + pdf_docs

    chunks = chunk_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = get_vector_store(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    prompt = build_prompt()

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
    )
    # llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

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

    return chain, retriever


chain, retriever = build_rag_chain()


def ask_question(question: str, chat_history: list = None) -> str:
    chat_history = chat_history or []
    return chain.invoke({
        "question": question,
        "chat_history": chat_history,
    })


if __name__ == "__main__":
    print("=" * 80)
    print("ESI FAQ Assistant - Interactive Mode")
    print("=" * 80)
    print("Building RAG chain...")
    print("\nType your questions below. Type 'exit' or 'quit' to end the conversation.")
    print("The AI will remember previous questions and answers in this session.\n")
    print("=" * 80 + "\n")
    
    chat_history = []
    question_count = 0
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                print("\n" + "=" * 80)
                print(f"Session ended. Total questions asked: {question_count}")
                print("Thank you for using ESI FAQ Assistant!")
                print("=" * 80)
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            question_count += 1
            print("\nAssistant: ", end="", flush=True)
            
            # Get response with chat history
            response = ask_question(user_input, chat_history)
            print(response)
            
            # Add user message and assistant response to chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            
            print("\n" + "-" * 80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("Session interrupted by user.")
            print(f"Total questions asked: {question_count}")
            print("=" * 80)
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")