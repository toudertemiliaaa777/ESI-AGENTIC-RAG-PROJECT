import hashlib
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from langchain_core.documents import Document

load_dotenv()

CSV_PATH = Path(__file__).resolve().parent / "ESI_FAQ_updated_v3-2.csv"
ALIA_PATH = Path(__file__).resolve().parent / "ALIA.txt"
WEBSITE_SOURCES = [
    {
    "url": "https://tresor.cse.club/",
    "category": "ESI Curriculum (Trésor ESI - Student Resource)",
    "source_type": "Non-official/community source",
    },
    {
            "url": "https://talents.esi.dz/scolar/index",
            "category": "ESI Official Website",
            "source_type": "Official ESI information",
    },
    
    {
                "url": "https://www.esi.dz/",
                "category": "ESI Official Website",
                "source_type": "Official ESI information",
        },
]
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
HASH_FILE = CHROMA_DIR / "source_hash.txt"



# --------------------ALIA FILE --------------(text)
from langchain_community.document_loaders import TextLoader

loader = TextLoader(ALIA_PATH, encoding="utf-8")
alia_docs = loader.load()

for doc in alia_docs:
    doc.metadata.update({
        "doc_type": "reference",
        "category": "ESI Reference (Cité El Alia)",
        "source_type": "Official ESI information",
        "source": "ALIA.txt",
        "disclaimer": "",
        "last_updated": "",
        "question": "",  # no natural Q&A pairing for this document
    })


#-------------ESI WEBSITES -------------(url)
from langchain_community.document_loaders import TextLoader, WebBaseLoader



def load_website_docs():
    web_docs = []
    for entry in WEBSITE_SOURCES:
        try:
            loaded = WebBaseLoader(entry["url"]).load()
        except Exception as e:
            print(f"Warning: failed to load {entry['url']}: {e}")
            continue
        for doc in loaded:
            doc.page_content = "\n".join(
                line.strip() for line in doc.page_content.splitlines() if line.strip()
            )
            doc.metadata.update({
                "doc_type": "reference",
                "category": entry["category"],
                "source_type": entry["source_type"],
                "source": entry["url"],
                "disclaimer": "",
                "last_updated": "",
                "question": "",
            })
            web_docs.append(doc)
    return web_docs


website_docs = load_website_docs()


def clean_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_file_hash(path, extra=""):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    if extra:
        hasher.update(extra.encode("utf-8"))
    return hasher.hexdigest()

def get_text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

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
        "answer": answer,
        "answer_length": len(answer),
        "has_disclaimer": bool(clean_value(row["disclaimer"])) if "disclaimer" in row.index else False,
    }


df = pd.read_csv(CSV_PATH)


def build_page_content(row):
    question = clean_value(row["question"]) if "question" in row.index else ""
    answer = clean_value(row["answer"]) if "answer" in row.index else ""
    return f"{question}\n{answer}".strip()


documents = [
    Document(
        page_content=build_page_content(row),
        metadata=build_metadata(row),
    )
    for _, row in df.iterrows()
]

csv_docs = documents

documents = csv_docs + alia_docs + website_docs

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)

chunks = splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INGEST_VERSION = "v3-alia-metadata"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_vector_store():
    website_urls_key = "|".join(entry["url"] for entry in WEBSITE_SOURCES)
    current_hash = (
        get_file_hash(CSV_PATH, extra=EMBEDDING_MODEL_NAME + INGEST_VERSION)
        + get_file_hash(ALIA_PATH)
        + get_text_hash(website_urls_key)
    )

    if CHROMA_DIR.exists() and HASH_FILE.exists():
        saved_hash = HASH_FILE.read_text(encoding="utf-8").strip()

        if saved_hash == current_hash:
            print("CSV unchanged. Loading existing Chroma database...")
            return Chroma(
                collection_name="esi_faq",
                embedding_function=embeddings,
                persist_directory=str(CHROMA_DIR),
            )

    print("CSV changed or database doesn't exist. Creating Chroma database...")
    CHROMA_DIR.mkdir(exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="esi_faq",
        persist_directory=str(CHROMA_DIR),
    )

    HASH_FILE.write_text(current_hash, encoding="utf-8")
    return vector_store


vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
RELEVANCE_THRESHOLD = 0.35
STRONG_MATCH_THRESHOLD = 0.55


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
3. If the user asks something unrelated to ESI, respond exactly with:
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


from langchain_core.messages import HumanMessage, AIMessage


from collections import Counter


def get_display_label(doc) -> str:
    """A human-readable label for a retrieved chunk, used in 'did you mean'
    suggestions. FAQ rows use their question; other documents (like
    ALIA.txt) fall back to a snippet of their own content."""
    question = doc.metadata.get("question", "").strip()
    if question:
        return question

    snippet = doc.page_content.strip().split("\n")[0].strip()
    if len(snippet) > 100:
        snippet = snippet[:100].rsplit(" ", 1)[0] + "..."
    return snippet


def get_close_matches(question: str, k: int = 5):
    try:
        results = vector_store.similarity_search_with_relevance_scores(question, k=k)
    except Exception:
        results = [(doc, None) for doc in retriever.invoke(question)]

    seen = set()
    matches = []
    categories = []
    top_score = None
    for doc, score in results:
        if top_score is None:
            top_score = score
        label = get_display_label(doc)
        cat = doc.metadata.get("category", "").strip() or "General"
        if label and label not in seen:
            seen.add(label)
            matches.append(label)
            categories.append(cat)
    return matches, top_score, categories


def is_same_topic_cluster(categories: list, min_ratio: float = 0.6) -> bool:
    if not categories:
        return False
    most_common_count = Counter(categories).most_common(1)[0][1]
    return (most_common_count / len(categories)) >= min_ratio


NO_MATCH_FLOOR = 0.05


def ask_question(question: str, chat_history: list = None) -> str:
    chat_history = chat_history or []
    search_question = question

    close_matches, top_score, categories = get_close_matches(search_question)

    if chat_history and (top_score is None or top_score < RELEVANCE_THRESHOLD):
        last_human = next(
            (m.content for m in reversed(chat_history) if isinstance(m, HumanMessage)),
            "",
        )
        if last_human:
            combined = f"{last_human} {question}"
            combined_matches, combined_score, combined_categories = get_close_matches(combined)
            if combined_score is not None and (top_score is None or combined_score > top_score):
                search_question = combined
                close_matches, top_score, categories = combined_matches, combined_score, combined_categories

    if top_score is None or top_score < NO_MATCH_FLOOR:
        return "I can only answer ESI-related questions based on the provided knowledge base."

    same_topic = is_same_topic_cluster(categories)

    if top_score < STRONG_MATCH_THRESHOLD and not same_topic and close_matches:
        bullet_list = "\n".join(f"- {q}" for q in close_matches)
        return (
            "Your question is ESI-related, but it's broader than what I have an exact match for. "
            "Did you mean one of these?\n\n"
            f"{bullet_list}\n\n"
            "Tell me which one (or rephrase) and I'll give you the exact answer."
        )

    return chain.invoke({
        "question": search_question,
        "chat_history": chat_history,
    })


if __name__ == "__main__":
    test_query = "what is esi concours give me informations about it ?"
    print("Building RAG chain and testing with one query...")
    print(f"Q: {test_query}")
    print("A:", ask_question(test_query))