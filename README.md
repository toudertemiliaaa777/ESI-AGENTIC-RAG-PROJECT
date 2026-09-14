# ESI FAQ Assistant

A retrieval-augmented chatbot that answers questions about **ESI (École nationale
Supérieure d'Informatique, Algiers)** — admissions, the concours, specializations,
student clubs, housing, and general campus life — grounded in a curated knowledge
base rather than the model's general training data.

Built with LangChain, Chroma, and Streamlit.

## How it works

1. **Ingestion** (`csv_ingest.py`) loads three kinds of sources:
   - `ESI_FAQ_updated_v3-2.csv` — hand-curated FAQ rows (question, answer, category, source, disclaimer)
   - `ALIA.txt` — a reference document on the El Alia student residence
   - `Guide_nouvel_etudiant_ESI.pdf` — a new-student survival guide covering admissions, clubs, labs, housing, and more

   All three are chunked, embedded with a sentence-transformer model, and stored
   in a local Chroma vector database (`chroma_db/`).

2. **Retrieval** pulls the most relevant chunks for a given question.

3. **Generation** hands those chunks, the question, and recent chat history to
   an LLM (Groq by default), which is instructed to answer only from the
   retrieved context, cite its source, and avoid inventing information not
   present in the knowledge base.

4. **Interface** (`app.py`) is a Streamlit chat app with dark/light mode and a
   set of common-question shortcuts in the sidebar.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
cd ESI_PROJECT
uv sync
```

Create a `.env` file in this folder with whichever API key your active LLM needs:

```
GROQ_API_KEY="your-key-here"
# GOOGLE_API_KEY="your-key-here"   # only if you switch back to Gemini
```

## Running the app

```bash
uv run streamlit run app.py
```

Or run the ingestion/chat pipeline directly from the terminal:

```bash
uv run python csv_ingest.py
```

## Rebuilding the knowledge base

The vector store caches itself using a hash of the source files, the embedding
model name, and an internal version string. If you edit the CSV, `ALIA.txt`,
or the PDF, it *should* rebuild automatically — but if you change chunking
logic, retrieval settings, or the system prompt, it's safest to force a clean
rebuild manually:

```bash
rm -r chroma_db   # or Remove-Item -Recurse -Force chroma_db on Windows
```

## Project structure

```
ESI_PROJECT/
├── app.py                          # Streamlit chat interface
├── csv_ingest.py                   # ingestion, retrieval, and RAG chain
├── ESI_FAQ_updated_v3-2.csv        # curated FAQ source
├── ALIA.txt                        # El Alia residence reference doc
├── Guide_nouvel_etudiant_ESI.pdf   # new-student guide
├── chroma_db/                      # local vector store (gitignored)
├── pyproject.toml                  # uv project config
└── .env                            # API keys (gitignored, not committed)
```

## Switching the LLM

The active model is set in `build_rag_chain()` inside `csv_ingest.py`. Groq
and Gemini are both wired up — comment/uncomment the relevant block to switch:

```python
# llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
```

## Notes on the knowledge base

Some content in the guide comes from unofficial/community sources (student
testimonials, Facebook groups, Instagram accounts) rather than ESI's official
channels. The assistant is instructed to answer only from what's in the
knowledge base and to flag when information should be verified against an
official source — but always double-check anything time-sensitive (deadlines,
thresholds, procedures) directly with ESI's administration.
