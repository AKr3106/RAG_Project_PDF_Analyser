# Atlas — PDF Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) app for asking cited questions about PDF documents.

## Description

Atlas is a powerful, local-first document analysis tool that leverages Retrieval-Augmented Generation (RAG) to help you understand and extract information from your PDF documents. By combining a sleek, modern React frontend with a robust FastAPI and LangChain backend, Atlas processes your uploaded PDFs, chunks them, and stores their semantic embeddings. When you ask a question, it retrieves the most relevant sections and uses an LLM (like OpenAI's models) to generate a precise, context-aware answer, complete with exact source citations from your original documents. It's designed to streamline research, document review, and knowledge extraction.

## Interface preview

Atlas provides a focused workspace for uploading PDF documents, asking natural-language questions, and reviewing source-grounded responses with page citations.

![Atlas PDF Knowledge Assistant interface](assets/atlas-interface.png)

## Tech-Stack

- **Frontend:** React + Vite + custom CSS (no UI framework required)
- **Backend:** FastAPI + PyMuPDF + LangChain (recursive splitter, Hugging Face embeddings, in-memory vector store)
- **Generation:** OpenAI, when `OPENAI_API_KEY` is configured; otherwise a source-grounded extractive response

## Run locally

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want generative answers.

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`. The Vite development server proxies API calls to the backend.

## API

- `POST /api/upload` — multipart PDF upload; returns document and chunk counts.
- `POST /api/ask` — JSON `{ "question": "..." }`; returns answer and page citations.
- `GET /api/documents` — indexed documents.
- `DELETE /api/documents/{id}` — remove an indexed document.

The included LangChain `InMemoryVectorStore` is process-local and intentionally simple for local use. Replace it with Chroma, Qdrant, or pgvector for multi-user/production persistence.
