from __future__ import annotations
import os
import time
import uuid
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
import fitz

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(title="Atlas RAG API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class DocumentRecord:
    id: str
    name: str
    pages: int
    chunks: int


class KnowledgeBase:
    """Small in-memory LangChain vector store with document bookkeeping."""

    def __init__(self) -> None:
        self._vectors: InMemoryVectorStore | None = None
        self.documents: dict[str, DocumentRecord] = {}
        self.chunk_ids: dict[str, list[str]] = {}

    @property
    def vectors(self) -> InMemoryVectorStore:
        """Load the embedding model only when a document is first indexed."""
        if self._vectors is None:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                encode_kwargs={"normalize_embeddings": True},
            )
            self._vectors = InMemoryVectorStore(embeddings)
        return self._vectors

    def add(self, record: DocumentRecord, chunks: list[Document]) -> None:
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.vectors.add_documents(chunks, ids=ids)
        self.documents[record.id] = record
        self.chunk_ids[record.id] = ids

    def remove(self, document_id: str) -> None:
        self.vectors.delete(ids=self.chunk_ids.pop(document_id, []))
        self.documents.pop(document_id, None)

    def search(self, question: str, limit: int = 5) -> list[tuple[Document, float]]:
        return self.vectors.similarity_search_with_score(question, k=limit)


knowledge_base = KnowledgeBase()
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=180)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class Citation(BaseModel):
    document: str
    page: int
    excerpt: str
    relevance: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    used_llm: bool


def load_pdf(pdf_bytes: bytes, filename: str) -> list[Document]:
    """Extract PDF pages, then hand LangChain Document objects to the RAG pipeline."""
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [
            Document(
                page_content=page.get_text("text"),
                metadata={"document_name": filename, "page": page_number},
            )
            for page_number, page in enumerate(pdf, start=1)
        ]
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable PDF.") from exc
    return pages


def build_chunks(pages: list[Document], document_id: str) -> list[Document]:
    chunks = splitter.split_documents(pages)
    usable_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    for chunk in usable_chunks:
        chunk.metadata["document_id"] = document_id
    return usable_chunks


def extractive_answer(context: list[tuple[Document, float]]) -> str:
    """Fallback handler that shows complete context excerpts without truncating mid-sentence."""
    if not context:
        return "No relevant passages were found in the uploaded documents."

    formatted_passages = []
    for idx, (doc, _) in enumerate(context[:4], start=1):
        doc_name = doc.metadata.get("document_name", "Document")
        page_num = doc.metadata.get("page", "?")
        passage_content = doc.page_content.strip()
        formatted_passages.append(
            f"**[Source {idx}: {doc_name} (Page {page_num})]**\n{passage_content}"
        )

    full_content = "\n\n---\n\n".join(formatted_passages)
    return (
        f"{full_content}\n\n"
        "> **Note:** Synthesized response was unavailable; presenting relevant source excerpts instead."
    )


def generate_answer(question: str, context: list[tuple[Document, float]]) -> tuple[str, bool]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return extractive_answer(context), False

    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError

        source_text = "\n\n".join(
            f"[Source {index}: {document.metadata['document_name']}, page {document.metadata['page']}]\n{document.page_content}"
            for index, (document, _) in enumerate(context, start=1)
        )

        prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:

"I could not find the answer in the PDF."

Do not make up information.

Context:
{source_text}

User Question:
{question}
"""

        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        # Retry loop for upstream 503 UNAVAILABLE / capacity spikes
        result = None
        for attempt in range(3):
            try:
                result = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                break
            except APIError as err:
                if err.code == 503 and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise

        answer = result.text if result else None
        client.close()
        return answer or extractive_answer(context), True

    except Exception as exc:
        logger.warning("Gemini generation failed; using source excerpt instead: %s", exc)
        return extractive_answer(context), False


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ready"}


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    return [record.__dict__ for record in knowledge_base.documents.values()]


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str]:
    if document_id not in knowledge_base.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    knowledge_base.remove(document_id)
    return {"status": "removed"}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF must be smaller than 25 MB.")

    name = Path(file.filename).name
    pages = load_pdf(data, name)
    document_id = str(uuid.uuid4())
    chunks = build_chunks(pages, document_id)
    if not chunks:
        raise HTTPException(status_code=422, detail="No selectable text was found in this PDF.")

    record = DocumentRecord(id=document_id, name=name, pages=len(pages), chunks=len(chunks))
    knowledge_base.add(record, chunks)
    return {"document": record.__dict__, "message": f"Indexed {record.chunks} passages from {record.pages} pages."}


@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    if not knowledge_base.documents:
        raise HTTPException(status_code=409, detail="Upload at least one PDF before asking a question.")
    matches = knowledge_base.search(payload.question)
    answer, used_llm = generate_answer(payload.question, matches)
    citations = [
        Citation(
            document=document.metadata["document_name"],
            page=document.metadata["page"],
            excerpt=document.page_content[:260].rsplit(" ", 1)[0] + "…" if len(document.page_content) > 260 else document.page_content,
            relevance=round(score, 3),
        )
        for document, score in matches
    ]
    return AskResponse(answer=answer, citations=citations, used_llm=used_llm)