from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware #required so the requests are not blocked between the ports
from pydantic import BaseModel #data checker
import pdfplumber #reading pdf
import faiss #vector database, facebook library
import numpy as np
from sentence_transformers import SentenceTransformer #embedding model, text->numbers
import google.generativeai as genai #library to call LLM
from dotenv import load_dotenv #.env file for the api key
import os #for accessing the env

load_dotenv()

# configure gemini with api key from .env
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# load gemini 2.5 flash model
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model...")
# BAAI/bge-base-en-v1.5 — better than MiniLM, outputs 768 dimensions
# uses query/passage prefix format for better retrieval accuracy
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

chunks = []
metadata = []
documents = []
faiss_index = None


def extract_text_from_pdf(filepath):
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append((i + 1, text))
    return pages


def chunk_text(text, chunk_size=500, overlap=50):
    result = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        result.append(text[start:end])
        start += chunk_size - overlap
    return result


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # 768 for BAAI/bge-base-en-v1.5
    faiss.normalize_L2(embeddings)   # normalize before adding — required for cosine similarity
    index = faiss.IndexFlatIP(dimension)  # IndexFlatIP = inner product = cosine after normalization
    index.add(embeddings)
    return index


def rebuild_faiss():
    """rebuilds the faiss index from scratch after a deletion"""
    global faiss_index
    if not chunks:
        faiss_index = None  # no chunks left, reset index
        return
    # passage: prefix tells BAAI/bge this is a document chunk (not a query)
    all_embeddings = embedder.encode(
        [f"passage: {chunk}" for chunk in chunks],
        show_progress_bar=False
    )
    all_embeddings = np.array(all_embeddings).astype("float32")
    faiss_index = build_faiss_index(all_embeddings)


@app.get("/documents")
def list_documents():
    return {"documents": documents}


@app.post("/upload")  # this runs whenever a pdf is uploaded
async def upload_pdf(file: UploadFile = File(...)):
    global chunks, metadata, documents, faiss_index

    # save pdf to disk
    filepath = f"uploads/{file.filename}"
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    pages = extract_text_from_pdf(filepath)
    if not pages:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    new_chunks = []
    new_metadata = []
    for page_num, page_text in pages:
        page_chunks = chunk_text(page_text)
        for chunk in page_chunks:
            new_chunks.append(chunk)
            new_metadata.append({"filename": file.filename, "page": page_num})

    chunks.extend(new_chunks)
    metadata.extend(new_metadata)
    if file.filename not in documents:
        documents.append(file.filename)

    # passage: prefix for document chunks — matches rebuild_faiss
    all_embeddings = embedder.encode(
        [f"passage: {chunk}" for chunk in chunks],
        show_progress_bar=False
    )
    all_embeddings = np.array(all_embeddings).astype("float32")
    faiss_index = build_faiss_index(all_embeddings)

    return {
        "message": f"Successfully processed {file.filename}",
        "chunks_added": len(new_chunks),
        "total_chunks": len(chunks)
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """removes a document and all its chunks from memory and rebuilds faiss"""
    global chunks, metadata, documents

    if filename not in documents:
        raise HTTPException(status_code=404, detail="Document not found")

    # filter out all chunks that belong to this file
    new_chunks = []
    new_metadata = []
    for i, meta in enumerate(metadata):
        if meta["filename"] != filename:
            new_chunks.append(chunks[i])
            new_metadata.append(metadata[i])

    chunks = new_chunks
    metadata = new_metadata
    documents = [d for d in documents if d != filename]

    # delete the actual pdf file from disk
    filepath = f"uploads/{filename}"
    if os.path.exists(filepath):
        os.remove(filepath)

    # rebuild faiss without the deleted doc
    rebuild_faiss()

    return {"message": f"Deleted {filename} successfully"}


class QuestionRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(request: QuestionRequest):
    if faiss_index is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")

    # STEP 1 — convert question to a vector
    # query: prefix tells BAAI/bge this is a search query (not a document)
    question_embedding = embedder.encode([f"query: {request.question}"])
    question_embedding = np.array(question_embedding).astype("float32")
    faiss.normalize_L2(question_embedding)  # normalize for cosine similarity

    # STEP 2 — search faiss for top 5 most similar chunks
    k = min(5, len(chunks))
    distances, indices = faiss_index.search(question_embedding, k)

    # STEP 3 — retrieve the actual text of those chunks
    retrieved = []
    sources = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved.append(chunks[idx])
            sources.append(metadata[idx])

    context = "\n\n---\n\n".join(retrieved)  # joins the 5 chunks into one big context

    # STEP 4 — build prompt and send to gemini
    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the context below.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."

CONTEXT:
{context}

QUESTION:
{request.question}

ANSWER:"""

    # STEP 5 — gemini generates the answer (it only sees the 5 most relevant chunks faiss found)
    response = gemini_model.generate_content(prompt)
    answer = response.text

    return {"answer": answer, "sources": sources}
