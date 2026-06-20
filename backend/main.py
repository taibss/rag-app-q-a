from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware #required so the requests are not blocked between the ports
from fastapi.responses import JSONResponse
from pydantic import BaseModel #data checker
import pdfplumber #reading pdf
import faiss #vector database, facebook library
import numpy as np
import requests
from dotenv import load_dotenv #.env file for the api key
import os #for accessing the env
import chromadb #persistent vector storage

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NIM_API_KEY = os.getenv("NIM_API_KEY")
EMBED_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
EMBED_DIM = 2048
OPENROUTER_EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "*", "Access-Control-Allow-Headers": "*"},
    )

# use /data on Render (persistent disk), fallback to local for development
CHROMA_PATH = "/data/chroma_db" if os.path.exists("/data") else "./chroma_db"
COLLECTION_NAME = "documents"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def create_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


collection = create_collection()


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


def get_stored_embed_dim() -> int | None:
    results = collection.get(include=["embeddings"])
    embeddings = results["embeddings"]
    if embeddings is None or len(embeddings) == 0:
        return None
    return len(embeddings[0])


def get_embeddings(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Embed text via OpenRouter (Nemotron Embed VL 1B V2)."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set in .env")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            OPENROUTER_EMBED_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "input": batch if len(batch) > 1 else batch[0],
                "encoding_format": "float",
            },
            timeout=120,
        )
        if not resp.ok:
            raise HTTPException(
                status_code=502,
                detail=f"OpenRouter embedding error: {resp.text}",
            )
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        all_embeddings.extend(item["embedding"] for item in data)

    return np.array(all_embeddings, dtype="float32")


def get_chat_answer(prompt: str) -> str:
    """Generate an answer via OpenRouter chat API."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not set in .env")

    resp = requests.post(
        OPENROUTER_CHAT_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    if not resp.ok:
        detail = resp.text
        if resp.status_code == 402:
            detail = (
                "OpenRouter free credits exhausted for this chat model. "
                "Use a :free model (default is set) or add credits at openrouter.ai/settings/credits."
            )
        raise HTTPException(status_code=502, detail=f"OpenRouter chat error: {detail}")
    answer = resp.json()["choices"][0]["message"]["content"]
    if not answer:
        raise HTTPException(status_code=502, detail="OpenRouter returned an empty answer")
    return answer


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # 2048 for Nemotron Embed VL 1B V2
    faiss.normalize_L2(embeddings)   # normalize for cosine similarity
    index = faiss.IndexFlatIP(dimension)  # inner product = cosine after normalization
    index.add(embeddings)
    return index


def get_document_filenames(user: str):
    results = collection.get(where={"user": user}, include=["metadatas"])
    if not results["metadatas"]:
        return []
    return sorted(set(m["filename"] for m in results["metadatas"]))


def get_user_index(user: str):
    """load one user's chunks from chromadb and build a faiss index"""
    results = collection.get(
        where={"user": user},
        include=["documents", "metadatas", "embeddings"],
    )
    if not results["documents"]:
        return None, [], []
    embeddings = np.array(results["embeddings"]).astype("float32")
    if embeddings.shape[1] != EMBED_DIM:
        return None, [], []
    index = build_faiss_index(embeddings)
    return index, results["documents"], results["metadatas"]


def reset_collection():
    """drop and recreate chromadb collection (needed when embedding dimension changes)"""
    global collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = create_collection()


def clear_user_documents(user: str):
    """remove all chunks for one user"""
    results = collection.get(where={"user": user})
    if results["ids"]:
        collection.delete(ids=results["ids"])


def purge_stale_embeddings():
    """drop legacy chunks from old models or before per-user storage"""
    try:
        results = collection.get(include=["embeddings", "metadatas"])
        if not results["documents"]:
            reset_collection()
            return
        embeddings = np.array(results["embeddings"]).astype("float32")
        if embeddings.shape[1] != EMBED_DIM:
            reset_collection()
            print("Cleared stale embeddings from old model.")
    except Exception as e:
        print(f"ChromaDB startup check: {e}")


purge_stale_embeddings()


@app.get("/documents")
def list_documents(user: str):
    return {"documents": get_document_filenames(user)}


@app.delete("/documents")
def delete_all_documents(user: str):
    clear_user_documents(user)
    return {"message": f"All documents cleared for {user}"}


@app.post("/upload")  # this runs whenever a pdf is uploaded
async def upload_pdf(file: UploadFile = File(...), user: str = Form(...)):
    user = user.strip()
    if not user:
        raise HTTPException(status_code=400, detail="User name is required")

    stored_dim = get_stored_embed_dim()
    if stored_dim is not None and stored_dim != EMBED_DIM:
        reset_collection()

    os.makedirs("uploads", exist_ok=True)

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
        for i, chunk in enumerate(page_chunks):
            new_chunks.append(chunk)
            new_metadata.append({"filename": file.filename, "page": page_num, "user": user})

    new_embeddings = get_embeddings(new_chunks)
    new_embeddings_list = new_embeddings.tolist()

    existing = collection.get(where={"user": user})
    base = len(existing["ids"]) if existing["ids"] else 0
    ids = [f"{user}_{file.filename}_{base + i}" for i in range(len(new_chunks))]
    try:
        collection.add(
            documents=new_chunks,
            metadatas=new_metadata,
            embeddings=new_embeddings_list,
            ids=ids,
        )
    except Exception as e:
        if "dimension" not in str(e).lower():
            raise
        reset_collection()
        collection.add(
            documents=new_chunks,
            metadatas=new_metadata,
            embeddings=new_embeddings_list,
            ids=ids,
        )

    user_chunks = get_user_index(user)[1]

    return {
        "message": f"Successfully processed {file.filename}",
        "chunks_added": len(new_chunks),
        "total_chunks": len(user_chunks),
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str, user: str):
    """removes a document and all its chunks from chromadb and rebuilds faiss"""
    results = collection.get(where={"$and": [{"filename": filename}, {"user": user}]})
    if not results["ids"]:
        raise HTTPException(status_code=404, detail="Document not found")

    collection.delete(ids=results["ids"])

    # delete the actual pdf file from disk
    filepath = f"uploads/{filename}"
    if os.path.exists(filepath):
        os.remove(filepath)

    return {"message": f"Deleted {filename} successfully"}


class QuestionRequest(BaseModel):
    question: str
    user: str
@app.post("/tts")
async def text_to_speech(request: Request):
    """convert text to speech using OpenRouter Kokoro TTS"""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    resp = requests.post(
        "https://openrouter.ai/api/v1/audio/speech",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "hexgrad/kokoro-82m",
            "input": text,
            "voice": "af_heart",
            "response_format": "mp3",
        },
        timeout=30,
    )

    if not resp.ok:
        raise HTTPException(status_code=502, detail=f"TTS error: {resp.text}")

    from fastapi.responses import Response
    return Response(content=resp.content, media_type="audio/mpeg")

@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    """transcribe audio using NVIDIA NIM Nemotron ASR Streaming via gRPC"""
    if not NIM_API_KEY:
        raise HTTPException(status_code=500, detail="NIM_API_KEY not set in .env")

    import riva.client
    import tempfile
    import subprocess
    import wave
    from copy import deepcopy

    audio_content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
        tmp_in.write(audio_content)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path.replace(".webm", ".wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp_wav_path],
            capture_output=True,
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"ffmpeg error: {result.stderr.decode()}")

        auth = riva.client.Auth(
            uri="grpc.nvcf.nvidia.com:443",
            use_ssl=True,
            metadata_args=[
                ["function-id", "bb0837de-8c7b-481f-9ec8-ef5663e9c1fa"],
                ["authorization", f"Bearer {NIM_API_KEY}"],
            ],
        )

        asr_service = riva.client.ASRService(auth)

        with wave.open(tmp_wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            raw_audio = wf.readframes(wf.getnframes())

        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            max_alternatives=1,
            enable_automatic_punctuation=True,
            audio_channel_count=n_channels,
        )

        streaming_config = riva.client.StreamingRecognitionConfig(
            config=deepcopy(config),
            interim_results=False,
        )

        chunk_size = 4800
        chunks = [raw_audio[i:i+chunk_size] for i in range(0, len(raw_audio), chunk_size)]
        if not chunks:
            raise HTTPException(status_code=400, detail="No audio data in file")

        responses = asr_service.streaming_response_generator(iter(chunks), streaming_config)

        transcript = ""
        for response in responses:
            for result in response.results:
                if result.alternatives:
                    transcript = result.alternatives[0].transcript
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT error: {str(e)}")
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)

    return {"text": transcript}

@app.post("/chat")
def chat(request: QuestionRequest):
    user = request.user.strip()
    if not user:
        raise HTTPException(status_code=400, detail="User name is required")

    faiss_index, chunks, metadata = get_user_index(user)
    if faiss_index is None:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")

    question_embedding = get_embeddings([request.question])
    faiss.normalize_L2(question_embedding)

    k = min(5, len(chunks))
    distances, indices = faiss_index.search(question_embedding, k)

    retrieved = []
    sources = []
    for idx in indices[0]:
        if idx < len(chunks):
            retrieved.append(chunks[idx])
            sources.append(metadata[idx])

    context = "\n\n---\n\n".join(retrieved)  # joins the 5 chunks into one big context

    prompt = f"""You are a helpful assistant. Answer the user's question based ONLY on the context below.
If the answer is not in the context, say "I couldn't find that in the uploaded documents."

CONTEXT:
{context}

QUESTION:
{request.question}

ANSWER:"""

    answer = get_chat_answer(prompt)

    # STEP 6 — re-evaluation layer (faithfulness, relevance, context sufficiency)
    eval_prompt = f"""You are a strict RAG evaluator. Given the context, question and answer below,
rate each metric from 0.0 to 1.0 and give a verdict.

CONTEXT:
{context}

QUESTION:
{request.question}

ANSWER:
{answer}

Evaluate these 3 metrics:
1. faithfulness: Is the answer grounded in the context? (1.0 = fully grounded, 0.0 = made up)
2. relevance: Does the answer address the question? (1.0 = fully relevant, 0.0 = irrelevant)
3. context_sufficiency: Did the context contain enough info to answer? (1.0 = sufficient, 0.0 = insufficient)

Respond ONLY with valid JSON, no explanation, no markdown:
{{"faithfulness": 0.0, "relevance": 0.0, "context_sufficiency": 0.0, "verdict": "pass or fail"}}

verdict is "pass" if ALL scores >= 0.5, otherwise "fail"."""

    try:
        eval_answer = get_chat_answer(eval_prompt)
        eval_text = eval_answer.strip().replace("```json", "").replace("```", "").strip()

        import json
        eval_result = json.loads(eval_text)

        faithfulness = eval_result.get("faithfulness", 0)
        relevance = eval_result.get("relevance", 0)
        context_sufficiency = eval_result.get("context_sufficiency", 0)
        verdict = eval_result.get("verdict", "fail")

        # if any metric fails threshold → reject answer
        if verdict == "fail" or faithfulness < 0.5 or relevance < 0.5:
            answer = "I couldn't find a reliable answer in the uploaded documents."

        return {
            "answer": answer,
            "sources": sources,
            "evaluation": {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "context_sufficiency": context_sufficiency,
                "verdict": verdict
            }
        }

    except Exception as e:
        print(f"Evaluation error: {e}")
        return {"answer": answer, "sources": sources}