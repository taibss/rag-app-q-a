# RAG Document Q&A App

## Setup & Run

### Backend
cd backend
conda activate rag-env
pip install -r requirements.txt
uvicorn main:app --reload

### Frontend
cd frontend
npm install
npm run dev

### Usage
1. Open http://localhost:5173
2. Upload a PDF
3. Ask questions about it!

## Tech Stack
- Frontend: React + Vite
- Backend: FastAPI
- Vector DB: FAISS
- Embeddings: SentenceTransformers
- LLM: Groq (llama-3.3-70b-versatile)
- PDF Processing: pdfplumber