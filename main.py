from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil

from pdf_processor import extract_chapters, build_faiss_index, load_index
from context_pruner import prune_context, build_prompt, calculate_tokens_saved
from tutor_engine import get_answer

app = FastAPI(title="Education Tutor India API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_DIR = "uploads"
INDEX_DIR = "indexes"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

sessions = {}

class QuestionRequest(BaseModel):
    session_id: str
    question: str

@app.get("/")
def root():
    return {"status": "Education Tutor API Running"}

@app.post("/upload")
async def upload_textbook(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    session_id = file.filename.replace(".pdf", "").replace(" ", "_")
    pdf_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")
    index_path = os.path.join(INDEX_DIR, f"{session_id}.faiss")
    meta_path = os.path.join(INDEX_DIR, f"{session_id}.pkl")

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chapters = extract_chapters(pdf_path)
    num_chapters = build_faiss_index(chapters, index_path, meta_path)

    sessions[session_id] = {
        "index_path": index_path,
        "meta_path": meta_path,
        "total_chapters": num_chapters,
        "filename": file.filename
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "chapters_found": num_chapters,
        "message": "Textbook processed successfully"
    }

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    if req.session_id not in sessions:
        index_path = os.path.join(INDEX_DIR, f"{req.session_id}.faiss")
        meta_path = os.path.join(INDEX_DIR, f"{req.session_id}.pkl")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            sessions[req.session_id] = {
                "index_path": index_path,
                "meta_path": meta_path
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found. Please upload a textbook first.")

    session = sessions[req.session_id]
    index, chapters = load_index(session["index_path"], session["meta_path"])

    pruned = prune_context(req.question, index, chapters)
    stats = calculate_tokens_saved(chapters, pruned)
    prompt = build_prompt(req.question, pruned)

    result = get_answer(prompt)

    return {
        "answer": result["answer"],
        "context_stats": stats,
        "tokens_used": result["tokens_used"],
        "chapters_referenced": [ch["title"] for ch in pruned]
    }

@app.get("/sessions")
def list_sessions():
    return {"sessions": list(sessions.keys())}
