from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import sys

# Wrap imports with error handling to prevent startup crash
try:
    from pdf_processor import extract_chapters, build_faiss_index, load_index
    from context_pruner import prune_context, build_prompt, calculate_tokens_saved
    from tutor_engine import get_answer
except Exception as e:
    print(f"Warning: Import error (app will still start): {e}", file=sys.stderr)
    extract_chapters = None
    build_faiss_index = None
    load_index = None
    prune_context = None
    build_prompt = None
    calculate_tokens_saved = None
    get_answer = None

app = FastAPI(title="Education Tutor India API")

@app.on_event("startup")
async def startup_event():
    print("✓ Shiksha API starting up...", file=sys.stderr)
    print(f"  Imports available: extract_chapters={bool(extract_chapters)}, get_answer={bool(get_answer)}", file=sys.stderr)

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
    return {"status": "Education Tutor API Running", "ready": all([extract_chapters, build_faiss_index, get_answer])}

@app.post("/upload")
async def upload_textbook(file: UploadFile = File(...)):
    if not extract_chapters or not build_faiss_index:
        raise HTTPException(status_code=503, detail="PDF processing not available. Please try again later.")
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
    try:
        if not get_answer or not prune_context or not load_index:
            raise HTTPException(status_code=503, detail="AI service not available.")

        # ✅ session check
        if req.session_id not in sessions:
            index_path = os.path.join(INDEX_DIR, f"{req.session_id}.faiss")
            meta_path = os.path.join(INDEX_DIR, f"{req.session_id}.pkl")

            if os.path.exists(index_path) and os.path.exists(meta_path):
                sessions[req.session_id] = {
                    "index_path": index_path,
                    "meta_path": meta_path
                }
            else:
                return {"answer": "Session not found. Please upload again."}

        session = sessions[req.session_id]

        index, chapters = load_index(session["index_path"], session["meta_path"])

        # ✅ SAFE CHECK 1
        if not chapters:
            return {"answer": "No content found in uploaded PDF."}

        pruned = prune_context(req.question, index, chapters)

        # ✅ SAFE CHECK 2 (MOST IMPORTANT)
        if not pruned:
            return {
                "answer": "I couldn't find relevant content. Try another question 😊",
                "context_stats": {
                    "reduction_percent": 0,
                    "chapters_used": 0,
                    "total_chapters": len(chapters)
                },
                "tokens_used": {
                    "total_tokens": 0
                },
                "chapters_referenced": []
            }

        stats = calculate_tokens_saved(chapters, pruned)
        prompt = build_prompt(req.question, pruned)

        result = get_answer(prompt)

        return {
            "answer": result["answer"],
            "context_stats": stats,
            "tokens_used": result["tokens_used"],
            "chapters_referenced": [ch["title"] for ch in pruned]
        }

    except Exception as e:
        print("ERROR in /ask:", str(e))
        return {
            "answer": "Something went wrong. Please try again later.",
            "error": str(e)
        }

@app.get("/sessions")
def list_sessions():
    return {"sessions": list(sessions.keys())}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
