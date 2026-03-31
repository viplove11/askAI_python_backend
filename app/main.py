import os
import pandas as pd
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# Lifespan: replaces deprecated @app.on_event
# Port binds FIRST, then startup tasks run.
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────
    os.makedirs("temp_reports", exist_ok=True)
    print("[INFO] temp_reports directory ready.")
    # Heavy models (SentenceTransformer, DB engine) are lazy-loaded on first
    # request — no blocking work here so the port binds immediately.
    yield
    # ── Shutdown ─────────────────────────────
    # Add any cleanup here if needed (e.g. close DB connections).
    print("[INFO] Shutting down.")


app = FastAPI(title="Sarvekshan Sahayak AI Assistant", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Sarvekshan Sahayak AI Assistant Root API is running 🚀"}


@app.post("/admin/generate-brain")
async def generate_brain():
    """Re-index the connected database and rebuild the FAISS vector store."""
    try:
        from app.core.admin_logic import build_and_save_index

        db_url = os.getenv("DATABASE_URL")
        out_dir = os.getenv("STORAGE_PATH", "./store/current_db")
        os.makedirs(out_dir, exist_ok=True)
        build_and_save_index(db_url, out_dir)
        return {"status": "success", "message": "Brain generated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_db(request: ChatRequest):
    """Main chat endpoint — runs NL → SQL → formatted answer pipeline."""
    try:
        from app.core.chat_logic import DatabaseChatbot

        bot = DatabaseChatbot()
        result = bot.ask(request.question)

        # ── Optional Excel report generation ─────────────────────────────────
        q = request.question.lower()
        report_keywords = ["report", "excel", "download", "chahiye", "list", "records"]

        if any(word in q for word in report_keywords) and result.get("data"):
            report_dir = "temp_reports"
            os.makedirs(report_dir, exist_ok=True)

            df = pd.DataFrame(result["data"])
            file_name = f"report_{datetime.now().strftime('%H%M%S')}.xlsx"
            file_path = os.path.join(report_dir, file_name)
            df.to_excel(file_path, index=False)

            backend_base = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")
            download_endpoint = f"/chat/download-report/{file_name}"
            full_download_url = f"{backend_base}{download_endpoint}"

            result["answer"] += (
                f"\n\n📊 **Excel Report Generated:** [Download Here]({full_download_url})"
            )
            result["report_url"] = full_download_url
            result["intent"] = "report"

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/download-report/{file_name}")
async def download_report(file_name: str):
    """Serve a previously generated Excel report file."""
    # Basic path traversal guard
    if ".." in file_name or "/" in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    file_path = os.path.join("temp_reports", file_name)
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            filename=file_name,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    raise HTTPException(status_code=404, detail="File not found.")


# ─────────────────────────────────────────────
# Run locally:
#   python -m uvicorn app.main:app --reload --port 8080
# ─────────────────────────────────────────────
