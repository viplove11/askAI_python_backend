import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.core.admin_logic import build_and_save_index
from app.core.chat_logic import DatabaseChatbot

app = FastAPI(title="Ask Your Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Sarvekshan Sahayak AI Assistant Root API is running 🚀"}


@app.post("/admin/generate-brain")
async def generate_brain():
    try:
        db_url = os.getenv("DATABASE_URL")
        out_dir = os.getenv("STORAGE_PATH", "./store/current_db")
        os.makedirs(out_dir, exist_ok=True)
        build_and_save_index(db_url, out_dir)
        return {"status": "success", "message": "Brain generated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_with_db(request: ChatRequest):
    try:
        bot = DatabaseChatbot()
        result = bot.ask(request.question)
        
        # 1. Detect Report Intent
        q = request.question.lower()
        report_keywords = ["report", "excel", "download", "chahiye", "list", "records"]
        
        if any(word in q for word in report_keywords) and result.get("data"):
            # 2. Setup directory and paths
            report_dir = "temp_reports"
            os.makedirs(report_dir, exist_ok=True)
            
            df = pd.DataFrame(result["data"])
            file_name = f"report_{datetime.now().strftime('%H%M%S')}.xlsx"
            file_path = os.path.join(report_dir, file_name)
            
            # 3. Generate the actual file
            df.to_excel(file_path, index=False)
            
            # 4. Get Domain from ENV (fallback to localhost if not set)
            backend_base = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")
            
            # 5. Construct Absolute Download URL
            download_endpoint = f"/chat/download-report/{file_name}"
            full_download_url = f"{backend_base}{download_endpoint}"
            
            # 6. Update the AI response
            result["answer"] += f"\n\n📊 **Excel Report Generated:** [Download Here]({full_download_url})"
            result["report_url"] = full_download_url
            result["intent"] = "report"
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        
@app.get("/chat/download-report/{file_name}")
async def download_report(file_name: str):
    file_path = os.path.join("temp_reports", file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=file_name)
    raise HTTPException(status_code=404, detail="File not found")

@app.on_event("startup")
async def startup():
    os.makedirs("temp_reports", exist_ok=True)


# Command for running project 
# python -m uvicorn app.main:app --reload --port 8080