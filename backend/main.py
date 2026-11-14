import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model_service import ModelNotReady, ModelService
from schemas import GenerateRequest, GenerateResponse, ModelStatus

load_dotenv()

service = ModelService(
    model_name=os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B"),
    dtype_name=os.getenv("DTYPE", "float16"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load in the background so /api/health works while weights download/load.
    threading.Thread(target=service.load, daemon=True).start()
    yield


app = FastAPI(title="BHARAT-AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_state": service.state}


@app.get("/api/model/status", response_model=ModelStatus)
def model_status():
    return service.status()


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        return service.generate(req.prompt, req.max_new_tokens)
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")