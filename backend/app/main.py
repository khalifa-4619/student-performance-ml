import os
import joblib
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core import config
from backend.app.api.routes import router as api_router

app = FastAPI(
    title="Student Performance Early Warning Engine",
    description="Production-grade modular MLOps infrastructure engine.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifecycle startup hooks to load the binary file globally into memory
@app.on_event("startup")
def load_ml_model():
    config.logger.info("⏳ Initializing System Lifespan: Mounting Serialized Pipeline Asset...")
    if not os.path.exists(config.MODEL_PATH):
        config.logger.error(f"❌ Critical Error: Missing pipeline binary at {config.MODEL_PATH}")
        raise RuntimeError("Model binary artifact missing.")
    try:
        config.model_pipeline = joblib.load(config.MODEL_PATH)
        config.logger.info("📦 Behavioral pipeline modularly mounted successfully into RAM.")
    except Exception as e:
        config.logger.critical(f"❌ Lifespan initialization crash: {str(e)}")
        raise e

# Mount your api router containing our domain routes under an automated API scope prefix
app.include_router(api_router, prefix="/api")

@app.get("/health", status_code=status.HTTP_200_OK)
async def system_health():
    return {"status": "healthy", "pipeline_active": config.model_pipeline is not None}