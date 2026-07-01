from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload.upload import router as upload_router
from routes.chat.chat import router as chat_router
from models.embedding.emb_model import model_manager

import os
os.environ["PADDLEX_OFFLINE_MODE"] = "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the actual worker process starts (not the reloader)
    model_manager.load_models()
    yield
    # Shutdown cleanup can go here if needed


app = FastAPI(
    title="Universal RAG API",
    description="API for Retrieval Augmented Generation (RAG) application",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def home():
    return {
        "message": "Universal RAG Backend Server is running!",
        "endpoints": {
            "upload": "/api/upload",
            "chat": "/api/chat"
        }
    }

# Include routers
app.include_router(
    upload_router,
    prefix="/api",
    tags=["Upload"]
)

app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
