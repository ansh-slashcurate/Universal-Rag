from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload.upload import router as upload_router
from routes.chat.chat import router as chat_router
from models.embedding.emb_model import model_manager


app = FastAPI(
    title="Universal RAG API",
    description="API for Retrieval Augmented Generation (RAG) application",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager.load_models()



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
