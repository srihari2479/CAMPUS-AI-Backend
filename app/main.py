import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api import rag_router, graph_router

# Load environment variables
load_dotenv()

app = FastAPI(
    title="College RAG Assistant API",
    description="Backend API for hybrid search, multimodal document extraction, self-correcting query loop, and knowledge graph queries.",
    version="1.0.0"
)

# CORS configuration to allow local & hosted React/Flutter frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(rag_router.router)
app.include_router(graph_router.router)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "College RAG Assistant API is running successfully.",
        "platform": "Hugging Face Spaces"
    }

@app.get("/health")
def health_check():
    # Simple endpoint for system monitoring
    return {"status": "healthy"}
