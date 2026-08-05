"""FastAPI application entry point for MedAssist RAG."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from api.routes import router
from config import settings
from rag.chain import RAGChain
from vectorstore.chroma_store import ChromaStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the vector store and RAG chain once at startup."""
    logger.info("Initializing ChromaDB store and embedding model...")
    store = ChromaStore()
    app.state.chroma_store = store
    app.state.rag_chain = RAGChain(store=store)
    logger.info("Startup complete. Collection stats: %s", store.get_collection_stats())
    yield
    logger.info("Shutting down MedAssist RAG API")


app = FastAPI(
    title="MedAssist RAG API",
    description="Medical Document Q&A System using Retrieval-Augmented Generation over openFDA drug labels.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/")
def root():
    return {"service": "MedAssist RAG API", "status": "running", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
