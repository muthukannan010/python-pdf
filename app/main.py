import logging
import sys
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.database.sqlite_db import SQLiteDB
from app.database.vector_store import VectorStore
from app.services.embedding_service import get_model
from app.services.hybrid_search import HybridSearchService
from app.services.keyword_search import KeywordSearchService
from app.services.semantic_search import SemanticSearchService
from app.utils.config import settings

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Set up all the services when the app starts, clean up on shutdown."""
    logger.info("=" * 60)
    logger.info("  PDF Document Search Engine - Starting up")
    logger.info("=" * 60)

    # make sure data folders exist
    settings.ensure_directories()
    logger.info("Data directories ready.")

    # init database
    db = SQLiteDB()
    app.state.db = db

    # init vector store
    vs = VectorStore()
    app.state.vector_store = vs

    # load embedding model into memory (first time downloads from HuggingFace)
    logger.info("Warming up embedding model ...")
    get_model()
    logger.info("Embedding model ready.")

    # wire up the search services
    kw_service = KeywordSearchService(db=db)
    sem_service = SemanticSearchService(vector_store=vs)
    hybrid_service = HybridSearchService(
        keyword_service=kw_service,
        semantic_service=sem_service,
    )
    app.state.hybrid_service = hybrid_service

    logger.info("Application startup complete.")
    logger.info("Open your browser at http://localhost:%d", settings.PORT)
    logger.info("=" * 60)

    # try to open browser automatically
    try:
        webbrowser.open(f"http://localhost:{settings.PORT}")
    except Exception as e:
        logger.warning(f"Could not open browser automatically: {e}")

    yield  # app runs here

    logger.info("Application shutting down …")


# create the FastAPI app
app = FastAPI(
    title="PDF Document Search Engine",
    description=(
        "On-premises PDF search with hybrid keyword + semantic search. "
        "No external APIs or cloud services required."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS - allow everything for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# register API routes
app.include_router(router, prefix="/api/v1", tags=["PDF Search"])

# serve the frontend
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
else:
    logger.warning("Frontend directory not found at %s", _FRONTEND_DIR)


# catch-all error handler so we never leak tracebacks to the client
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again."},
    )
