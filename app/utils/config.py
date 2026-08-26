from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PDF_DIR: Path = DATA_DIR / "pdfs"
    EXTRACTED_DIR: Path = DATA_DIR / "extracted"
    INDEX_DIR: Path = DATA_DIR / "index"

    # SQLite
    SQLITE_DB_PATH: Path = DATA_DIR / "search.db"

    # FAISS index files
    FAISS_INDEX_PATH: Path = INDEX_DIR / "faiss.index"
    FAISS_METADATA_PATH: Path = INDEX_DIR / "faiss_metadata.json"

    # Embedding model - runs locally, no API key needed
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Chunking params
    CHUNK_SIZE: int = 600       # words per chunk
    CHUNK_OVERLAP: int = 100    # overlap between chunks

    # Search defaults
    TOP_K: int = 5
    KEYWORD_WEIGHT: float = 0.4
    SEMANTIC_WEIGHT: float = 0.6
    MAX_FILE_SIZE_MB: int = 50
    SNIPPET_LENGTH: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self):
        """Create data directories if they don't exist yet."""
        for d in (self.PDF_DIR, self.EXTRACTED_DIR, self.INDEX_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
