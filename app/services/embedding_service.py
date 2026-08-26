"""Handles loading the sentence-transformer model and encoding text."""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.utils.config import settings

logger = logging.getLogger(__name__)

# singleton - only load the model once
_model = None


def get_model() -> SentenceTransformer:
    """Load the model on first call, then reuse it."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' …", settings.MODEL_NAME)
        _model = SentenceTransformer(settings.MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def is_model_loaded() -> bool:
    return _model is not None


def encode_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of strings into embedding vectors (float32 numpy array)."""
    if not texts:
        return np.empty((0, settings.EMBEDDING_DIMENSION), dtype=np.float32)

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize in the vector store instead
    )
    logger.debug("Encoded %d texts → shape %s", len(texts), embeddings.shape)
    return embeddings.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """Encode a single query string into a 1-D vector."""
    result = encode_texts([query])
    return result[0]
