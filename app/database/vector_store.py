"""FAISS vector store - stores embeddings and supports similarity search."""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.utils.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Local FAISS-backed vector store with a JSON file for metadata."""

    def __init__(
        self,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        dimension: int = settings.EMBEDDING_DIMENSION,
    ):
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self.metadata_path = metadata_path or settings.FAISS_METADATA_PATH
        self.dimension = dimension
        self._lock = Lock()

        # maps str(faiss_id) -> chunk metadata dict
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._next_id: int = 0

        self._index = self._load_or_create()

    def _load_or_create(self) -> faiss.Index:
        """Try loading existing index from disk, or create a fresh one."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._metadata = data.get("metadata", {})
                self._next_id = data.get("next_id", index.ntotal)
                logger.info(
                    "Loaded FAISS index (%d vectors) from %s",
                    index.ntotal, self.index_path,
                )
                return index
            except Exception as exc:
                logger.warning("Could not load FAISS index: %s — creating fresh.", exc)

        logger.info("Creating new FAISS IndexFlatIP (dim=%d)", self.dimension)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        return index

    def save(self):
        """Write index and metadata to disk."""
        with self._lock:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump({"next_id": self._next_id, "metadata": self._metadata},
                          f, ensure_ascii=False, indent=2)
        logger.debug("Saved FAISS index (%d vectors)", self._index.ntotal)

    def add_vectors(self, embeddings: np.ndarray, chunk_metas: List[Dict[str, Any]]):
        """Add embeddings and their metadata to the index."""
        if len(embeddings) != len(chunk_metas):
            raise ValueError("embeddings and chunk_metas must have the same length.")

        # normalize for cosine similarity (inner product on unit vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized = (embeddings / norms).astype(np.float32)

        with self._lock:
            ids = np.arange(self._next_id, self._next_id + len(embeddings), dtype=np.int64)
            self._index.add_with_ids(normalized, ids)
            for fid, meta in zip(ids.tolist(), chunk_metas):
                self._metadata[str(fid)] = meta
            self._next_id += len(embeddings)

        self.save()
        logger.debug("Added %d vectors to FAISS index", len(embeddings))

    def delete_by_document(self, document_id: str) -> int:
        """Remove all vectors for a given document. Returns count removed."""
        ids_to_remove = [
            int(fid) for fid, meta in self._metadata.items()
            if meta.get("document_id") == document_id
        ]
        if not ids_to_remove:
            return 0

        id_selector = faiss.IDSelectorBatch(
            len(ids_to_remove),
            faiss.swig_ptr(np.array(ids_to_remove, dtype=np.int64)),
        )
        with self._lock:
            removed = self._index.remove_ids(id_selector)
            for fid in ids_to_remove:
                self._metadata.pop(str(fid), None)

        self.save()
        logger.info("Removed %d vectors for document '%s'", removed, document_id)
        return removed

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find the top-k most similar chunks to the query vector."""
        if self._index.ntotal == 0:
            logger.debug("FAISS index is empty — no results.")
            return []

        # normalize query vector
        q = query_embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        actual_k = min(top_k, self._index.ntotal)
        distances, faiss_ids = self._index.search(q, actual_k)

        results = []
        for dist, fid in zip(distances[0], faiss_ids[0]):
            if fid < 0:
                continue
            meta = self._metadata.get(str(fid))
            if meta is None:
                continue
            results.append({
                **meta,
                "semantic_score": float(np.clip(dist, 0.0, 1.0)),
            })

        logger.debug("FAISS search returned %d results", len(results))
        return results

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal
