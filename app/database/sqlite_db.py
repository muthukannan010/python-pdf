"""SQLite database layer with FTS5 for keyword search."""

import logging
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.utils.config import settings

logger = logging.getLogger(__name__)


class SQLiteDB:
    """Manages the SQLite database - stores documents, chunks, and handles keyword search."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.SQLITE_DB_PATH
        self._init_tables()

    def _init_tables(self):
        """Create the tables if they don't exist yet."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id   TEXT PRIMARY KEY,
                    document_name TEXT NOT NULL,
                    page_count    INTEGER NOT NULL DEFAULT 0,
                    chunk_count   INTEGER NOT NULL DEFAULT 0,
                    source_path   TEXT NOT NULL,
                    uploaded_at   TEXT NOT NULL
                )
            """)

            # FTS5 virtual table for full-text search with porter stemming
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                    chunk_id,
                    document_id,
                    document_name,
                    page_number,
                    text,
                    source_path,
                    content_rowid=rowid,
                    tokenize='porter unicode61'
                )
            """)

            # regular table for exact lookups by chunk_id
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_meta (
                    chunk_id      TEXT PRIMARY KEY,
                    document_id   TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    page_number   INTEGER NOT NULL,
                    text          TEXT NOT NULL,
                    source_path   TEXT NOT NULL,
                    locations     TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                )
            """)
            conn.commit()
        logger.info("SQLite database initialised at %s", self.db_path)

    @contextmanager
    def _connect(self):
        """Get a database connection (auto-closes)."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # --- Document operations ---

    def insert_document(self, document_id, document_name, page_count, chunk_count, source_path):
        uploaded_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents
                    (document_id, document_name, page_count, chunk_count, source_path, uploaded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, document_name, page_count, chunk_count, source_path, uploaded_at),
            )
            conn.commit()
        logger.debug("Inserted document '%s'", document_name)

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return dict(row) if row else None

    def document_exists(self, document_id: str) -> bool:
        return self.get_document(document_id) is not None

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks. Returns True if it existed."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM chunk_meta WHERE document_id = ?", (document_id,))
            result = conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            deleted = result.rowcount > 0
        if deleted:
            logger.info("Deleted document '%s' from SQLite", document_id)
        return deleted

    def get_document_count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    # --- Chunk operations ---

    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """Insert chunks into both the FTS table and the metadata table."""
        fts_rows = [
            (c["chunk_id"], c["document_id"], c["document_name"],
             str(c["page_number"]), c["text"], c["source_path"])
            for c in chunks
        ]
        meta_rows = [
            (c["chunk_id"], c["document_id"], c["document_name"],
             c["page_number"], c["text"], c["source_path"],
             json.dumps(c.get("locations", [])))
            for c in chunks
        ]
        with self._connect() as conn:
            conn.executemany(
                """INSERT INTO chunks
                   (chunk_id, document_id, document_name, page_number, text, source_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                fts_rows,
            )
            conn.executemany(
                """INSERT OR REPLACE INTO chunk_meta
                   (chunk_id, document_id, document_name, page_number, text, source_path, locations)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                meta_rows,
            )
            conn.commit()
        logger.debug(
            "Inserted %d chunks for document '%s'",
            len(chunks), chunks[0]["document_id"] if chunks else "?"
        )

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunk_meta WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        if row:
            res = dict(row)
            res["locations"] = json.loads(res["locations"]) if res.get("locations") else []
            return res
        return None

    # --- Keyword search ---

    def keyword_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Run an FTS5 query and return matching chunks with metadata."""
        if not query.strip():
            return []

        sql = """
            SELECT
                c.chunk_id, c.document_id, c.document_name,
                CAST(c.page_number AS INTEGER) AS page_number,
                c.text, c.source_path, m.locations
            FROM chunks c
            JOIN chunk_meta m ON c.chunk_id = m.chunk_id
            WHERE c.chunks MATCH ?
            LIMIT ?
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(sql, (query, top_k)).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("Keyword search error (query=%r): %s", query, exc)
            return []

        results = []
        for r in rows:
            d = dict(r)
            d["locations"] = json.loads(d["locations"]) if d.get("locations") else []
            results.append(d)
            
        logger.debug("Keyword search returned %d results for query=%r", len(results), query)
        return results
