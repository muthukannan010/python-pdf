"""Tests for keyword, semantic, and hybrid search."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_sample_chunks(n=3):
    """Create a few sample chunks for testing."""
    texts = [
        "Employees must change their password every 90 days.",
        "The company provides health insurance for all full-time staff.",
        "Vacation policy allows 15 days of paid leave per year.",
    ]
    return [
        {
            "chunk_id": f"doc_page0001_chunk{i:03d}",
            "document_id": "doc",
            "document_name": "employee_handbook.pdf",
            "page_number": i + 1,
            "text": texts[i % len(texts)],
            "source_path": "data/pdfs/doc.pdf",
        }
        for i in range(n)
    ]


class TestKeywordSearch:

    @pytest.fixture
    def db(self, tmp_path):
        from app.database.sqlite_db import SQLiteDB
        db = SQLiteDB(db_path=tmp_path / "test.db")
        chunks = _make_sample_chunks(3)
        db.insert_document(
            document_id="doc", document_name="employee_handbook.pdf",
            page_count=3, chunk_count=3, source_path="data/pdfs/doc.pdf",
        )
        db.insert_chunks(chunks)
        return db

    def test_100_percent_keyword_match(self, db):
        from app.services.keyword_search import KeywordSearchService
        svc = KeywordSearchService(db=db)
        results = svc.search("password days", top_k=5)
        assert any(r["keyword_score"] == 100.0 for r in results)

    def test_partial_keyword_match(self, db):
        from app.services.keyword_search import KeywordSearchService
        svc = KeywordSearchService(db=db)
        # "password" is in chunk 0, "health" is in chunk 1
        results = svc.search("password health", top_k=5)
        assert len(results) >= 2
        for r in results:
            assert r["keyword_score"] == 50.0

    def test_0_percent_keyword_match(self, db):
        from app.services.keyword_search import KeywordSearchService
        svc = KeywordSearchService(db=db)
        results = svc.search("xyzzy nonexistent term abc", top_k=5)
        assert len(results) == 0

    def test_duplicate_query_keywords(self, db):
        from app.services.keyword_search import KeywordSearchService
        svc = KeywordSearchService(db=db)
        results = svc.search("password password", top_k=5)
        assert len(results) > 0
        assert results[0]["keyword_score"] == 100.0

    def test_stop_word_removal(self, db):
        from app.services.keyword_search import KeywordSearchService
        svc = KeywordSearchService(db=db)
        results = svc.search("the password of", top_k=5)
        assert len(results) > 0
        assert results[0]["keyword_score"] == 100.0
        assert results[0]["matched_keywords"] == ["password"]

    def test_case_insensitive_matching(self, db):
        from app.services.keyword_search import KeywordSearchService
        svc = KeywordSearchService(db=db)
        results = svc.search("PaSsWoRd", top_k=5)
        assert len(results) > 0
        assert results[0]["keyword_score"] == 100.0

    def test_keyword_highlighting(self, db):
        from app.services.keyword_search import KeywordSearchService
        svc = KeywordSearchService(db=db)
        results = svc.search("password", top_k=5)
        assert len(results) > 0
        assert "<mark>password</mark>" in results[0]["text"].lower()

    def test_empty_query_returns_empty(self, db):
        from app.services.keyword_search import KeywordSearchService
        svc = KeywordSearchService(db=db)
        assert svc.search("", top_k=5) == []


class TestSemanticSearch:

    @pytest.fixture
    def vector_store_with_data(self, tmp_path):
        from app.database.vector_store import VectorStore
        vs = VectorStore(
            index_path=tmp_path / "faiss.index",
            metadata_path=tmp_path / "faiss_meta.json",
            dimension=8,
        )
        chunks = _make_sample_chunks(3)
        embeddings = np.random.rand(3, 8).astype(np.float32)
        vs.add_vectors(embeddings=embeddings, chunk_metas=chunks)
        return vs

    def test_search_returns_results(self, vector_store_with_data):
        from app.services.semantic_search import SemanticSearchService
        mock_vec = np.random.rand(8).astype(np.float32)
        with patch("app.services.semantic_search.encode_query", return_value=mock_vec):
            svc = SemanticSearchService(vector_store=vector_store_with_data)
            results = svc.search("password policy", top_k=3)
        assert len(results) > 0

    def test_semantic_scores_scaled_to_100(self, vector_store_with_data):
        mock_vec = np.random.rand(8).astype(np.float32)
        with patch("app.services.semantic_search.encode_query", return_value=mock_vec):
            from app.services.semantic_search import SemanticSearchService
            svc = SemanticSearchService(vector_store=vector_store_with_data)
            results = svc.search("anything", top_k=3)
        for r in results:
            assert 0.0 <= r["semantic_score"] <= 100.0

    def test_empty_index_returns_empty(self, tmp_path):
        from app.database.vector_store import VectorStore
        from app.services.semantic_search import SemanticSearchService
        vs = VectorStore(
            index_path=tmp_path / "empty.index",
            metadata_path=tmp_path / "empty_meta.json",
            dimension=8,
        )
        mock_vec = np.random.rand(8).astype(np.float32)
        with patch("app.services.semantic_search.encode_query", return_value=mock_vec):
            svc = SemanticSearchService(vector_store=vs)
            results = svc.search("query", top_k=3)
        assert results == []


class TestHybridSearch:

    def _mock_kw(self, results):
        svc = MagicMock()
        svc.search.return_value = results
        return svc

    def _mock_sem(self, results):
        svc = MagicMock()
        svc.search.return_value = results
        return svc

    def test_hybrid_combines_both_results(self):
        from app.services.hybrid_search import HybridSearchService
        kw = [{"chunk_id": "c1", "document_name": "a.pdf", "page_number": 1,
               "text": "hello", "source_path": "data/pdfs/a.pdf", "keyword_score": 80.0,
               "matched_keywords": ["hello"], "unmatched_keywords": []}]
        sem = [{"chunk_id": "c2", "document_name": "b.pdf", "page_number": 2,
                "text": "world", "source_path": "data/pdfs/b.pdf", "semantic_score": 90.0}]

        svc = HybridSearchService(keyword_service=self._mock_kw(kw),
                                  semantic_service=self._mock_sem(sem))
        result = svc.search("hello world", top_k=5)
        assert result["total_results"] == 2
        ids = [r["chunk_id"] for r in result["results"]]
        assert "c1" in ids and "c2" in ids

    def test_hybrid_correct_score_calculation(self):
        from app.services.hybrid_search import HybridSearchService
        shared = {"chunk_id": "c1", "document_name": "a.pdf", "page_number": 1,
                  "text": "password policy", "source_path": "data/pdfs/a.pdf"}
        kw = [{**shared, "keyword_score": 100.0, "matched_keywords": ["password"],
               "unmatched_keywords": []}]
        sem = [{**shared, "semantic_score": 80.0}]

        svc = HybridSearchService(keyword_service=self._mock_kw(kw),
                                  semantic_service=self._mock_sem(sem))
        result = svc.search("password", top_k=5, keyword_weight=0.4, semantic_weight=0.6)

        c = result["results"][0]
        assert c["keyword_score"] == 100.0
        assert c["semantic_score"] == 80.0
        # 100*0.4 + 80*0.6 = 88.0
        assert abs(c["score"] - 88.0) < 1e-4

    def test_hybrid_correct_result_ranking(self):
        from app.services.hybrid_search import HybridSearchService
        kw = [
            {"chunk_id": "c1", "document_name": "a.pdf", "page_number": 1,
             "text": "text1", "source_path": "data/pdfs/a.pdf", "keyword_score": 100.0},
            {"chunk_id": "c2", "document_name": "b.pdf", "page_number": 2,
             "text": "text2", "source_path": "data/pdfs/b.pdf", "keyword_score": 50.0},
        ]
        sem = [
            {"chunk_id": "c1", "semantic_score": 80.0},
            {"chunk_id": "c2", "semantic_score": 80.0},
        ]

        svc = HybridSearchService(keyword_service=self._mock_kw(kw),
                                  semantic_service=self._mock_sem(sem))
        # with kw_weight=0, both get same semantic score, but c1 wins on tiebreaker
        result = svc.search("text", top_k=5, keyword_weight=0.0, semantic_weight=1.0)
        assert result["results"][0]["chunk_id"] == "c1"
        assert result["results"][1]["chunk_id"] == "c2"

    def test_hybrid_page_number_preservation(self):
        from app.services.hybrid_search import HybridSearchService
        kw = [{"chunk_id": "c1", "document_name": "a.pdf", "page_number": 15,
               "text": "text", "source_path": "data/pdfs/a.pdf", "keyword_score": 100.0}]

        svc = HybridSearchService(keyword_service=self._mock_kw(kw),
                                  semantic_service=self._mock_sem([]))
        result = svc.search("text", top_k=3)
        assert result["results"][0]["page_number"] == 15
