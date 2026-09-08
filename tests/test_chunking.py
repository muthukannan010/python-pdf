"""Tests for text chunking logic."""

import unittest
from app.services.chunk_service import chunk_page_text, create_chunks_from_pages


class TestChunkPageText(unittest.TestCase):

    def _make_words_list(self, n):
        return [{"text": f"word{i}", "bbox": [0, 0, 10, 10]} for i in range(n)]

    def test_short_text_produces_one_chunk(self):
        words_list = self._make_words_list(50)
        chunks = chunk_page_text(
            words_list=words_list, document_id="doc1", document_name="doc1.pdf",
            page_number=1, source_path="data/pdfs/doc1.pdf",
            chunk_size=600, chunk_overlap=100,
        )
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self):
        words_list = self._make_words_list(1500)
        chunks = chunk_page_text(
            words_list=words_list, document_id="doc1", document_name="doc1.pdf",
            page_number=1, source_path="data/pdfs/doc1.pdf",
            chunk_size=600, chunk_overlap=100,
        )
        assert len(chunks) > 1

    def test_chunk_word_count_does_not_exceed_size(self):
        words_list = self._make_words_list(2000)
        chunks = chunk_page_text(
            words_list=words_list, document_id="doc1", document_name="doc1.pdf",
            page_number=1, source_path="data/pdfs/doc1.pdf",
            chunk_size=600, chunk_overlap=100,
        )
        for chunk in chunks:
            wc = len(chunk["text"].split())
            assert wc <= 600, f"Chunk has {wc} words, expected <= 600"

    def test_chunk_ids_are_unique(self):
        words_list = self._make_words_list(2000)
        chunks = chunk_page_text(
            words_list=words_list, document_id="doc1", document_name="doc1.pdf",
            page_number=1, source_path="data/pdfs/doc1.pdf",
            chunk_size=600, chunk_overlap=100,
        )
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_metadata_fields_present(self):
        words_list = self._make_words_list(100)
        chunks = chunk_page_text(
            words_list=words_list, document_id="my_doc", document_name="my_doc.pdf",
            page_number=5, source_path="data/pdfs/my_doc.pdf",
        )
        expected_keys = {"chunk_id", "document_id", "document_name",
                         "page_number", "text", "source_path", "locations"}
        for chunk in chunks:
            missing = expected_keys - chunk.keys()
            assert not missing, f"Missing keys: {missing}"

    def test_page_number_preserved_in_chunks(self):
        words_list = self._make_words_list(100)
        chunks = chunk_page_text(
            words_list=words_list, document_id="doc1", document_name="doc1.pdf",
            page_number=12, source_path="data/pdfs/doc1.pdf",
        )
        for chunk in chunks:
            assert chunk["page_number"] == 12

    def test_overlap_creates_shared_words(self):
        """Adjacent chunks should share some words when overlap > 0."""
        words_list = self._make_words_list(700)
        chunks = chunk_page_text(
            words_list=words_list, document_id="doc1", document_name="doc1.pdf",
            page_number=1, source_path="data/pdfs/doc1.pdf",
            chunk_size=400, chunk_overlap=100,
        )
        if len(chunks) < 2:
            raise unittest.SkipTest("Need at least 2 chunks")

        words_0 = set(chunks[0]["text"].split())
        words_1 = set(chunks[1]["text"].split())
        assert len(words_0 & words_1) > 0, "No shared words between adjacent chunks"


class TestCreateChunksFromPages(unittest.TestCase):

    def _make_pages(self, n_pages, words_per_page=200):
        return [
            {
                "document": "test.pdf",
                "page": i + 1,
                "text": " ".join(f"word{i}_{j}" for j in range(words_per_page)),
                "words": [{"text": f"word{i}_{j}", "bbox": [0, 0, 10, 10]} for j in range(words_per_page)]
            }
            for i in range(n_pages)
        ]

    def test_multi_page_document_produces_chunks(self):
        pages = self._make_pages(3, 300)
        chunks = create_chunks_from_pages(
            pages=pages, document_id="test",
            document_name="test.pdf", source_path="data/pdfs/test.pdf",
        )
        assert len(chunks) > 0

    def test_all_chunk_ids_are_unique_across_pages(self):
        pages = self._make_pages(5, 700)
        chunks = create_chunks_from_pages(
            pages=pages, document_id="test",
            document_name="test.pdf", source_path="data/pdfs/test.pdf",
        )
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Found duplicate chunk IDs across pages"

    def test_empty_pages_skipped(self):
        pages = [
            {"document": "test.pdf", "page": 1, "text": "", "words": []},
            {"document": "test.pdf", "page": 2, "text": "   ", "words": []},
            {"document": "test.pdf", "page": 3, "text": "Real content here.", "words": [{"text": "Real", "bbox": [0,0,10,10]}, {"text": "content", "bbox": [0,0,10,10]}, {"text": "here.", "bbox": [0,0,10,10]}]},
        ]
        chunks = create_chunks_from_pages(
            pages=pages, document_id="test",
            document_name="test.pdf", source_path="data/pdfs/test.pdf",
        )
        # only page 3 has actual content
        assert all(c["page_number"] == 3 for c in chunks)
