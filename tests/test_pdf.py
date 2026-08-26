"""Tests for PDF validation and text extraction."""

import pytest

from app.services.pdf_service import PDFProcessingError, sanitize_filename, validate_pdf_bytes


class TestSanitizeFilename:

    def test_basic_name(self):
        assert sanitize_filename("employee_handbook.pdf") == "employee_handbook.pdf"

    def test_spaces_become_underscores(self):
        result = sanitize_filename("My Document.pdf")
        assert " " not in result
        assert "My" in result

    def test_path_separators_removed(self):
        result = sanitize_filename("../../etc/passwd.pdf")
        assert "/" not in result
        assert "\\" not in result

    def test_leading_dot_stripped(self):
        result = sanitize_filename(".hidden.pdf")
        assert not result.startswith(".")

    def test_empty_name_gets_default(self):
        assert sanitize_filename("") == "document.pdf"


class TestValidatePdfBytes:

    def test_empty_bytes_raises(self):
        with pytest.raises(PDFProcessingError, match="empty"):
            validate_pdf_bytes(b"", "test.pdf")

    def test_non_pdf_magic_raises(self):
        with pytest.raises(PDFProcessingError, match="valid PDF"):
            validate_pdf_bytes(b"PK\x03\x04fake-zip", "test.pdf")

    def test_valid_magic_passes(self):
        # should not raise - has the %PDF magic bytes
        validate_pdf_bytes(b"%PDF-1.4 fake content", "test.pdf")

    def test_oversized_file_raises(self, monkeypatch):
        import app.services.pdf_service as svc
        monkeypatch.setattr(svc, "_MAX_BYTES", 10)
        with pytest.raises(PDFProcessingError, match="limit"):
            validate_pdf_bytes(b"%PDF" + b"x" * 20, "big.pdf")


class TestExtractTextFromPdf:

    @pytest.fixture
    def sample_pdf(self, tmp_path):
        """Create a small test PDF with known text."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 100), "Employees must change their password every 90 days.")
        page.insert_text((50, 120), "This is the second sentence on page 1.")
        pdf_path = tmp_path / "sample.pdf"
        doc.save(str(pdf_path))
        doc.close()
        return pdf_path

    def test_extracts_text(self, sample_pdf):
        from app.services.pdf_service import extract_text_from_pdf
        pages = extract_text_from_pdf(sample_pdf)
        assert len(pages) >= 1
        full_text = " ".join(p["text"] for p in pages)
        assert "password" in full_text.lower()

    def test_page_numbers_are_one_based(self, sample_pdf):
        from app.services.pdf_service import extract_text_from_pdf
        pages = extract_text_from_pdf(sample_pdf)
        assert pages[0]["page"] == 1

    def test_document_name_preserved(self, sample_pdf):
        from app.services.pdf_service import extract_text_from_pdf
        pages = extract_text_from_pdf(sample_pdf)
        assert pages[0]["document"] == sample_pdf.name

    def test_nonexistent_file_raises(self, tmp_path):
        from app.services.pdf_service import extract_text_from_pdf
        with pytest.raises(PDFProcessingError):
            extract_text_from_pdf(tmp_path / "does_not_exist.pdf")
