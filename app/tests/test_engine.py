"""
DVS - Unit Tests
Run with: pytest tests/
"""
import pytest
from app.core.engine import (
    classify_document,
    extract_fields,
    fuzzy_name_match,
    match_documents,
    process_document,
)
from app.models.schemas import DocumentType, VerificationStatus


# ── classify_document ──────────────────────────────────────────────────────

def test_classify_passport():
    doc_type, conf = classify_document("passport_john.pdf", 100)
    assert doc_type == DocumentType.PASSPORT
    assert 0.0 < conf <= 1.0


def test_classify_driving_license():
    doc_type, conf = classify_document("driving_license.jpg", 200)
    assert doc_type == DocumentType.DRIVERS_LICENSE


def test_classify_unknown_fallback():
    doc_type, conf = classify_document("random_file.pdf", 1)
    assert isinstance(doc_type, DocumentType)
    assert conf > 0


# ── extract_fields ─────────────────────────────────────────────────────────

def test_extract_fields_passport():
    fields = extract_fields(DocumentType.PASSPORT)
    names  = [f.field_name for f in fields]
    assert "full_name" in names
    assert "date_of_birth" in names
    assert "passport_number" in names
    for f in fields:
        assert 0.0 <= f.confidence <= 1.0


def test_extract_fields_visa():
    fields = extract_fields(DocumentType.VISA)
    names  = [f.field_name for f in fields]
    assert "holder_name" in names
    assert "valid_until" in names


# ── fuzzy_name_match ───────────────────────────────────────────────────────

def test_fuzzy_exact_match():
    score = fuzzy_name_match("John Michael Doe", "JOHN MICHAEL DOE")
    assert score == 1.0


def test_fuzzy_partial_match():
    score = fuzzy_name_match("John Doe", "John Michael Doe")
    assert 0.4 < score < 1.0


def test_fuzzy_no_match():
    score = fuzzy_name_match("Alice Smith", "Bob Johnson")
    assert score < 0.30


def test_fuzzy_empty_strings():
    score = fuzzy_name_match("", "John Doe")
    assert score == 0.0


# ── process_document ───────────────────────────────────────────────────────

def test_process_passport():
    result = process_document("passport_scan.pdf", b"fake_pdf_bytes")
    assert result.document_type == DocumentType.PASSPORT
    assert result.has_face is True
    assert result.face_confidence is not None
    assert len(result.extracted_fields) > 0
    assert result.processing_time_ms >= 0
    assert result.status == VerificationStatus.VERIFIED


def test_process_utility_bill():
    result = process_document("utility_bill.png", b"fake_image")
    assert result.document_type == DocumentType.UTILITY_BILL
    assert result.has_face is False
    assert result.face_confidence is None


# ── match_documents ────────────────────────────────────────────────────────

def test_match_single_doc_returns_error():
    doc = process_document("passport.pdf", b"data")
    conf, scores, face, issues = match_documents([doc])
    assert conf == 0.0
    assert len(issues) > 0


def test_match_two_docs_returns_confidence():
    doc1 = process_document("passport_john.pdf",  b"data1")
    doc2 = process_document("driver_license.jpg", b"data2")
    conf, scores, face, issues = match_documents([doc1, doc2])
    assert 0.0 <= conf <= 1.0
    assert isinstance(scores, list)
    assert isinstance(issues, list)


def test_match_confidence_range():
    docs = [
        process_document("passport.pdf",    b"a"),
        process_document("driver_lic.jpg",  b"b"),
        process_document("national_id.png", b"c"),
    ]
    conf, _, _, _ = match_documents(docs)
    assert 0.0 <= conf <= 1.0