"""
DVS - Core Processing Engine
==============================
Handles:
  • Document classification  (ML-based in production → keyword heuristic here)
  • OCR field extraction     (Google Vision / Tesseract in production → simulated here)
  • Biometric face detection (face_recognition/dlib in production → simulated here)
  • Cross-document identity matching (fuzzy name, DOB, face score)

To wire in real OCR:
  1. pip install google-cloud-vision   → replace `extract_fields()` body
  2. pip install pytesseract easyocr   → fallback OCR
  3. pip install face-recognition      → replace face section
  4. pip install rapidfuzz spacy       → replace `fuzzy_name_match()`
"""

from __future__ import annotations

import random
import re
import time
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from app.models.schemas import (
    DocumentResult,
    DocumentType,
    ExtractedField,
    MatchScore,
    VerificationStatus,
)

# ── Document type → keywords for filename-based classification ─────────────
DOCUMENT_KEYWORDS: Dict[DocumentType, List[str]] = {
    DocumentType.PASSPORT:        ["passport", "travel"],
    DocumentType.DRIVERS_LICENSE: ["driver", "license", "licence", "driving", "dl"],
    DocumentType.NATIONAL_ID:     ["national", "aadhaar", "aadhar", "id_card", "nid"],
    DocumentType.VISA:            ["visa", "immigration", "consulate", "entry"],
    DocumentType.UTILITY_BILL:    ["utility", "electric", "water", "gas", "bill"],
    DocumentType.BANK_STATEMENT:  ["bank", "statement", "account", "passbook"],
    DocumentType.EMPLOYMENT_DOC:  ["employment", "payslip", "salary", "offer", "company"],
}

# ── Field templates per document type (simulates OCR output) ──────────────
FIELD_TEMPLATES: Dict[DocumentType, List[Dict]] = {
    DocumentType.PASSPORT: [
        {"field_name": "full_name",       "value": "JOHN MICHAEL DOE",                               "confidence": 0.97},
        {"field_name": "nationality",     "value": "IND",                                             "confidence": 0.96},
        {"field_name": "date_of_birth",   "value": "1990-03-15",                                      "confidence": 0.98},
        {"field_name": "expiry_date",     "value": "2030-03-14",                                      "confidence": 0.97},
        {"field_name": "passport_number", "value": "X1234567",                                        "confidence": 0.99},
        {"field_name": "mrz_line1",       "value": "P<INDDOE<<JOHN<MICHAEL<<<<<<<<<<<<<<<<<<<<",      "confidence": 0.95},
        {"field_name": "mrz_line2",       "value": "X12345671IND9003151M3003144<<<<<<<<<<<<<<<6",     "confidence": 0.95},
    ],
    DocumentType.DRIVERS_LICENSE: [
        {"field_name": "full_name",       "value": "John Michael Doe",                                "confidence": 0.95},
        {"field_name": "address",         "value": "123 Main Street, Ludhiana, Punjab 141001",        "confidence": 0.92},
        {"field_name": "date_of_birth",   "value": "1990-03-15",                                      "confidence": 0.96},
        {"field_name": "license_number",  "value": "PB0120190012345",                                 "confidence": 0.99},
        {"field_name": "expiry_date",     "value": "2030-03-14",                                      "confidence": 0.94},
        {"field_name": "vehicle_class",   "value": "LMV",                                             "confidence": 0.98},
    ],
    DocumentType.NATIONAL_ID: [
        {"field_name": "full_name",       "value": "John Michael Doe",                                "confidence": 0.96},
        {"field_name": "date_of_birth",   "value": "1990-03-15",                                      "confidence": 0.97},
        {"field_name": "id_number",       "value": "1234 5678 9012",                                  "confidence": 0.99},
        {"field_name": "address",         "value": "123 Main Street, Ludhiana, Punjab 141001",        "confidence": 0.91},
    ],
    DocumentType.VISA: [
        {"field_name": "holder_name",     "value": "JOHN MICHAEL DOE",                                "confidence": 0.94},
        {"field_name": "valid_from",      "value": "2024-01-01",                                      "confidence": 0.95},
        {"field_name": "valid_until",     "value": "2024-12-31",                                      "confidence": 0.95},
        {"field_name": "visa_type",       "value": "Tourist",                                         "confidence": 0.92},
        {"field_name": "entry_type",      "value": "Multiple Entry",                                  "confidence": 0.93},
        {"field_name": "issuing_country", "value": "USA",                                             "confidence": 0.99},
    ],
    DocumentType.UTILITY_BILL: [
        {"field_name": "account_name",    "value": "John M Doe",                                      "confidence": 0.90},
        {"field_name": "address",         "value": "123 Main Street, Ludhiana, Punjab 141001",        "confidence": 0.88},
        {"field_name": "account_number",  "value": "UB-9876543210",                                   "confidence": 0.98},
        {"field_name": "issue_date",      "value": "2024-11-01",                                      "confidence": 0.97},
        {"field_name": "due_date",        "value": "2024-11-25",                                      "confidence": 0.97},
        {"field_name": "amount_due",      "value": "2340.50",                                         "confidence": 0.99},
    ],
    DocumentType.BANK_STATEMENT: [
        {"field_name": "account_holder",  "value": "Mr. John Michael Doe",                            "confidence": 0.93},
        {"field_name": "account_number",  "value": "XXXX XXXX 7890",                                  "confidence": 0.99},
        {"field_name": "address",         "value": "123 Main Street, Ludhiana, Punjab 141001",        "confidence": 0.89},
        {"field_name": "statement_period","value": "01-Oct-2024 to 31-Oct-2024",                      "confidence": 0.96},
        {"field_name": "bank_name",       "value": "Punjab National Bank",                            "confidence": 0.99},
    ],
    DocumentType.EMPLOYMENT_DOC: [
        {"field_name": "employee_name",   "value": "John Michael Doe",                                "confidence": 0.94},
        {"field_name": "employer_name",   "value": "Acme Technologies Pvt. Ltd.",                     "confidence": 0.97},
        {"field_name": "designation",     "value": "Senior Software Engineer",                        "confidence": 0.96},
        {"field_name": "date_of_joining", "value": "2018-06-01",                                      "confidence": 0.95},
        {"field_name": "monthly_salary",  "value": "85000",                                           "confidence": 0.98},
    ],
}

# Document types that contain a face photo
FACE_DOC_TYPES = {
    DocumentType.PASSPORT,
    DocumentType.DRIVERS_LICENSE,
    DocumentType.NATIONAL_ID,
    DocumentType.VISA,
}

# Name field names per document type (for cross-matching)
NAME_FIELDS = {
    "full_name", "holder_name", "account_name",
    "account_holder", "employee_name",
}


# ── Classification ─────────────────────────────────────────────────────────

def classify_document(filename: str, file_size: int) -> Tuple[DocumentType, float]:
    """
    Classify document type from filename keywords.
    In production: replace with an ML image classifier.
    """
    fname = filename.lower()
    for doc_type, keywords in DOCUMENT_KEYWORDS.items():
        for kw in keywords:
            if kw in fname:
                return doc_type, round(random.uniform(0.88, 0.99), 3)

    # Fallback: cycle through types deterministically for demo
    types = list(FIELD_TEMPLATES.keys())
    chosen = types[file_size % len(types)]
    return chosen, round(random.uniform(0.70, 0.88), 3)


# ── OCR Field Extraction ───────────────────────────────────────────────────

def extract_fields(doc_type: DocumentType) -> List[ExtractedField]:
    """
    Extract structured fields from a document image.
    In production: call Google Vision API or Tesseract here.
    """
    template = FIELD_TEMPLATES.get(doc_type, [])
    fields: List[ExtractedField] = []
    for f in template:
        # Add small confidence variance to simulate real OCR
        conf = min(1.0, max(0.50, f["confidence"] + random.uniform(-0.04, 0.02)))
        fields.append(ExtractedField(
            field_name=f["field_name"],
            value=f["value"],
            confidence=round(conf, 3),
        ))
    return fields


# ── Name Matching ──────────────────────────────────────────────────────────

def fuzzy_name_match(name1: str, name2: str) -> float:
    """
    Token-based Jaccard similarity for name matching.
    In production: replace with rapidfuzz.token_sort_ratio()
    """
    def tokenize(n: str) -> set:
        return set(re.sub(r"[^a-zA-Z\s]", "", n).upper().split())

    t1, t2 = tokenize(name1), tokenize(name2)
    if not t1 or not t2:
        return 0.0
    return round(len(t1 & t2) / len(t1 | t2), 3)


def _get_name(doc: DocumentResult) -> Optional[str]:
    for f in doc.extracted_fields:
        if f.field_name in NAME_FIELDS and f.value:
            return f.value
    return None


def _get_dob(doc: DocumentResult) -> Optional[str]:
    for f in doc.extracted_fields:
        if "birth" in f.field_name and f.value:
            return f.value
    return None


# ── Cross-Document Matching ────────────────────────────────────────────────

def match_documents(
        docs: List[DocumentResult],
) -> Tuple[float, List[MatchScore], Optional[float], List[str]]:
    """
    Cross-reference all uploaded documents and produce:
      • overall_confidence  (0–1)
      • match_scores        (per field comparison)
      • face_match_score    (biometric check, if applicable)
      • issues              (list of human-readable warnings)
    """
    match_scores: List[MatchScore] = []
    issues: List[str] = []

    if len(docs) < 2:
        return 0.0, [], None, ["Need at least 2 documents for cross-verification."]

    # ── Name cross-match ───────────────────────────────────────────────────
    name_pairs = [(d.filename, _get_name(d)) for d in docs]
    valid_names = [(fn, n) for fn, n in name_pairs if n]

    if len(valid_names) >= 2:
        raw_scores: List[float] = []
        for i in range(len(valid_names)):
            for j in range(i + 1, len(valid_names)):
                fn_a, n_a = valid_names[i]
                fn_b, n_b = valid_names[j]
                s = fuzzy_name_match(n_a, n_b)
                raw_scores.append(s)
                match_scores.append(MatchScore(
                    field=f"Name  ({fn_a}  vs  {fn_b})",
                    score=s,
                    matched=s >= 0.70,
                ))
                if s < 0.70:
                    issues.append(
                        f"Name mismatch between '{fn_a}' and '{fn_b}' "
                        f"(similarity {s:.0%})."
                    )
        name_score = sum(raw_scores) / len(raw_scores)
    else:
        name_score = 0.50
        issues.append("Not enough documents with name fields for full name comparison.")

    # ── Date-of-birth cross-match ──────────────────────────────────────────
    dobs = [_get_dob(d) for d in docs]
    valid_dobs = [d for d in dobs if d]

    if len(valid_dobs) >= 2:
        all_same = all(d == valid_dobs[0] for d in valid_dobs)
        dob_score = 1.0 if all_same else 0.10
        match_scores.append(MatchScore(
            field="Date of Birth",
            score=dob_score,
            matched=all_same,
        ))
        if not all_same:
            issues.append("Date of birth does not match across documents.")
    else:
        dob_score = 0.60  # partial credit — not enough DOB data

    # ── Biometric face match ───────────────────────────────────────────────
    face_docs = [d for d in docs if d.has_face]
    face_score: Optional[float] = None

    if len(face_docs) >= 2:
        # In production: compare face encodings from face_recognition library
        face_score = round(random.uniform(0.80, 0.98), 3)
        match_scores.append(MatchScore(
            field="Biometric Face",
            score=face_score,
            matched=face_score >= 0.85,
        ))
        if face_score < 0.85:
            issues.append(
                f"Face similarity {face_score:.0%} is below the required 85% threshold."
            )

    # ── Expiry validation ──────────────────────────────────────────────────
    for doc in docs:
        for field in doc.extracted_fields:
            if "expiry" in field.field_name and field.value:
                try:
                    exp = datetime.strptime(field.value, "%Y-%m-%d").date()
                    if exp < date.today():
                        issues.append(
                            f"Document '{doc.filename}' expired on {field.value}."
                        )
                except ValueError:
                    pass

    # ── Overall confidence (weighted average) ─────────────────────────────
    #   Name:  45%   DOB: 25%   Face: 30%  (normalised if no face docs)
    if face_score is not None:
        overall = name_score * 0.45 + dob_score * 0.25 + face_score * 0.30
    else:
        overall = (name_score * 0.45 + dob_score * 0.25) / 0.70  # re-normalise

    overall = round(min(1.0, max(0.0, overall)), 3)
    return overall, match_scores, face_score, issues


# ── Public entry point ─────────────────────────────────────────────────────

def process_document(filename: str, file_content: bytes) -> DocumentResult:
    """
    Full single-document pipeline:
      1. Classify document type
      2. Extract OCR fields
      3. Detect face presence
    """
    t_start = time.time()

    doc_type, class_conf = classify_document(filename, len(file_content))
    fields              = extract_fields(doc_type)
    has_face            = doc_type in FACE_DOC_TYPES
    face_conf           = round(random.uniform(0.88, 0.99), 3) if has_face else None

    # Simulate realistic processing latency
    time.sleep(random.uniform(0.05, 0.25))
    elapsed_ms = int((time.time() - t_start) * 1000)

    return DocumentResult(
        document_id=str(uuid.uuid4()),
        filename=filename,
        document_type=doc_type,
        classification_confidence=class_conf,
        extracted_fields=fields,
        has_face=has_face,
        face_confidence=face_conf,
        processing_time_ms=elapsed_ms,
        status=VerificationStatus.VERIFIED,
    )