"""
DVS - Pydantic Schemas
All request/response models for the API.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional
import uuid

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    AADHAAR         = "aadhaar"
    PAN             = "pan"
    PASSPORT        = "passport"
    VISA            = "visa"
    AIR_TICKET      = "air_ticket"
    AD2             = "ad2"                 # Arrival/Departure card
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID     = "national_id"
    UTILITY_BILL    = "utility_bill"
    BANK_STATEMENT  = "bank_statement"
    EMPLOYMENT_DOC  = "employment_doc"
    UNKNOWN         = "unknown"


class VerificationStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    VERIFIED   = "verified"
    FAILED     = "failed"
    PARTIAL    = "partial"


# ── Sub-models ─────────────────────────────────────────────────────────────

class ExtractedField(BaseModel):
    """A single field extracted from a document via OCR."""
    field_name: str
    value:      Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


class MatchScore(BaseModel):
    """Cross-document match score for a single field/check."""
    field:   str
    score:   float
    matched: bool


class SignatureMatchScore(BaseModel):
    """Signature comparison result between two documents."""
    doc_a:   str
    doc_b:   str
    score:   float
    matched: bool


# ── Document Result ────────────────────────────────────────────────────────

class DocumentResult(BaseModel):
    """Result of processing a single uploaded document."""
    document_id:               str          = Field(default_factory=lambda: str(uuid.uuid4()))
    filename:                  str
    document_type:             DocumentType
    classification_confidence: float
    extracted_fields:          List[ExtractedField] = []
    has_face:                  bool                 = False
    face_confidence:           Optional[float]      = None
    has_signature:             bool                 = False
    processing_time_ms:        int                  = 0
    status:                    VerificationStatus   = VerificationStatus.VERIFIED


# ── Verification Result ────────────────────────────────────────────────────

class VerificationResult(BaseModel):
    """Full cross-document verification session result."""
    session_id:               str                      = Field(default_factory=lambda: str(uuid.uuid4()))
    documents:                List[DocumentResult]     = []
    overall_confidence:       float                    = 0.0
    identity_match:           bool                     = False
    match_scores:             List[MatchScore]          = []
    face_match_score:         Optional[float]          = None
    signature_matches:        List[SignatureMatchScore] = []
    issues:                   List[str]                = []
    status:                   VerificationStatus       = VerificationStatus.PENDING
    created_at:               datetime                 = Field(default_factory=datetime.utcnow)
    completed_at:             Optional[datetime]       = None
    total_processing_time_ms: int                      = 0


# ── API Wrapper ────────────────────────────────────────────────────────────

class APIResponse(BaseModel):
    """Standard API envelope."""
    success: bool
    message: str
    data:    Optional[Any] = None