"""
DVS - REST API Routes
========================
Endpoints:
  POST   /api/v1/verify/upload          Upload & verify 1-10 documents
  GET    /api/v1/verify/{session_id}    Get a verification result
  GET    /api/v1/sessions               List all sessions (last 50)
  DELETE /api/v1/sessions/{session_id}  Delete a session
  GET    /api/v1/stats                  System analytics
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.engine import match_documents, process_document
from app.models.schemas import (
    APIResponse,
    VerificationResult,
    VerificationStatus,
)

router = APIRouter()

# ── In-memory session store ────────────────────────────────────────────────
# Replace with PostgreSQL + SQLAlchemy in production (Milestone 3)
_sessions: dict[str, dict] = {}

# ── Constants ──────────────────────────────────────────────────────────────
MAX_FILES      = 10
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
ALLOWED_TYPES  = {
    "application/pdf",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
}


# ── POST /verify/upload ────────────────────────────────────────────────────

@router.post(
    "/verify/upload",
    response_model=APIResponse,
    tags=["Verification"],
    summary="Upload documents for identity verification",
)
async def upload_and_verify(files: List[UploadFile] = File(...)):
    """
    Upload between 1 and 10 documents (PDF / JPG / PNG / TIFF).

    Returns a **VerificationResult** containing:
    - Extracted fields per document
    - Cross-document name & DOB match scores
    - Biometric face-match score (where applicable)
    - Overall identity confidence (0–1)
    - Flagged issues
    """
    # ── Validate input ─────────────────────────────────────────────────────
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least 1 document is required.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_FILES} documents per batch.")

    session_id = str(uuid.uuid4())
    t_start    = time.time()
    doc_results = []

    # ── Process each file ──────────────────────────────────────────────────
    for upload in files:
        content = await upload.read()

        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File '{upload.filename}' exceeds the 50 MB limit.",
            )

        result = process_document(upload.filename or "document", content)
        doc_results.append(result)

    # ── Cross-verify ───────────────────────────────────────────────────────
    overall_conf, match_scores, face_score, issues = match_documents(doc_results)
    total_ms = int((time.time() - t_start) * 1000)

    # ── Determine status ───────────────────────────────────────────────────
    if overall_conf >= 0.75:
        status = VerificationStatus.VERIFIED
    elif overall_conf >= 0.50:
        status = VerificationStatus.PARTIAL
    else:
        status = VerificationStatus.FAILED

    # ── Build result ───────────────────────────────────────────────────────
    verification = VerificationResult(
        session_id=session_id,
        documents=doc_results,
        overall_confidence=overall_conf,
        identity_match=overall_conf >= 0.75,
        match_scores=match_scores,
        face_match_score=face_score,
        issues=issues,
        status=status,
        completed_at=datetime.utcnow(),
        total_processing_time_ms=total_ms,
    )

    # ── Persist session ────────────────────────────────────────────────────
    _sessions[session_id] = verification.model_dump(mode="json")

    return APIResponse(
        success=True,
        message="Verification complete.",
        data=_sessions[session_id],
    )


# ── GET /verify/{session_id} ───────────────────────────────────────────────

@router.get(
    "/verify/{session_id}",
    response_model=APIResponse,
    tags=["Verification"],
    summary="Retrieve a verification result by session ID",
)
async def get_verification(session_id: str):
    result = _sessions.get(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found.")
    return APIResponse(success=True, message="Session retrieved.", data=result)


# ── GET /sessions ──────────────────────────────────────────────────────────

@router.get(
    "/sessions",
    response_model=APIResponse,
    tags=["Verification"],
    summary="List all verification sessions (last 50)",
)
async def list_sessions():
    all_sessions = sorted(
        _sessions.values(),
        key=lambda s: s.get("created_at", ""),
        reverse=True,
    )
    return APIResponse(
        success=True,
        message=f"{len(all_sessions)} session(s) found.",
        data=all_sessions[:50],
    )


# ── DELETE /sessions/{session_id} ─────────────────────────────────────────

@router.delete(
    "/sessions/{session_id}",
    response_model=APIResponse,
    tags=["Verification"],
    summary="Delete a verification session",
)
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del _sessions[session_id]
    return APIResponse(success=True, message="Session deleted.")


# ── GET /stats ─────────────────────────────────────────────────────────────

@router.get(
    "/stats",
    response_model=APIResponse,
    tags=["System"],
    summary="System-level analytics",
)
async def get_stats():
    total    = len(_sessions)
    verified = sum(1 for s in _sessions.values() if s.get("status") == "verified")
    failed   = sum(1 for s in _sessions.values() if s.get("status") == "failed")
    partial  = sum(1 for s in _sessions.values() if s.get("status") == "partial")
    docs     = sum(len(s.get("documents", [])) for s in _sessions.values())
    avg_conf = (
        round(sum(s.get("overall_confidence", 0) for s in _sessions.values()) / total, 3)
        if total > 0 else 0.0
    )

    return APIResponse(
        success=True,
        message="System stats.",
        data={
            "total_sessions":            total,
            "verified":                  verified,
            "failed":                    failed,
            "partial":                   partial,
            "total_documents_processed": docs,
            "avg_confidence":            avg_conf,
        },
    )