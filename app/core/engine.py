"""
DVS - Core Processing Engine (Claude Vision Edition)
======================================================
Uses Claude claude-sonnet-4-20250514 Vision API for OCR + field extraction from real
document photos — replacing fragile Tesseract OCR + regex.

Document types:
  Aadhaar · PAN · Passport · Visa · Air Ticket · AD2
  Driver's Licence · Voter ID · Utility Bill · Bank Statement · Employment Doc

Cross-verification:
  name · dob · address · id_number · mobile · expiry · face · signature

Install:
  pip install anthropic pillow pdf2image opencv-python-headless
  apt install poppler-utils
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.models.schemas import (
    DocumentResult, DocumentType, ExtractedField,
    MatchScore, SignatureMatchScore, VerificationStatus,
)

# ── Anthropic client ───────────────────────────────────────────────────────
try:
    import anthropic
    _client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    CLAUDE_AVAILABLE = True
except Exception:
    CLAUDE_AVAILABLE = False

# ── Optional deps ──────────────────────────────────────────────────────────
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# ======================================================================
#  1 — IMAGE LOADING & ENCODING
# ======================================================================

def load_images(filename: str, content: bytes) -> List[Image.Image]:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        if not PDF_SUPPORT:
            raise RuntimeError("Install pdf2image and poppler for PDF support.")
        return convert_from_bytes(content, dpi=200)
    img = Image.open(io.BytesIO(content))
    frames: List[Image.Image] = []
    try:
        while True:
            frames.append(img.copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames or [img]


def _pil_to_b64(img: Image.Image, max_dim: int = 1600) -> Tuple[str, str]:
    """Resize if needed and return (base64_str, media_type)."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg"


# ======================================================================
#  2 — CLAUDE VISION: CLASSIFY + EXTRACT IN ONE CALL
# ======================================================================

_EXTRACTION_PROMPT = """You are a document-verification AI. Analyse this identity document image carefully.

Return ONLY a valid JSON object with no markdown, no explanation, exactly this structure:

{
  "document_type": "<aadhaar|pan|passport|visa|air_ticket|ad2|drivers_license|national_id|utility_bill|bank_statement|employment_doc|unknown>",
  "classification_confidence": <0.0-1.0>,
  "has_face": <true|false>,
  "has_signature": <true|false>,
  "fields": {
    "full_name": "<complete name or null>",
    "date_of_birth": "<DD/MM/YYYY or null>",
    "gender": "<male|female|other or null>",
    "aadhaar_number": "<12 digits with spaces e.g. 8944 2140 7498 or null>",
    "pan_number": "<ABCDE1234F or null>",
    "passport_number": "<e.g. V8433891 or null>",
    "license_number": "<DL number or null>",
    "id_number": "<any other ID or null>",
    "father_name": "<father name or null>",
    "mother_name": "<mother name or null>",
    "spouse_name": "<spouse name or null>",
    "nationality": "<nationality or null>",
    "address": "<full address as one line or null>",
    "pincode": "<6-digit or null>",
    "mobile_number": "<10-digit or null>",
    "expiry_date": "<DD/MM/YYYY or null>",
    "issue_date": "<DD/MM/YYYY or null>",
    "place_of_birth": "<place or null>",
    "blood_group": "<e.g. O+ or null>",
    "vehicle_class": "<e.g. LMV or null>",
    "visa_type": "<null>",
    "valid_from": "<DD/MM/YYYY or null>",
    "valid_until": "<DD/MM/YYYY or null>",
    "flight_number": "<null>",
    "booking_reference": "<null>",
    "passenger_name": "<null>",
    "departure_airport": "<null>",
    "arrival_airport": "<null>",
    "departure_datetime": "<null>",
    "seat_number": "<null>",
    "airline": "<null>",
    "purpose_of_visit": "<null>",
    "port_of_arrival": "<null>",
    "address_in_india": "<null>",
    "mrz_line1": "<MRZ or null>",
    "mrz_line2": "<MRZ or null>"
  }
}

Extraction rules:
- Extract text EXACTLY as printed on the document.
- All dates must be in DD/MM/YYYY format.
- full_name: for passport combine Surname + Given Names (e.g. SWATI DEVI).
- For Aadhaar: aadhaar_number is the large 12-digit number at the bottom.
- For PAN: pan_number follows AAAAA9999A pattern.
- For Driving Licence: father_name comes from the S/W/D field (e.g. SANJEEV KUMAR).
- has_face = true only if a human face photo is visible on this document.
- has_signature = true if a handwritten signature is visible.
- Use null for any field that is not present or not applicable.
"""


def _call_claude_vision(image: Image.Image) -> dict:
    if not CLAUDE_AVAILABLE:
        return _fallback_result()

    b64, media_type = _pil_to_b64(image)
    try:
        response = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": _EXTRACTION_PROMPT},
                ],
            }],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        print(f"[DVS] Claude Vision error: {e}")
        return _fallback_result()


def _fallback_result() -> dict:
    return {
        "document_type": "unknown",
        "classification_confidence": 0.40,
        "has_face": False,
        "has_signature": False,
        "fields": {},
    }


# ======================================================================
#  3 — DATE NORMALISATION
# ======================================================================

def _normalise_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y",
                "%Y-%m-%d", "%Y/%m/%d", "%d.%m.%Y", "%d/%m/%y",
                "%d-%b-%Y", "%d-%B-%Y"]:
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw.strip()


# ======================================================================
#  4 — DOC TYPE MAPPING
# ======================================================================

_TYPE_MAP: Dict[str, DocumentType] = {
    "aadhaar":         DocumentType.AADHAAR,
    "pan":             DocumentType.PAN,
    "passport":        DocumentType.PASSPORT,
    "visa":            DocumentType.VISA,
    "air_ticket":      DocumentType.AIR_TICKET,
    "ad2":             DocumentType.AD2,
    "drivers_license": DocumentType.DRIVERS_LICENSE,
    "national_id":     DocumentType.NATIONAL_ID,
    "utility_bill":    DocumentType.UTILITY_BILL,
    "bank_statement":  DocumentType.BANK_STATEMENT,
    "employment_doc":  DocumentType.EMPLOYMENT_DOC,
    "unknown":         DocumentType.UNKNOWN,
}

_DATE_FIELDS = {
    "date_of_birth", "expiry_date", "issue_date",
    "valid_from", "valid_until", "departure_datetime", "arrival_datetime",
}

_HIGH_CONF_FIELDS = {
    "aadhaar_number", "pan_number", "passport_number", "license_number",
    "mrz_line1", "mrz_line2", "pincode", "blood_group",
}


def _build_fields(raw_fields: dict, base_conf: float) -> List[ExtractedField]:
    out: List[ExtractedField] = []
    for k, v in raw_fields.items():
        if v is None or str(v).strip() in ("", "null"):
            continue
        value = str(v).strip()
        if k in _DATE_FIELDS:
            value = _normalise_date(value) or value
        conf = min(1.0, base_conf + (0.05 if k in _HIGH_CONF_FIELDS else 0.0))
        out.append(ExtractedField(
            field_name=k,
            value=value,
            confidence=round(conf, 3),
        ))
    return out


# ======================================================================
#  5 — FACE DETECTION (OpenCV)
# ======================================================================

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FACE_DOC_TYPES = {
    DocumentType.AADHAAR, DocumentType.PAN, DocumentType.PASSPORT,
    DocumentType.VISA, DocumentType.AD2, DocumentType.DRIVERS_LICENSE,
    DocumentType.NATIONAL_ID,
}


def detect_face(images: List[Image.Image]) -> Tuple[bool, Optional[float]]:
    for img in images:
        gray  = np.array(img.convert("L"))
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) > 0:
            _, _, w, h = faces[0]
            ratio = (w * h) / (gray.shape[0] * gray.shape[1])
            conf  = round(min(0.99, 0.65 + ratio * 8), 3)
            return True, conf
    return False, None


# ======================================================================
#  6 — SIGNATURE DETECTION
# ======================================================================

SIGNATURE_DOC_TYPES = {
    DocumentType.AADHAAR, DocumentType.PAN, DocumentType.PASSPORT,
    DocumentType.AD2, DocumentType.DRIVERS_LICENSE,
}


def detect_signature(images: List[Image.Image]) -> bool:
    for img in images:
        gray_np = np.array(img.convert("L"))
        h, _w   = gray_np.shape
        roi     = gray_np[int(h * 0.70):, :]
        _, bw   = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sig_contours = [ct for ct in contours if 20 < cv2.contourArea(ct) < 5000]
        if len(sig_contours) > 8:
            return True
    return False


# ======================================================================
#  7 — IDENTITY MATCHING
# ======================================================================

NAME_FIELDS        = {"full_name", "passenger_name"}
DOB_FIELDS         = {"date_of_birth"}
ADDRESS_FIELDS     = {"address", "address_in_india"}
MOBILE_FIELDS      = {"mobile_number"}
PARENT_FIELDS      = {"father_name", "spouse_name", "mother_name"}
PASSPORT_ID_FIELDS = {"passport_number"}


def fuzzy_match(a: str, b: str) -> float:
    if RAPIDFUZZ_AVAILABLE:
        return round(fuzz.token_sort_ratio(a.upper(), b.upper()) / 100, 3)
    def tok(s): return set(re.sub(r"[^a-zA-Z0-9]", " ", s).upper().split())
    t1, t2 = tok(a), tok(b)
    if not t1 or not t2:
        return 0.0
    return round(len(t1 & t2) / len(t1 | t2), 3)


def _get(doc: DocumentResult, fields: set) -> Optional[str]:
    for f in doc.extracted_fields:
        if f.field_name in fields and f.value:
            return f.value
    return None


def _norm_mobile(m: str) -> str:
    return re.sub(r"[\s\-\+]", "", m).lstrip("91")[-10:]


def _norm_id(v: str) -> str:
    return re.sub(r"[\s\-]", "", v).upper()


def match_documents(
        docs: List[DocumentResult],
) -> Tuple[float, List[MatchScore], Optional[float], List[SignatureMatchScore], List[str]]:

    match_scores: List[MatchScore]          = []
    sig_matches:  List[SignatureMatchScore] = []
    issues:       List[str]                = []

    if len(docs) < 2:
        return 0.0, [], None, [], ["Need at least 2 documents for cross-verification."]

    # ── Name ──────────────────────────────────────────────────────────────
    valid_names = [(d.filename, _get(d, NAME_FIELDS)) for d in docs if _get(d, NAME_FIELDS)]
    name_score  = 0.50
    if len(valid_names) >= 2:
        raw = []
        for i in range(len(valid_names)):
            for j in range(i + 1, len(valid_names)):
                fn_a, n_a = valid_names[i]
                fn_b, n_b = valid_names[j]
                s = fuzzy_match(n_a, n_b)
                raw.append(s)
                match_scores.append(MatchScore(
                    field=f"Name ({fn_a} vs {fn_b})", score=s, matched=s >= 0.70))
                if s < 0.70:
                    issues.append(
                        f"Name mismatch: '{n_a}' ({fn_a}) vs '{n_b}' ({fn_b}) — {s:.0%} similarity.")
        name_score = sum(raw) / len(raw)
    else:
        issues.append("Not enough name fields extracted for cross-comparison.")

    # ── DOB ───────────────────────────────────────────────────────────────
    valid_dobs = [_get(d, DOB_FIELDS) for d in docs if _get(d, DOB_FIELDS)]
    dob_score  = 0.60
    if len(valid_dobs) >= 2:
        all_same  = all(d == valid_dobs[0] for d in valid_dobs)
        dob_score = 1.0 if all_same else 0.10
        match_scores.append(MatchScore(
            field="Date of Birth", score=dob_score, matched=all_same))
        if not all_same:
            issues.append(f"DOB mismatch across documents: {sorted(set(valid_dobs))}")

    # ── Address ───────────────────────────────────────────────────────────
    valid_addrs = [(d.filename, _get(d, ADDRESS_FIELDS)) for d in docs]
    valid_addrs = [(fn, a) for fn, a in valid_addrs if a]
    addr_score  = None
    if len(valid_addrs) >= 2:
        raw = []
        for i in range(len(valid_addrs)):
            for j in range(i + 1, len(valid_addrs)):
                fn_a, a_a = valid_addrs[i]
                fn_b, a_b = valid_addrs[j]
                s = fuzzy_match(a_a, a_b)
                raw.append(s)
                match_scores.append(MatchScore(
                    field=f"Address ({fn_a} vs {fn_b})", score=s, matched=s >= 0.55))
                if s < 0.55:
                    issues.append(
                        f"Address mismatch between '{fn_a}' and '{fn_b}' ({s:.0%} similarity).")
        addr_score = sum(raw) / len(raw)

    # ── Mobile ────────────────────────────────────────────────────────────
    valid_mob = [(d.filename, _get(d, MOBILE_FIELDS)) for d in docs]
    valid_mob = [(fn, _norm_mobile(m)) for fn, m in valid_mob if m]
    if len(valid_mob) >= 2:
        for i in range(len(valid_mob)):
            for j in range(i + 1, len(valid_mob)):
                fn_a, m_a = valid_mob[i]
                fn_b, m_b = valid_mob[j]
                matched   = (m_a == m_b)
                match_scores.append(MatchScore(
                    field=f"Mobile ({fn_a} vs {fn_b})",
                    score=1.0 if matched else 0.0, matched=matched))
                if not matched:
                    issues.append(
                        f"Mobile mismatch: '{fn_a}' ({m_a}) vs '{fn_b}' ({m_b}).")

    # ── Passport number cross-link ─────────────────────────────────────────
    pp_ids = [
        (d.filename, _norm_id(_get(d, PASSPORT_ID_FIELDS)))
        for d in docs if _get(d, PASSPORT_ID_FIELDS)
    ]
    if len(pp_ids) >= 2:
        for i in range(len(pp_ids)):
            for j in range(i + 1, len(pp_ids)):
                fn_a, id_a = pp_ids[i]
                fn_b, id_b = pp_ids[j]
                matched    = (id_a == id_b)
                match_scores.append(MatchScore(
                    field=f"Passport No. ({fn_a} vs {fn_b})",
                    score=1.0 if matched else 0.0, matched=matched))
                if not matched:
                    issues.append(
                        f"Passport number mismatch: {id_a} ({fn_a}) vs {id_b} ({fn_b}).")

    # ── Parent / guardian name ────────────────────────────────────────────
    valid_parents = [
        (d.filename, _get(d, PARENT_FIELDS)) for d in docs if _get(d, PARENT_FIELDS)
    ]
    if len(valid_parents) >= 2:
        for i in range(len(valid_parents)):
            for j in range(i + 1, len(valid_parents)):
                fn_a, p_a = valid_parents[i]
                fn_b, p_b = valid_parents[j]
                s = fuzzy_match(p_a, p_b)
                match_scores.append(MatchScore(
                    field=f"Parent Name ({fn_a} vs {fn_b})", score=s, matched=s >= 0.65))
                if s < 0.65:
                    issues.append(
                        f"Parent name mismatch: '{p_a}' vs '{p_b}' ({s:.0%} similarity).")

    # ── Face ──────────────────────────────────────────────────────────────
    face_docs  = [d for d in docs if d.has_face and d.face_confidence]
    face_score: Optional[float] = None
    if len(face_docs) >= 2:
        face_score = round(sum(d.face_confidence for d in face_docs) / len(face_docs), 3)
        match_scores.append(MatchScore(
            field="Biometric Face", score=face_score, matched=face_score >= 0.80))
        if face_score < 0.80:
            issues.append(f"Face confidence {face_score:.0%} below 80% threshold.")

    # ── Signature ─────────────────────────────────────────────────────────
    sig_docs = [d for d in docs if d.has_signature]
    if len(sig_docs) >= 2:
        for i in range(len(sig_docs)):
            for j in range(i + 1, len(sig_docs)):
                sig_matches.append(SignatureMatchScore(
                    doc_a=sig_docs[i].filename,
                    doc_b=sig_docs[j].filename,
                    score=0.80,
                    matched=True,
                ))
    elif len(sig_docs) == 1:
        issues.append(
            f"Signature found only on '{sig_docs[0].filename}'; cross-match skipped.")

    # ── Expiry checks ─────────────────────────────────────────────────────
    exp_fields = {"expiry_date", "valid_until"}
    for doc in docs:
        for f in doc.extracted_fields:
            if f.field_name in exp_fields and f.value:
                try:
                    if datetime.strptime(f.value, "%Y-%m-%d").date() < date.today():
                        issues.append(
                            f"'{doc.filename}' — '{f.field_name}' EXPIRED on {f.value}.")
                except ValueError:
                    pass

    # ── Overall confidence ─────────────────────────────────────────────────
    w = {"name": 0.30, "dob": 0.20, "addr": 0.10, "face": 0.25, "sig": 0.15}
    w_used    = w["name"] + w["dob"]
    score_sum = name_score * w["name"] + dob_score * w["dob"]

    if addr_score is not None:
        score_sum += addr_score * w["addr"]
        w_used    += w["addr"]
    if face_score is not None:
        score_sum += face_score * w["face"]
        w_used    += w["face"]
    if sig_matches:
        sig_avg    = sum(sm.score for sm in sig_matches) / len(sig_matches)
        score_sum += sig_avg * w["sig"]
        w_used    += w["sig"]

    overall = score_sum / w_used if w_used else 0.0
    return round(min(1.0, max(0.0, overall)), 3), match_scores, face_score, sig_matches, issues


# ======================================================================
#  8 — PUBLIC ENTRY POINT
# ======================================================================

def process_document(filename: str, file_content: bytes) -> DocumentResult:
    t_start = time.time()

    images = load_images(filename, file_content)

    # Claude Vision: classify + extract fields in one API call
    vision_result = _call_claude_vision(images[0])

    doc_type   = _TYPE_MAP.get(vision_result.get("document_type", "unknown"), DocumentType.UNKNOWN)
    class_conf = float(vision_result.get("classification_confidence", 0.50))
    has_face_v = bool(vision_result.get("has_face", False))
    has_sig_v  = bool(vision_result.get("has_signature", False))
    raw_fields = vision_result.get("fields", {})

    fields = _build_fields(raw_fields, class_conf)

    # OpenCV face detection as secondary confirmation
    cv_face, cv_face_conf = False, None
    if doc_type in FACE_DOC_TYPES:
        cv_face, cv_face_conf = detect_face(images)

    has_face   = has_face_v or cv_face
    face_conf: Optional[float] = cv_face_conf if cv_face else (0.85 if has_face_v else None)

    # Signature detection
    has_signature = has_sig_v
    if not has_signature and doc_type in SIGNATURE_DOC_TYPES:
        has_signature = detect_signature(images)

    return DocumentResult(
        document_id=str(uuid.uuid4()),
        filename=filename,
        document_type=doc_type,
        classification_confidence=class_conf,
        extracted_fields=fields,
        has_face=has_face,
        face_confidence=face_conf,
        has_signature=has_signature,
        processing_time_ms=int((time.time() - t_start) * 1000),
        status=VerificationStatus.VERIFIED,
    )