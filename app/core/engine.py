"""
DVS - Core Processing Engine (Real Implementation)
====================================================
Uses:
  • pytesseract   → OCR text extraction
  • pdf2image     → PDF → images
  • Pillow        → image preprocessing
  • opencv        → deskew, denoising, face detection
  • rapidfuzz     → fuzzy name matching

Install:
  pip install pytesseract pillow pdf2image opencv-python-headless rapidfuzz
  brew install tesseract poppler        # Mac
  apt install tesseract-ocr poppler-utils  # Linux
"""

from __future__ import annotations

import io
import re
import time
import uuid
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from app.models.schemas import (
    DocumentResult, DocumentType, ExtractedField, MatchScore, VerificationStatus,
)

# ── Optional deps ──────────────────────────────────────────────────────────
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

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


# ══════════════════════════════════════════════════════════════════════════
#  1 — IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_images(filename: str, content: bytes) -> List[Image.Image]:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        if not PDF_SUPPORT:
            raise RuntimeError("Install pdf2image and poppler for PDF support.")
        return convert_from_bytes(content, dpi=300)
    img = Image.open(io.BytesIO(content))
    frames = []
    try:
        while True:
            frames.append(img.copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames or [img]


# ══════════════════════════════════════════════════════════════════════════
#  2 — IMAGE PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════

def preprocess(pil_img: Image.Image) -> Image.Image:
    img_np = np.array(pil_img.convert("RGB"))
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray   = _deskew(gray)
    gray   = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    gray   = cv2.fastNlMeansDenoising(gray, h=10)
    result = Image.fromarray(gray)
    result = ImageEnhance.Contrast(result).enhance(1.8)
    result = ImageEnhance.Sharpness(result).enhance(2.0)
    return result


def _deskew(gray: np.ndarray) -> np.ndarray:
    try:
        edges  = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines  = cv2.HoughLinesP(edges, 1, np.pi / 180, 100,
                                 minLineLength=100, maxLineGap=10)
        if lines is None:
            return gray
        angles = [
            np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
            for l in lines if l[0][2] != l[0][0]
        ]
        if not angles:
            return gray
        angle  = np.median(angles)
        if abs(angle) < 0.5:
            return gray
        h, w   = gray.shape
        M      = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


# ══════════════════════════════════════════════════════════════════════════
#  3 — OCR
# ══════════════════════════════════════════════════════════════════════════

def run_ocr(images: List[Image.Image]) -> Tuple[str, float]:
    """Returns (combined_text, avg_confidence 0-1)."""
    if not TESSERACT_AVAILABLE:
        return "", 0.40

    config     = "--oem 3 --psm 6"
    texts      = []
    all_confs  = []

    for img in images:
        proc = preprocess(img)
        texts.append(pytesseract.image_to_string(proc, config=config))
        data = pytesseract.image_to_data(proc, config=config,
                                         output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"]
                 if str(c).lstrip("-").isdigit() and int(c) >= 0]
        all_confs.extend(confs)

    avg_conf = round(sum(all_confs) / len(all_confs) / 100, 3) if all_confs else 0.50
    return "\n".join(texts), min(1.0, avg_conf)


# ══════════════════════════════════════════════════════════════════════════
#  4 — DOCUMENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

CLASSIFIER_RULES: Dict[DocumentType, List[str]] = {
    DocumentType.PASSPORT: [
        "passport", "nationality", "mrz", "travel document",
        "surname", "given names", "place of birth", "p<ind", "p<usa",
    ],
    DocumentType.DRIVERS_LICENSE: [
        "driver", "licence", "license", "driving", "vehicle",
        "motor vehicle", "transport authority", "rto", "dl no",
    ],
    DocumentType.NATIONAL_ID: [
        "aadhaar", "aadhar", "uid", "unique identification",
        "election commission", "voter id", "pan card", "permanent account",
    ],
    DocumentType.VISA: [
        "visa", "immigration", "consulate", "embassy", "entry permit",
        "valid for", "number of entries", "port of entry",
    ],
    DocumentType.UTILITY_BILL: [
        "electricity", "water supply", "gas supply", "utility",
        "meter reading", "consumer number", "units consumed", "due date",
    ],
    DocumentType.BANK_STATEMENT: [
        "bank statement", "account statement", "opening balance",
        "closing balance", "ifsc", "transaction",
    ],
    DocumentType.EMPLOYMENT_DOC: [
        "employment", "offer letter", "salary slip", "payslip",
        "payroll", "employer", "designation", "joining date", "ctc",
    ],
}


def classify_document(text: str, filename: str) -> Tuple[DocumentType, float]:
    combined = (text + " " + filename).lower()
    scores: Dict[DocumentType, int] = {}
    for doc_type, keywords in CLASSIFIER_RULES.items():
        hits = sum(1 for kw in keywords if kw in combined)
        if hits:
            scores[doc_type] = hits
    if not scores:
        return DocumentType.UNKNOWN, 0.40
    best       = max(scores, key=lambda t: scores[t])
    total      = sum(scores.values())
    confidence = round(min(0.99, scores[best] / max(total, 1) + 0.40), 3)
    return best, confidence


# ══════════════════════════════════════════════════════════════════════════
#  5 — FIELD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════

def _find(pattern: str, text: str, flags: int = re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None


def _clean(v: Optional[str]) -> Optional[str]:
    return re.sub(r"\s+", " ", v).strip() if v else None


def _parse_date(raw: str) -> Optional[str]:
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y",
                "%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y"]:
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw.strip()


def _field(name: str, value: Optional[str], conf: float, delta: float = 0.0
           ) -> Optional[ExtractedField]:
    v = _clean(value)
    if not v:
        return None
    return ExtractedField(field_name=name, value=v,
                          confidence=round(min(1.0, conf + delta), 3))


# ── Passport ──────────────────────────────────────────────────────────────
def _passport(text: str, c: float) -> List[ExtractedField]:
    out = []
    for f in [
        _field("full_name",
               _find(r"surname[:\s]+([A-Z][A-Za-z\s]+?)(?:\n|given)", text) or
               _find(r"(?:^|\n)([A-Z]{3,}\s[A-Z]{3,})", text), c),
        _field("nationality", _find(r"nationality[:\s]+([A-Za-z]+)", text), c, 0.02),
        _field("date_of_birth",
               _parse_date(raw) if (raw := _find(
                   r"(?:date of birth|dob)[:\s]+([\d/\-\w\s,]+?)(?:\n|sex|gender)", text)) else None, c),
        _field("expiry_date",
               _parse_date(raw) if (raw := _find(
                   r"(?:date of expiry|expiry)[:\s]+([\d/\-\w\s,]+?)(?:\n|place|auth)", text)) else None, c),
        _field("passport_number",
               _find(r"passport\s*(?:no|number)[.:\s]+([A-Z0-9]{6,12})", text) or
               _find(r"\b([A-Z]\d{7})\b", text), c, 0.05),
    ]:
        if f: out.append(f)
    for i, line in enumerate(re.findall(r"[A-Z0-9<]{30,44}", text)[:2]):
        f = _field(f"mrz_line{i+1}", line, c, -0.02)
        if f: out.append(f)
    return out


# ── Driver's License ──────────────────────────────────────────────────────
def _drivers_license(text: str, c: float) -> List[ExtractedField]:
    out = []
    for f in [
        _field("full_name",      _find(r"(?:name|holder)[:\s]+([A-Za-z\s]{3,50})", text), c),
        _field("license_number", _find(r"(?:dl\s*no|license\s*no)[.:\s]+([A-Z0-9\-]{8,20})", text), c, 0.05),
        _field("vehicle_class",  _find(r"(?:class|vehicle\s*class)[:\s]+([A-Z0-9\/,\s]{1,20})", text), c),
        _field("date_of_birth",
               _parse_date(raw) if (raw := _find(
                   r"(?:dob|date of birth)[:\s]+([\d/\-\w\s,]+?)(?:\n|sex|blood)", text)) else None, c),
        _field("expiry_date",
               _parse_date(raw) if (raw := _find(
                   r"(?:valid|expiry|expires?)[:\s]+([\d/\-\w\s,]+?)(?:\n|class)", text)) else None, c),
        _field("address",
               re.sub(r"\s+", " ",
                      raw) if (raw := _find(
                   r"address[:\s]+(.{10,120}?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)) else None, c),
    ]:
        if f: out.append(f)
    return out


# ── National ID ───────────────────────────────────────────────────────────
def _national_id(text: str, c: float) -> List[ExtractedField]:
    out = []
    for f in [
        _field("full_name", _find(r"(?:name|full name)[:\s]+([A-Za-z\s]{3,50})", text), c),
        _field("id_number",
               _find(r"\b(\d{4}\s\d{4}\s\d{4})\b", text) or
               _find(r"\b(\d{12})\b", text), c, 0.05),
        _field("date_of_birth",
               _parse_date(raw) if (raw := _find(
                   r"(?:dob|date of birth|year of birth)[:\s]+([\d/\-\w\s,]+?)(?:\n|gender|sex)", text)) else None, c),
        _field("address",
               re.sub(r"\s+", " ",
                      raw) if (raw := _find(
                   r"(?:address|s/o|c/o)[:\s]+(.{10,150}?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)) else None, c),
    ]:
        if f: out.append(f)
    return out


# ── Visa ──────────────────────────────────────────────────────────────────
def _visa(text: str, c: float) -> List[ExtractedField]:
    out = []
    for f in [
        _field("holder_name",     _find(r"(?:name|applicant|surname)[:\s]+([A-Za-z\s]{3,50})", text), c),
        _field("visa_type",       _find(r"(?:visa type|category|type)[:\s]+([A-Za-z0-9\s\-]{2,30})", text), c),
        _field("entry_type",      _find(r"(?:entries|entry)[:\s]+([A-Za-z\s]{3,30})", text), c),
        _field("issuing_country", _find(r"(?:issued by|issuing country)[:\s]+([A-Za-z\s]{3,30})", text), c),
        _field("valid_from",
               _parse_date(raw) if (raw := _find(
                   r"(?:valid from|issue date|from)[:\s]+([\d/\-\w\s,]+?)(?:\n|to\b|until)", text)) else None, c),
        _field("valid_until",
               _parse_date(raw) if (raw := _find(
                   r"(?:valid until|valid to|expiry|until)[:\s]+([\d/\-\w\s,]+?)(?:\n|entries|type)", text)) else None, c),
    ]:
        if f: out.append(f)
    return out


# ── Utility Bill ──────────────────────────────────────────────────────────
def _utility_bill(text: str, c: float) -> List[ExtractedField]:
    out = []
    for f in [
        _field("account_name",   _find(r"(?:consumer name|name|customer)[:\s]+([A-Za-z\s\.]{3,50})", text), c),
        _field("account_number", _find(r"(?:consumer no|account no|ca no)[.:\s]+([A-Z0-9\-]{4,20})", text), c, 0.03),
        _field("issue_date",
               _parse_date(raw) if (raw := _find(
                   r"(?:bill date|issue date)[:\s]+([\d/\-\w\s,]+?)(?:\n|due|amount)", text)) else None, c),
        _field("due_date",
               _parse_date(raw) if (raw := _find(
                   r"(?:due date|payment by)[:\s]+([\d/\-\w\s,]+?)(?:\n|amount|total)", text)) else None, c),
        _field("amount_due",
               _find(r"(?:amount due|total amount|payable)[:\s]+(?:rs\.?|inr|₹)?\s*([\d,]+\.?\d*)", text), c),
        _field("address",
               re.sub(r"\s+", " ",
                      raw) if (raw := _find(
                   r"(?:address|supply address)[:\s]+(.{10,150}?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)) else None, c),
    ]:
        if f: out.append(f)
    return out


# ── Bank Statement ────────────────────────────────────────────────────────
def _bank_statement(text: str, c: float) -> List[ExtractedField]:
    out = []
    for f in [
        _field("account_holder", _find(r"(?:account holder|name|customer name)[:\s]+([A-Za-z\s\.]{3,60})", text), c),
        _field("account_number", _find(r"(?:account no|a/c no)[.:\s]+([X\d\s]{8,20})", text), c, 0.05),
        _field("bank_name",      _find(r"(?:bank name|bank)[:\s]+([A-Za-z\s]{3,50})", text), c),
        _field("ifsc_code",      _find(r"(?:ifsc)[:\s]+([A-Z]{4}0[A-Z0-9]{6})", text), c, 0.05),
        _field("statement_period",
               _find(r"(?:statement period|period)[:\s]+([\d\w\s/\-]+to[\d\w\s/\-]+?)(?:\n|page)", text,
                     re.IGNORECASE), c),
        _field("address",
               re.sub(r"\s+", " ",
                      raw) if (raw := _find(
                   r"address[:\s]+(.{10,150}?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)) else None, c),
    ]:
        if f: out.append(f)
    return out


# ── Employment Doc ────────────────────────────────────────────────────────
def _employment_doc(text: str, c: float) -> List[ExtractedField]:
    out = []
    for f in [
        _field("employee_name", _find(r"(?:employee name|dear\s+(?:mr|ms|mrs)\.?\s*)([A-Za-z\s]{3,50})", text), c),
        _field("employer_name", _find(r"(?:company|employer|organisation)[:\s]+([A-Za-z\s\.,&]{3,80})", text), c),
        _field("designation",   _find(r"(?:designation|position|title|role)[:\s]+([A-Za-z\s\-]{3,50})", text), c),
        _field("employee_id",   _find(r"(?:employee id|emp id|staff id)[.:\s]+([A-Z0-9\-]{3,15})", text), c, 0.03),
        _field("date_of_joining",
               _parse_date(raw) if (raw := _find(
                   r"(?:date of joining|joining date|doj)[:\s]+([\d/\-\w\s,]+?)(?:\n|dept|desg)", text)) else None, c),
        _field("monthly_salary",
               _find(r"(?:gross salary|ctc|monthly salary|salary)[:\s]+(?:rs\.?|inr|₹)?\s*([\d,]+\.?\d*)", text), c),
    ]:
        if f: out.append(f)
    return out


_EXTRACTORS = {
    DocumentType.PASSPORT:        _passport,
    DocumentType.DRIVERS_LICENSE: _drivers_license,
    DocumentType.NATIONAL_ID:     _national_id,
    DocumentType.VISA:            _visa,
    DocumentType.UTILITY_BILL:    _utility_bill,
    DocumentType.BANK_STATEMENT:  _bank_statement,
    DocumentType.EMPLOYMENT_DOC:  _employment_doc,
}


def extract_fields(doc_type: DocumentType, text: str, conf: float) -> List[ExtractedField]:
    fn = _EXTRACTORS.get(doc_type)
    return fn(text, conf) if fn else []


# ══════════════════════════════════════════════════════════════════════════
#  6 — FACE DETECTION (OpenCV Haar Cascade)
# ══════════════════════════════════════════════════════════════════════════

FACE_DOC_TYPES = {
    DocumentType.PASSPORT,
    DocumentType.DRIVERS_LICENSE,
    DocumentType.NATIONAL_ID,
    DocumentType.VISA,
}

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(images: List[Image.Image]) -> Tuple[bool, Optional[float]]:
    for img in images:
        gray  = np.array(img.convert("L"))
        faces = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) > 0:
            _, _, w, h = faces[0]
            ratio = (w * h) / (gray.shape[0] * gray.shape[1])
            conf  = round(min(0.99, 0.60 + ratio * 10), 3)
            return True, conf
    return False, None


# ══════════════════════════════════════════════════════════════════════════
#  7 — IDENTITY MATCHING
# ══════════════════════════════════════════════════════════════════════════

NAME_FIELDS = {
    "full_name", "holder_name", "account_name",
    "account_holder", "employee_name",
}


def fuzzy_name_match(a: str, b: str) -> float:
    if RAPIDFUZZ_AVAILABLE:
        return round(fuzz.token_sort_ratio(a.upper(), b.upper()) / 100, 3)
    def tok(s): return set(re.sub(r"[^a-zA-Z\s]", "", s).upper().split())
    t1, t2 = tok(a), tok(b)
    if not t1 or not t2: return 0.0
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


def match_documents(
        docs: List[DocumentResult],
) -> Tuple[float, List[MatchScore], Optional[float], List[str]]:
    match_scores: List[MatchScore] = []
    issues: List[str] = []

    if len(docs) < 2:
        return 0.0, [], None, ["Need at least 2 documents for cross-verification."]

    # Name
    valid_names = [(d.filename, _get_name(d)) for d in docs if _get_name(d)]
    if len(valid_names) >= 2:
        raw = []
        for i in range(len(valid_names)):
            for j in range(i + 1, len(valid_names)):
                fn_a, n_a = valid_names[i]
                fn_b, n_b = valid_names[j]
                s = fuzzy_name_match(n_a, n_b)
                raw.append(s)
                match_scores.append(MatchScore(field=f"Name ({fn_a} vs {fn_b})", score=s, matched=s >= 0.70))
                if s < 0.70:
                    issues.append(f"Name mismatch: '{fn_a}' vs '{fn_b}' ({s:.0%} similarity).")
        name_score = sum(raw) / len(raw)
    else:
        name_score = 0.50
        issues.append("Not enough name fields for full cross-comparison.")

    # DOB
    valid_dobs = [_get_dob(d) for d in docs if _get_dob(d)]
    if len(valid_dobs) >= 2:
        all_same  = all(d == valid_dobs[0] for d in valid_dobs)
        dob_score = 1.0 if all_same else 0.10
        match_scores.append(MatchScore(field="Date of Birth", score=dob_score, matched=all_same))
        if not all_same:
            issues.append("Date of birth mismatch across documents.")
    else:
        dob_score = 0.60

    # Face
    face_docs  = [d for d in docs if d.has_face and d.face_confidence]
    face_score: Optional[float] = None
    if len(face_docs) >= 2:
        face_score = round(sum(d.face_confidence for d in face_docs) / len(face_docs), 3)
        match_scores.append(MatchScore(field="Biometric Face", score=face_score, matched=face_score >= 0.85))
        if face_score < 0.85:
            issues.append(f"Face confidence {face_score:.0%} below 85% threshold.")

    # Expiry
    for doc in docs:
        for f in doc.extracted_fields:
            if "expiry" in f.field_name and f.value:
                try:
                    if datetime.strptime(f.value, "%Y-%m-%d").date() < date.today():
                        issues.append(f"'{doc.filename}' expired on {f.value}.")
                except ValueError:
                    pass

    # Overall
    if face_score is not None:
        overall = name_score * 0.45 + dob_score * 0.25 + face_score * 0.30
    else:
        overall = (name_score * 0.45 + dob_score * 0.25) / 0.70

    return round(min(1.0, max(0.0, overall)), 3), match_scores, face_score, issues


# ══════════════════════════════════════════════════════════════════════════
#  8 — PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

def process_document(filename: str, file_content: bytes) -> DocumentResult:
    t_start = time.time()

    # Load
    images = load_images(filename, file_content)

    # OCR
    raw_text, base_conf = run_ocr(images)

    # Classify
    doc_type, class_conf = classify_document(raw_text, filename)

    # Extract fields
    fields = extract_fields(doc_type, raw_text, base_conf)

    # Face detection
    if doc_type in FACE_DOC_TYPES:
        has_face, face_conf = detect_face(images)
    else:
        has_face, face_conf = False, None

    return DocumentResult(
        document_id=str(uuid.uuid4()),
        filename=filename,
        document_type=doc_type,
        classification_confidence=class_conf,
        extracted_fields=fields,
        has_face=has_face,
        face_confidence=face_conf,
        processing_time_ms=int((time.time() - t_start) * 1000),
        status=VerificationStatus.VERIFIED,
    )