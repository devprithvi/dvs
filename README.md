# Document Verification System (DVS)

Automated Identity Verification Platform — FastAPI + single-page frontend.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
uvicorn app.main:app --reload --port 8000
```

Open in browser:
- **UI**      → http://localhost:8000
- **Swagger** → http://localhost:8000/api/docs
- **ReDoc**   → http://localhost:8000/api/redoc

---

## Project Structure

```
dvs/
├── app/
│   ├── main.py                  # FastAPI app, mounts routes & frontend
│   ├── api/
│   │   └── routes.py            # All REST API endpoints
│   ├── core/
│   │   └── engine.py            # OCR, classification, matching logic
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── services/                # Future: ocr_service, face_service, etc.
│   └── frontend/
│       └── index.html           # Single-page UI (dark, professional)
├── tests/
│   └── test_engine.py           # Unit tests (pytest)
├── requirements.txt
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/verify/upload` | Upload & verify 1–10 documents |
| GET  | `/api/v1/verify/{session_id}` | Get result by session ID |
| GET  | `/api/v1/sessions` | List all sessions |
| DELETE | `/api/v1/sessions/{session_id}` | Delete a session |
| GET  | `/api/v1/stats` | System analytics |
| GET  | `/health` | Health check |

---

## Supported Document Types

| Type | Face Check | Key Fields |
|------|-----------|------------|
| Passport | ✅ | Name, DOB, Expiry, MRZ, Nationality |
| Driver's License | ✅ | Name, DOB, Address, License No. |
| National ID | ✅ | Name, DOB, ID Number, Address |
| Visa | ✅ | Holder, Validity Dates, Entry Type |
| Utility Bill | ❌ | Account Name, Address, Dates |
| Bank Statement | ❌ | Account Holder, Bank, Period |
| Employment Doc | ❌ | Employee, Employer, Salary |

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Production Roadmap (from DVS System Design)

| Milestone | What to add |
|-----------|-------------|
| M1 — OCR Live | Replace `extract_fields()` with Google Vision API + Tesseract |
| M2 — Verification Engine | Replace face simulation with `face_recognition` / dlib |
| M2 — Fuzzy Matching | Replace name matching with `rapidfuzz` |
| M3 — Database | Connect PostgreSQL via SQLAlchemy + Alembic |
| M3 — Task Queue | Add Celery + Redis for async batch processing |
| M3 — Auth | Add JWT authentication |
| M4 — Production | Docker Compose, Prometheus/Grafana monitoring, ELK logging |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| API Framework | **FastAPI** |
| Server | **Uvicorn** |
| Data Validation | **Pydantic v2** |
| OCR (prod) | Google Vision API + Tesseract + EasyOCR |
| Face Recognition | face_recognition / dlib |
| Fuzzy Matching | rapidfuzz + spaCy |
| Database | PostgreSQL + SQLAlchemy |
| Task Queue | Celery + Redis |
| Storage | MinIO / AWS S3 |
| Frontend | Vanilla JS + CSS (no build step needed) |
| Monitoring | Prometheus + Grafana |
| Logging | ELK Stack |