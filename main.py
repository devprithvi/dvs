from fastapi import FastAPI
from starlette.responses import HTMLResponse

from app.api.routes import router
from pathlib import Path
app = FastAPI(title="DVS - Document Verification System")

# include API routes
app.include_router(router, prefix="/api/v1")

# path to frontend folder
frontend_path = Path(__file__).parent / "app" / "frontend"

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(frontend_path / "index.html") as f:
        return f.read()
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
