"""
FastAPI server for the VCKG RAG demo.

Serves the React/Tailwind GUI at http://localhost:8000
and exposes REST endpoints used by the frontend.

Usage:
    .venv/bin/python src/rag/server.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make sure the project root is on the path so we can import rag_sparql_gen
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rag.rag_sparql_gen import (
    DEFAULT_MODEL,
    EVAL_QUESTIONS,
    answer_no_rag,
    answer_with_rag,
    build_schema_summary,
    load_graph,
)

# ---------------------------------------------------------------------------
# App setup — load graph once at startup
# ---------------------------------------------------------------------------
app = FastAPI(title="VCKG RAG Demo")

STATIC_DIR = Path(__file__).parent / "static"

_graph  = None
_schema = None


@app.on_event("startup")
def startup() -> None:
    global _graph, _schema
    print("Loading VCKG graph …")
    _graph  = load_graph()
    _schema = build_schema_summary(_graph)
    print("Ready.")


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    model: str = DEFAULT_MODEL


class AskResponse(BaseModel):
    question:  str
    baseline:  str
    sparql:    str
    vars:      list[str]
    rows:      list[list[str]]
    repairs:   int
    error:     Optional[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/schema")
def get_schema():
    return JSONResponse({"schema": _schema})


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    baseline = answer_no_rag(req.question, model=req.model)
    result   = answer_with_rag(_graph, _schema, req.question, model=req.model)
    return AskResponse(
        question = req.question,
        baseline = baseline,
        sparql   = result["query"],
        vars     = result["vars"],
        rows     = [list(r) for r in result["rows"]],
        repairs  = result["repairs"],
        error    = result["error"],
    )


@app.get("/api/eval")
def run_eval(model: str = DEFAULT_MODEL):
    results = []
    for q in EVAL_QUESTIONS:
        baseline = answer_no_rag(q, model=model)
        result   = answer_with_rag(_graph, _schema, q, model=model)
        results.append({
            "question": q,
            "baseline": baseline,
            "sparql":   result["query"],
            "vars":     result["vars"],
            "rows":     [list(r) for r in result["rows"]],
            "repairs":  result["repairs"],
            "error":    result["error"],
        })
    return JSONResponse(results)


@app.get("/api/eval/stream")
def eval_stream(model: str = DEFAULT_MODEL):
    """SSE endpoint — yields one JSON event per question as it completes."""
    import json
    from fastapi.responses import StreamingResponse

    def generate():
        total = len(EVAL_QUESTIONS)
        for i, q in enumerate(EVAL_QUESTIONS):
            # Send a "progress" event so the UI can show which question is running
            yield f"data: {json.dumps({'type': 'progress', 'index': i, 'total': total, 'question': q})}\n\n"

            baseline = answer_no_rag(q, model=model)
            result   = answer_with_rag(_graph, _schema, q, model=model)

            payload = {
                "type":     "result",
                "index":    i,
                "question": q,
                "baseline": baseline,
                "sparql":   result["query"],
                "vars":     result["vars"],
                "rows":     [list(r) for r in result["rows"]],
                "repairs":  result["repairs"],
                "error":    result["error"],
            }
            yield f"data: {json.dumps(payload)}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'total': total})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Serve the PDF report
# ---------------------------------------------------------------------------
REPORTS_DIR = ROOT / "reports"


@app.get("/report")
def serve_report():
    pdfs = list(REPORTS_DIR.glob("*.pdf"))
    if not pdfs:
        return JSONResponse({"error": "No PDF report found."}, status_code=404)
    return FileResponse(str(pdfs[0]), media_type="application/pdf")


# ---------------------------------------------------------------------------
# Serve the React SPA
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("src.rag.server:app", host="0.0.0.0", port=8000, reload=False)
