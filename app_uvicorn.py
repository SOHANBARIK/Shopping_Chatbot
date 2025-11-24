


# server.py – FastAPI server for your RAG backend (synced handoff logic)
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from openai import OpenAI
import os
from contextlib import asynccontextmanager
import logging
import re
import time

from rag_postgres import (
    init_db,
    create_session_in_db,
    list_sessions,
    get_session_messages,
    delete_session,
    log_conversation_db,
    query_and_rerank,
    detect_analytic_query,
    get_most_expensive,
    get_cheapest,
    get_all_products,
    filter_products_by_price,
    load_products_csv,
    FALLBACK_IMAGE,
    detect_handoff_intent,
    HANDOFF_CONTACTS,
    HANDOFF_KEYWORDS,
)

# Load env
from dotenv import load_dotenv
load_dotenv()

# LLM config
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/mistral-7b-instruct")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if os.path.exists("myntra_lipstick_data.csv"):
        load_products_csv("myntra_lipstick_data.csv")
    logger.info("🚀 Server started with PostgreSQL + Chroma RAG")
    yield
    # optional shutdown logic here

app = FastAPI(title="RAG API Server", lifespan=lifespan)

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Request Models ----
class ChatRequest(BaseModel):
    session_id: Optional[str]
    query: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class AnalyticsRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {
        "message": "🚀 RAG FastAPI server is running!",
        "endpoints": {
            "/chat": "Chat with RAG + LLM",
            "/search": "Vector search",
            "/analytics": "CSV analytics",
            "/sessions": "List sessions",
            "/debug/handoff": "Debug handoff detector"
        }
    }

# ---- Debug endpoint for handoff detector ----
@app.post("/debug/handoff")
def debug_handoff(req: ChatRequest):
    """
    Return raw output of detect_handoff_intent and simple fallback topic detection.
    Useful to debug why a query did/didn't trigger a handoff.
    """
    q = req.query or ""
    detected = detect_handoff_intent(q)
    fallback_topic = _fallback_topic_for_query(q)
    return {"query": q, "detected": detected, "fallback_topic": fallback_topic}

# ---- Helper: fallback topic detection using HANDOFF_KEYWORDS from rag_postgres ----
def _fallback_topic_for_query(q: str) -> Optional[str]:
    if not q:
        return None
    ql = q.lower()
    # look for the best matching topic by keywords defined in HANDOFF_KEYWORDS
    for topic, kws in (HANDOFF_KEYWORDS or {}).items():
        for kw in kws:
            if kw.lower() in ql:
                return topic
    # regex fallback
    if re.search(r"\b(complain|complaint|customer\s+care|support|helpline)\b", ql):
        return "complaint"
    if re.search(r"\b(return|refund|exchange|replace)\b", ql):
        return "returns"
    if re.search(r"\b(offer|discount|coupon|promo|sale|deal)\b", ql):
        return "offers"
    return None

# ---- Chat Endpoint ----
@app.post("/chat")
def chat(req: ChatRequest):
    q = (req.query or "").strip()
    sid = req.session_id

    # Auto-create session if missing
    if not sid:
        sid = f"session_{int(time.time())}"
        create_session_in_db(sid)

    # 0️⃣ FIRST: Detect handoff intent using module detector
    try:
        handoff = detect_handoff_intent(q)
    except Exception as exc:
        logger.exception("detect_handoff_intent raised an exception: %s", exc)
        handoff = None

    logger.debug("detect_handoff_intent -> %s", handoff)

    # If detector didn't match, compute fallback topic using HANDOFF_KEYWORDS
    fallback_topic = None
    if not handoff:
        fallback_topic = _fallback_topic_for_query(q)
        logger.debug("fallback_topic -> %s", fallback_topic)

    if handoff or fallback_topic:
        # prefer structured handoff from detector if present
        if handoff and isinstance(handoff, dict):
            topic = handoff.get("topic")
            contact = handoff.get("contact") or HANDOFF_CONTACTS.get(topic, {})
        else:
            topic = fallback_topic or "complaint"
            contact = HANDOFF_CONTACTS.get(topic, {})

        # Build consistent textual reply and structured handoff object
        reply_text = (
            f"{topic.title()} Support:\n"
            f"Phone: {contact.get('phone', 'N/A')}\n"
            f"Email: {contact.get('email', 'N/A')}\n"
            f"Instructions: {contact.get('instructions', 'N/A')}"
        )

        # persist and return (skip RAG & LLM)
        log_conversation_db(sid, q, reply_text)
        return {
            "session_id": sid,
            "reply": reply_text,
            "handoff": {
                "topic": topic,
                "phone": contact.get("phone"),
                "email": contact.get("email"),
                "instructions": contact.get("instructions")
            },
            "top_matches": []
        }

    # Save user message (assistant column empty for now)
    log_conversation_db(sid, q, "")

    # 1️⃣ Check analytics
    analytic = detect_analytic_query(q)
    if analytic:
        if analytic == "most_expensive":
            return {"session_id": sid, "reply": get_most_expensive()}
        if analytic == "cheapest":
            return {"session_id": sid, "reply": get_cheapest()}
        if analytic == "all_products":
            return {"session_id": sid, "reply": get_all_products()}
        if analytic == "price_filter":
            nums = re.findall(r"\d+", q)
            if nums:
                return {"session_id": sid, "reply": filter_products_by_price(0, int(nums[0]))}

    # 2️⃣ Vector RAG
    hits = query_and_rerank(q, n=5)
    docs = [h.get("doc") for h in hits]
    metas = [h.get("meta") for h in hits]

    # 3️⃣ LLM Call
    prompt = f"""
You are a strict product assistant.
Only answer using the context below.
If answer is not available, say "I don't know".

Context:
{docs}

User question:
{q}
"""

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "Strict assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.0,
        )
        reply = resp.choices[0].message.content
    except Exception:
        logger.exception("LLM call failed; returning fallback 'I don't know'")
        reply = "I don't know"

    # persist assistant reply and return result
    log_conversation_db(sid, q, reply)

    return {
        "session_id": sid,
        "reply": reply,
        "top_matches": metas
    }

# ---- Vector Search ----
@app.post("/search")
def search(req: SearchRequest):
    hits = query_and_rerank(req.query, n=req.top_k)
    return hits

# ---- Analytics ----
@app.post("/analytics")
def analytics(req: AnalyticsRequest):
    q = req.query.lower()
    if "most expensive" in q:
        return get_most_expensive()
    if "cheapest" in q:
        return get_cheapest()
    if "all" in q:
        return get_all_products()
    if "under" in q:
        nums = re.findall(r"\d+", q)
        if nums:
            return filter_products_by_price(0, int(nums[0]))
    return {"error": "No analytics detected"}

# ---- Session Endpoints ----
@app.post("/sessions/new")
def new_session():
    sid = f"session_{int(time.time())}"
    create_session_in_db(sid)
    return {"session_id": sid}

@app.get("/sessions")
def sessions():
    return list_sessions()

@app.get("/sessions/{sid}")
def get_session(sid: str):
    return get_session_messages(sid)

@app.delete("/sessions/{sid}")
def delete_session_endpoint(sid: str):
    delete_session(sid)
    return {"status": "deleted", "session_id": sid}

# ---- Run Uvicorn ----
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
