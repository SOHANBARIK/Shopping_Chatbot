"""
rag_postgres.py

A cleaned, PostgreSQL-ready rewrite of your original rag.py.

Features:
- SQLAlchemy engine configured for PostgreSQL via POSTGRES_URL env var
- dotenv support
- safer connection pooling and logging
- same public functions as original (init_db, create_session_in_db, log_conversation_db, list_sessions, get_session_messages, delete_session)
- chroma initialization preserved
- improved error handling and docstrings

Usage:
- set POSTGRES_URL env var (e.g. postgresql+psycopg2://postgres:password@localhost:5432/ragdb)
- pip install -r requirements.txt (sqlalchemy, psycopg2-binary, chromadb)
- call init_db() once to create tables
"""

from __future__ import annotations

import os
import pandas as pd
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

import chromadb
from chromadb.utils import embedding_functions

# Load environment variables from .env (if present)
load_dotenv()

# -----------------------------
# Configuration / Constants
# -----------------------------
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/ragdb",
)
# Fallback for compatibility with older deployments that used a file-based DB
DB_PATH = os.getenv("CHAT_DB_PATH", "chat_demo_modular.db")

DEFAULT_CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "products")
CHROMA_DIR = os.getenv("CHROMA_DB_PATH", "./chroma_db_data")

# Use raw string to avoid Windows path unicode escape issues
FALLBACK_IMAGE = os.getenv("FALLBACK_IMAGE", r"C:\Users\sohan\Downloads\lipstick.jpg")

# Handoff contacts and keywords (kept from original)
HANDOFF_CONTACTS: Dict[str, Dict[str, str]] = {
    "returns": {
        "phone": "+91-1800-111-222",
        "email": "returns@company.example",
        "instructions": "Call for returns/exchanges.",
    },
    "offers": {
        "phone": "+91-1800-111-333",
        "email": "offers@company.example",
        "instructions": "Promotions team will assist.",
    },
    "complaint": {
        "phone": "+91-1800-111-444",
        "email": "complaints@company.example",
        "instructions": "Customer care will assist.",
    },
}

HANDOFF_KEYWORDS: Dict[str, List[str]] = {
    "returns": ["return", "refund", "exchange", "replace", "return policy"],
    "offers": ["offer", "discount", "coupon", "promo", "sale", "deal"],
    "complaint": ["complaint", "not delivered", "damaged", "missing", "issue", "problem"],
}

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag_postgres")
# -----------------------------
# Global product dataframe (loaded once)
PRODUCTS_DF = None

def load_products_csv(path: str):
    """Load the product CSV into memory for analytics queries."""
    global PRODUCTS_DF
    try:
        df = pd.read_csv(path)
        # Ensure price is numeric
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0)
        PRODUCTS_DF = df
        logger.info(f"Loaded {len(df)} products from CSV.")
    except Exception as e:
        logger.exception(f"Failed to load CSV: {e}")
        PRODUCTS_DF = None

# -----------------------------
# SQLAlchemy Engine (Postgres)
# -----------------------------
# The engine is created lazily inside get_engine() so imports/tests won't fail
_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Return a SQLAlchemy Engine configured for PostgreSQL.

    The engine is cached to avoid recreating pools repeatedly.
    """
    global _engine
    if _engine is not None:
        return _engine

    try:
        _engine = create_engine(
            POSTGRES_URL,
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
            pool_pre_ping=True,
            echo=(os.getenv("DB_ECHO", "false").lower() == "true"),
        )
        logger.info("Created SQLAlchemy engine for PostgreSQL")
        return _engine
    except Exception as e:
        logger.exception("Failed to create SQLAlchemy engine: %s", e)
        raise

# -----------------------------
# ChromaDB client + embeddings
# -----------------------------
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    logger.info("Initialized Chroma client and embedding function")
except Exception:
    # If chroma import/initialization fails, we allow the module to be imported but
    # surface the error when functions that require Chroma are invoked.
    chroma_client = None  # type: ignore
    embed_fn = None  # type: ignore
    logger.exception("Failed to initialize Chroma client or embedding function")

# -----------------------------
# Utilities
# -----------------------------

def now_utc() -> datetime:
    """Return current UTC timestamp aware datetime."""
    return datetime.now(timezone.utc)

# -----------------------------
# Database schema initialization
# -----------------------------

def init_db() -> None:
    """Create necessary tables in the configured PostgreSQL database.

    This function uses raw SQL compatible with SQLite and PostgreSQL.
    Call it once during app startup (or call repeatedly; CREATE TABLE IF NOT EXISTS is idempotent).
    """
    engine = get_engine()

    create_sessions_sql = text(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP WITH TIME ZONE,
            last_active TIMESTAMP WITH TIME ZONE
        )
        """
    )

    create_chat_logs_sql = text(
        """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id SERIAL PRIMARY KEY,
            session_id TEXT,
            user_message TEXT,
            assistant_message TEXT,
            created_at TIMESTAMP WITH TIME ZONE
        )
        """
    )

    try:
        with engine.begin() as conn:
            conn.execute(create_sessions_sql)
            conn.execute(create_chat_logs_sql)
        logger.info("Database tables ensured (sessions, chat_logs)")
    except SQLAlchemyError:
        logger.exception("Error creating tables in the database")
        raise

# -----------------------------
# Session management functions
# -----------------------------

def create_session_in_db(sid: str) -> None:
    """Create or update a session row with the current timestamp."""
    now = now_utc()
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO sessions (session_id, created_at, last_active) VALUES (:s, :c, :l)"
                    " ON CONFLICT (session_id) DO UPDATE SET last_active = EXCLUDED.last_active"
                ),
                {"s": sid, "c": now, "l": now},
            )
        logger.debug("Session created/updated: %s", sid)
    except SQLAlchemyError:
        logger.exception("Failed to create/update session: %s", sid)
        raise


def log_conversation_db(sid: str, user_msg: str, assistant_msg: str) -> None:
    """Log a single conversational exchange and update session last_active."""
    now = now_utc()
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO chat_logs (session_id, user_message, assistant_message, created_at)"
                    " VALUES (:s, :u, :a, :c)"
                ),
                {"s": sid, "u": user_msg, "a": assistant_msg, "c": now},
            )
            conn.execute(
                text("UPDATE sessions SET last_active = :l WHERE session_id = :s"),
                {"l": now, "s": sid},
            )
        logger.debug("Logged conversation for session: %s", sid)
    except SQLAlchemyError:
        logger.exception("Failed to log conversation for session: %s", sid)
        raise


def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    """Return up to `limit` sessions ordered by last_active desc."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT session_id, last_active FROM sessions ORDER BY last_active DESC LIMIT :l"
                ),
                {"l": limit},
            ).fetchall()
        return [dict(r._mapping) for r in rows]
    except SQLAlchemyError:
        logger.exception("Failed to list sessions")
        raise


def get_session_messages(sid: str) -> List[Dict[str, Any]]:
    """Return ordered messages for a session."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT user_message, assistant_message, created_at FROM chat_logs WHERE session_id = :s ORDER BY created_at"
                ),
                {"s": sid},
            ).fetchall()
        return [dict(r._mapping) for r in rows]
    except SQLAlchemyError:
        logger.exception("Failed to fetch session messages for: %s", sid)
        raise


def delete_session(sid: str) -> None:
    """Delete a session and all its chat logs."""
    engine = get_engine()
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM chat_logs WHERE session_id = :s"), {"s": sid})
            conn.execute(text("DELETE FROM sessions WHERE session_id = :s"), {"s": sid})
        logger.info("Deleted session and logs for: %s", sid)
    except SQLAlchemyError:
        logger.exception("Failed to delete session: %s", sid)
        raise

# -----------------------------
# Vector DB helpers (Chroma)
# -----------------------------

def initialize_vector_db(df, collection_name: str = DEFAULT_CHROMA_COLLECTION):
    """Index pandas DataFrame rows into a Chroma collection.

    The DataFrame is expected to have columns similar to the original implementation
    (e.g. product_id, Product Name / Name, Brand, Price, URL).
    """
    if chroma_client is None or embed_fn is None:
        raise RuntimeError("Chroma client or embedding function is not initialized")

    col = chroma_client.get_or_create_collection(name=collection_name, embedding_function=embed_fn)

    # If collection already has vectors, skip re-populating to avoid duplicates
    try:
        # Some versions of Chroma expose a count method
        if hasattr(col, "count") and col.count() > 0:
            logger.info("Chroma collection '%s' already populated, skipping indexing", collection_name)
            return col
    except Exception:
        # Not fatal — proceed to add
        logger.debug("Unable to check collection count; proceeding to (re)index")

    ids, docs, metas = [], [], []
    for i, row in df.iterrows():
        pid = str(row.get("product_id") or f"idx-{i}")
        name = str(row.get("Product Name") or row.get("Name") or "")
        brand = str(row.get("Brand") or "")
        price = row.get("Price", "")
        url = row.get("URL") or row.get("url") or ""
        if not url:
            url = FALLBACK_IMAGE

        doc_text = f"Product: {name}. Brand: {brand}. Price: {price}."
        ids.append(pid)
        docs.append(doc_text)
        metas.append({
            "product_id": pid,
            "name": name,
            "brand": brand,
            "price": price,
            "url": str(url),
        })

    # Add in a single batch
    try:
        col.add(ids=ids, documents=docs, metadatas=metas)
        # Persist if client supports it
        if hasattr(chroma_client, "persist"):
            try:
                chroma_client.persist()
            except Exception:
                logger.debug("Chroma client persist() raised an error; ignoring")
        logger.info("Indexed %d items into Chroma collection '%s'", len(ids), collection_name)
    except Exception:
        logger.exception("Failed to add items to Chroma collection")
        raise

    return col


def detect_handoff_intent(user_query: str) -> Optional[Dict[str, Any]]:
    """Detect handoff intent from a user query and return contact info if matched."""
    if not user_query:
        return None

    q = user_query.lower()
    for topic, keywords in HANDOFF_KEYWORDS.items():
        for kw in keywords:
            if kw in q:
                return {"topic": topic, "contact": HANDOFF_CONTACTS.get(topic)}
    return None


def query_and_rerank(q: str, n: int = 5):
    """Query Chroma for nearest results and apply simple reranking heuristics.

    Returns a list of dicts with keys: 'doc', 'meta', 'score'.
    """
    if chroma_client is None or embed_fn is None:
        raise RuntimeError("Chroma client or embedding function is not initialized")

    col = chroma_client.get_or_create_collection(name=DEFAULT_CHROMA_COLLECTION, embedding_function=embed_fn)

    # Query the collection
    try:
        res = col.query(query_texts=[q], n_results=n, include=["metadatas", "documents", "distances"])
    except Exception:
        logger.exception("Chroma query failed")
        raise

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if "distances" in res else [None] * len(docs)

    hits = []
    ql = q.lower()
    for doc, meta, dist in zip(docs, metas, dists):
        score = 0.0
        try:
            if meta and meta.get("name") and meta["name"].lower() in ql:
                score += 10
            if meta and meta.get("brand") and meta["brand"].lower() in ql:
                score += 5
            if dist is not None:
                # distances from some chroma builds are smaller = better; we convert to a similarity-like score
                score += max(0.0, 1.0 - float(dist))
        except Exception:
            logger.debug("Error computing score for meta: %s", meta)
        hits.append({"doc": doc, "meta": meta, "score": score})

    return sorted(hits, key=lambda x: x["score"], reverse=True)

# -----------------------------
# CSV Analytics Functions
# -----------------------------

def get_all_products() -> List[Dict[str, Any]]:
    """Return all products as list of dicts."""
    if PRODUCTS_DF is None:
        return []
    return PRODUCTS_DF.to_dict(orient="records")


def get_most_expensive() -> Optional[Dict[str, Any]]:
    """Return the highest priced product."""
    if PRODUCTS_DF is None or PRODUCTS_DF.empty:
        return None
    row = PRODUCTS_DF.loc[PRODUCTS_DF["Price"].idxmax()]
    return row.to_dict()


def get_cheapest() -> Optional[Dict[str, Any]]:
    """Return the lowest priced product."""
    if PRODUCTS_DF is None or PRODUCTS_DF.empty:
        return None
    row = PRODUCTS_DF.loc[PRODUCTS_DF["Price"].idxmin()]
    return row.to_dict()


def filter_products_by_price(min_price=0, max_price=999999) -> List[Dict[str, Any]]:
    """Return products within a price range."""
    if PRODUCTS_DF is None:
        return []
    df = PRODUCTS_DF[
        (PRODUCTS_DF["Price"] >= min_price) &
        (PRODUCTS_DF["Price"] <= max_price)
    ]
    return df.to_dict(orient="records")


def detect_analytic_query(q: str) -> Optional[str]:
    """Detect high-level queries requiring full CSV instead of vector RAG."""
    q = q.lower()

    if "most expensive" in q or "highest price" in q or "expensive" in q:
        return "most_expensive"

    if "cheapest" in q or "lowest price" in q:
        return "cheapest"

    if "all products" in q or "show all" in q:
        return "all_products"

    if "under" in q or "<" in q or "less than" in q:
        return "price_filter"

    return None

# -----------------------------
# Module-level convenience
# -----------------------------
if __name__ == "__main__":
    try:
        e = get_engine()
        init_db()
        load_products_csv("myntra_lipstick_data.csv")   # <--- ADD THIS
        logger.info("rag_postgres module initialized, DB + CSV loaded.")
    except Exception as exc:
        logger.exception("Startup check failed: %s", exc)

