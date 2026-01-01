import os
import time
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import html, re

# Import RAG + analytics helpers
from rag_postgres import (
    init_db,
    create_session_in_db,
    list_sessions,
    get_session_messages,
    log_conversation_db,
    initialize_vector_db,
    query_and_rerank,
    detect_handoff_intent,
    FALLBACK_IMAGE,
    DEFAULT_CHROMA_COLLECTION,
    HANDOFF_CONTACTS,
    HANDOFF_KEYWORDS,
    load_products_csv,
    detect_analytic_query,
    get_most_expensive,
    get_cheapest,
    get_all_products,
    filter_products_by_price,
)

load_dotenv()

# Config
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/mistral-7b-instruct")
PRODUCTS_CSV = os.getenv("PRODUCTS_CSV", "myntra_lipstick_data.csv")
MEMORY_WINDOW = 6

# Init PostgreSQL DB + Load CSV
init_db()
if os.path.exists(PRODUCTS_CSV):
    load_products_csv(PRODUCTS_CSV)
    df = pd.read_csv(PRODUCTS_CSV)
    initialize_vector_db(df, collection_name=DEFAULT_CHROMA_COLLECTION)

# --- HELPER FUNCTIONS ---

def force_rerun():
    """Forces a Streamlit rerun."""
    try:
        st.rerun()
    except:
        st.info("Please refresh manually.")

def load_messages_from_db(sid):
    """
    Fetches messages from DB and updates Session State.
    This acts as the 'Source of Truth'.
    """
    try:
        rows = get_session_messages(sid)
        msgs = []
        for r in rows:
            if r["user_message"]:
                msgs.append({"role": "user", "content": r["user_message"]})
            if r["assistant_message"]:
                msgs.append({"role": "assistant", "content": r["assistant_message"]})
        st.session_state["messages"] = msgs
    except Exception as e:
        st.error(f"Error loading history: {e}")

# Page Setup
st.set_page_config(page_title="AI Support", page_icon="💄", layout="wide")
st.markdown(
    "<div style='text-align:right'><a href='https://www.myntra.com/personal-care?f=Categories%3ALipstick' target='_blank'>Shop Lipsticks on Myntra</a></div>",
    unsafe_allow_html=True)
st.title("AI Support Chatbot 💄")

# Ensure session state structure exists
if "current_session" not in st.session_state:
    st.session_state["current_session"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar Session Management
with st.sidebar:
    if st.button("➕ New Session"):
        sid = f"session_{int(time.time())}"
        try:
            create_session_in_db(sid)
            st.session_state["current_session"] = sid
            st.session_state["messages"] = []
            force_rerun()
        except Exception as e:
            st.error(f"Failed to create session: {e}")

    st.markdown("### Saved sessions")
    # Wrap db listing in try/except to prevent sidebar crash
    try:
        sessions = list_sessions()
    except:
        sessions = []
        
    for s in sessions:
        sid = s["session_id"]
        last = s["last_active"]

        cols = st.columns([6, 1])
        if cols[0].button(f"{sid}\n{last}", key=f"load_{sid}"):
            st.session_state["current_session"] = sid
            load_messages_from_db(sid)
            force_rerun()

        if cols[1].button("Del", key=f"del_{sid}"):
            from rag_postgres import delete_session
            delete_session(sid)
            st.success(f"Deleted {sid}")
            force_rerun()

# --- MAIN CHAT RENDER LOOP ---
# Always render what is in the state (which we sync from DB)
for m in st.session_state["messages"]:
    c = m["content"]
    bg = "#E8F0FE" if m["role"] == "user" else "#FFE3ED"
    st.markdown(f"<div style='background:{bg};color:black;padding:8px;border-radius:8px'>{c}</div>", unsafe_allow_html=True)

# --- CHAT INPUT HANDLER ---
q = st.chat_input("Ask about products or prices...")

if q:
    # 1. AUTO-CREATE SESSION if missing
    if st.session_state["current_session"] is None:
        new_sid = f"session_{int(time.time())}"
        try:
            create_session_in_db(new_sid)
            st.session_state["current_session"] = new_sid
        except Exception as e:
            st.error(f"CRITICAL ERROR: Could not create session in DB. {e}")
            st.stop()
    
    sid = st.session_state["current_session"]

    # 2. DEFINE THE ASSISTANT RESPONSE
    # (We calculate the response *before* displaying it so we can save both at once)
    
    assistant_text = ""
    should_stop = False

    # A. Check Handoff
    handoff = detect_handoff_intent(q)
    if handoff:
        info = handoff["contact"]
        topic = handoff["topic"].title()
        assistant_text = f"<b>{topic} Support</b><br>Phone: {info['phone']}<br>Email: {info['email']}<br>{info['instructions']}"
        should_stop = True

    # B. Check Analytics
    if not assistant_text:
        analytic = detect_analytic_query(q)
        if analytic == "most_expensive":
            item = get_most_expensive()
            assistant_text = f"💎 <b>Most Expensive Lipstick</b><br>{item['Brand']} {item['Product Name']} — ₹{item['Price']}<br><a href='{item['URL']}' target='_blank'>link</a>"
            should_stop = True
        elif analytic == "cheapest":
            item = get_cheapest()
            assistant_text = f"💸 <b>Cheapest Lipstick</b><br>{item['Brand']} {item['Product Name']} — ₹{item['Price']}<br><a href='{item['URL']}' target='_blank'>link</a>"
            should_stop = True
        elif analytic == "all_products":
            items = get_all_products()
            assistant_text = "<b>All Products:</b><br>"
            for i in items:
                assistant_text += f"- {i['Brand']} {i['Product Name']} — ₹{i['Price']} → <a href='{i['URL']}' target='_blank'>link</a><br>"
            should_stop = True
        elif analytic == "price_filter":
            nums = re.findall(r"\d+", q)
            if nums:
                max_price = int(nums[0])
                items = filter_products_by_price(0, max_price)
                assistant_text = f"<b>Products under ₹{max_price}:</b><br>"
                for i in items:
                    assistant_text += f"- {i['Brand']} {i['Product Name']} — ₹{i['Price']} → <a href='{i['URL']}' target='_blank'>link</a><br>"
                should_stop = True

    # C. RAG / Vector Search
    if not assistant_text:
        hits = query_and_rerank(q, n=5)
        docs = [h["doc"] for h in hits]
        metas = [h["meta"] for h in hits]

        # Context Memory
        memory_rows = get_session_messages(sid)[-MEMORY_WINDOW:]
        memory_text = "\n".join(
            [("User: " + r["user_message"]) if r["user_message"] else ("Assistant: " + r["assistant_message"])
             for r in memory_rows]
        )

        system_prompt = """
        You are a STRICT product assistant.
        Rules:
        1. Only use the CONTEXT provided.
        2. If the answer is not in the context, say "I don't have that information."
        3. Never guess or make up products, names, brands, or prices.
        4. Be short, factual, and helpful.
        """

        user_prompt = f"""
        User question: {q}
        Context: {docs}
        Memory: {memory_text}
        """

        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=250,
            temperature=0.0
        )
        assistant_text = html.escape(resp.choices[0].message.content.strip())
        
        # Append matches
        if metas:
            assistant_text += "<br><br><b>Top Matches:</b><br>"
            for m in metas:
                url = m.get("url", FALLBACK_IMAGE)
                assistant_text += f"- {m['brand']} {m['name']} — ₹{m['price']} → <a href='{url}' target='_blank'>link</a><br>"

    # 3. CRITICAL STEP: SAVE TO DB THEN RELOAD STATE
    try:
        # Save to PostgreSQL
        log_conversation_db(sid, q, assistant_text)
        
        # Reload strict from PostgreSQL (Ensures UI and DB are identical)
        load_messages_from_db(sid)
        
        # Force Rerun to update the view immediately
        force_rerun()
        
    except Exception as e:
        st.error(f"❌ DATA SAVE FAILED: {e}")
        # We do NOT rerun here, so you can see the error message.
