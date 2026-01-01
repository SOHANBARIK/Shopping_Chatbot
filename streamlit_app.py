import os
import time
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import html, re

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="AI Support", page_icon="💄", layout="wide")
load_dotenv()

# Import helpers (Assuming these work correctly)
try:
    from rag_postgres import (
        init_db, create_session_in_db, list_sessions, get_session_messages,
        log_conversation_db, initialize_vector_db, query_and_rerank,
        detect_handoff_intent, detect_analytic_query, get_most_expensive,
        get_cheapest, get_all_products, filter_products_by_price,
        load_products_csv, FALLBACK_IMAGE, DEFAULT_CHROMA_COLLECTION
    )
except ImportError:
    st.error("CRITICAL: 'rag_postgres.py' not found. Please ensure it is in the same directory.")
    st.stop()

# Environment Variables
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/mistral-7b-instruct")
PRODUCTS_CSV = os.getenv("PRODUCTS_CSV", "myntra_lipstick_data.csv")
MEMORY_WINDOW = 6

# --- INITIALIZATION (ONCE ONLY) ---
if "init_done" not in st.session_state:
    init_db()
    if os.path.exists(PRODUCTS_CSV):
        load_products_csv(PRODUCTS_CSV)
        df = pd.read_csv(PRODUCTS_CSV)
        initialize_vector_db(df, collection_name=DEFAULT_CHROMA_COLLECTION)
    st.session_state["init_done"] = True

# --- SESSION STATE SETUP ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_session" not in st.session_state:
    st.session_state["current_session"] = None

# --- HELPER: SAFE RERUN ---
def safe_rerun():
    time.sleep(0.1)  # tiny buffer to allow state to settle
    st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.title("History")
    if st.button("➕ New Chat", use_container_width=True):
        sid = f"session_{int(time.time())}"
        try:
            create_session_in_db(sid)
            st.session_state["current_session"] = sid
            st.session_state["messages"] = [] # Clear local view
            safe_rerun()
        except Exception as e:
            st.error(f"DB Error: {e}")

    st.markdown("---")
    
    # Load Sessions (Wrapped in Try/Except)
    try:
        sessions = list_sessions()
    except Exception:
        sessions = []

    for s in sessions:
        sid = s["session_id"]
        # Load Button
        if st.button(f"📄 {sid}", key=f"load_{sid}"):
            st.session_state["current_session"] = sid
            # Fetch from DB to Sync
            rows = get_session_messages(sid)
            loaded_msgs = []
            for r in rows:
                if r["user_message"]:
                    loaded_msgs.append({"role": "user", "content": r["user_message"]})
                if r["assistant_message"]:
                    loaded_msgs.append({"role": "assistant", "content": r["assistant_message"]})
            st.session_state["messages"] = loaded_msgs
            safe_rerun()

# --- MAIN CHAT INTERFACE ---
st.title("AI Support Chatbot 💄")

# 1. Display Chat History (From Session State)
for msg in st.session_state["messages"]:
    bg_color = "#E8F0FE" if msg["role"] == "user" else "#FFE3ED"
    st.markdown(
        f"""<div style='background:{bg_color}; color:black; padding:10px; border-radius:10px; margin-bottom:5px;'>
        {msg['content']}</div>""", 
        unsafe_allow_html=True
    )

# 2. Handle Input
user_input = st.chat_input("Ask about products or prices...")

if user_input:
    # A. Handle Session Creation if Missing
    if st.session_state["current_session"] is None:
        new_sid = f"session_{int(time.time())}"
        try:
            create_session_in_db(new_sid)
            st.session_state["current_session"] = new_sid
        except Exception as e:
            st.error(f"Could not create session: {e}")
            st.stop()
    
    sid = st.session_state["current_session"]

    # B. Add User Message to State & Display INSTANTLY
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.markdown(
        f"<div style='background:#E8F0FE; color:black; padding:10px; border-radius:10px; margin-bottom:5px;'>{user_input}</div>", 
        unsafe_allow_html=True
    )

    # C. Generate Response Logic
    response_text = ""
    
    # Check 1: Handoff
    if not response_text:
        handoff = detect_handoff_intent(user_input)
        if handoff:
            info = handoff["contact"]
            response_text = f"<b>{handoff['topic'].title()} Support</b><br>Phone: {info['phone']}<br>Email: {info['email']}"

    # Check 2: Analytics
    if not response_text:
        analytic = detect_analytic_query(user_input)
        if analytic == "most_expensive":
            item = get_most_expensive()
            response_text = f"💎 <b>Most Expensive:</b> {item['Brand']} {item['Product Name']} — ₹{item['Price']}"
        elif analytic == "cheapest":
            item = get_cheapest()
            response_text = f"💸 <b>Cheapest:</b> {item['Brand']} {item['Product Name']} — ₹{item['Price']}"
        elif analytic == "all_products":
            items = get_all_products()
            response_text = "<b>All Products:</b><br>" + "".join([f"- {i['Brand']} {i['Product Name']} — ₹{i['Price']}<br>" for i in items])
        elif analytic == "price_filter":
            nums = re.findall(r"\d+", user_input)
            if nums:
                max_p = int(nums[0])
                items = filter_products_by_price(0, max_p)
                response_text = f"<b>Under ₹{max_p}:</b><br>" + "".join([f"- {i['Brand']} {i['Product Name']} — ₹{i['Price']}<br>" for i in items])

    # Check 3: RAG / LLM
    if not response_text:
        # Get Context
        hits = query_and_rerank(user_input, n=3)
        docs = [h["doc"] for h in hits]
        
        # Prepare Memory (Local State)
        mem_history = st.session_state["messages"][-6:]
        mem_str = "\n".join([f"{m['role']}: {m['content']}" for m in mem_history])

        system_prompt = "You are a helpful shopping assistant. Use the Context to answer. If unsure, say so."
        user_prompt = f"Question: {user_input}\nContext: {docs}\nHistory: {mem_str}"

        try:
            client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=250, temperature=0
            )
            response_text = html.escape(resp.choices[0].message.content.strip())
            
            # Append Links
            if hits:
                response_text += "<br><br><b>Matches:</b><br>"
                for m in [h["meta"] for h in hits]:
                    response_text += f"- {m['brand']} {m['name']} — ₹{m['price']} <a href='{m.get('url', '#')}'>link</a><br>"
        except Exception as e:
            response_text = f"AI Error: {e}"

    # D. Display Assistant Response INSTANTLY
    st.session_state["messages"].append({"role": "assistant", "content": response_text})
    st.markdown(
        f"<div style='background:#FFE3ED; color:black; padding:10px; border-radius:10px; margin-bottom:5px;'>{response_text}</div>", 
        unsafe_allow_html=True
    )

    # E. Save to DB (Silent)
    # If this fails or triggers a reload, the user has already seen the text above.
    try:
        log_conversation_db(sid, user_input, response_text)
    except Exception as e:
        print(f"DB Write Failed: {e}") 
        # We print to console, NOT UI, to avoid disrupting the user experience
