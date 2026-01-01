import os
import time
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import html, re

# --- CONFIG & SETUP ---
st.set_page_config(page_title="AI Support", page_icon="💄", layout="wide")
load_dotenv()

# Import helpers
try:
    from rag_postgres import (
        init_db, create_session_in_db, list_sessions, get_session_messages,
        log_conversation_db, initialize_vector_db, query_and_rerank,
        detect_handoff_intent, detect_analytic_query, get_most_expensive,
        get_cheapest, get_all_products, filter_products_by_price,
        load_products_csv, FALLBACK_IMAGE, DEFAULT_CHROMA_COLLECTION
    )
except ImportError:
    st.error("CRITICAL ERROR: 'rag_postgres.py' not found.")
    st.stop()

# Environment Variables
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "mistralai/mistral-7b-instruct")
PRODUCTS_CSV = os.getenv("PRODUCTS_CSV", "myntra_lipstick_data.csv")

# --- INITIALIZATION ---
if "init_done" not in st.session_state:
    init_db()
    if os.path.exists(PRODUCTS_CSV):
        load_products_csv(PRODUCTS_CSV)
        df = pd.read_csv(PRODUCTS_CSV)
        initialize_vector_db(df, collection_name=DEFAULT_CHROMA_COLLECTION)
    st.session_state["init_done"] = True

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_session" not in st.session_state:
    st.session_state["current_session"] = None

# --- HELPER: FORMAT BUY BUTTON ---
def format_buy_link(url):
    """Creates a nice green button for the link"""
    if not url or url == "#":
        return ""
    # Green button style
    return f"""<a href='{url}' target='_blank' style='
        background-color:#28a745; 
        color:white; 
        padding:4px 8px; 
        text-decoration:none; 
        border-radius:4px; 
        font-size:0.9em;
        font-weight:bold;
    '>BUY NOW 🛒</a>"""

# --- CORE LOGIC: THE CALLBACK ---
def process_input():
    """Run this BEFORE the page reloads."""
    user_text = st.session_state.user_input
    if not user_text:
        return

    # 1. Ensure Session
    if st.session_state["current_session"] is None:
        new_sid = f"session_{int(time.time())}"
        try:
            create_session_in_db(new_sid)
            st.session_state["current_session"] = new_sid
        except Exception as e:
            st.error(f"Session Error: {e}")
            return
    
    sid = st.session_state["current_session"]

    # 2. Add User Message (Instant)
    st.session_state["messages"].append({"role": "user", "content": user_text})

    # 3. Generate Response
    response_text = ""
    
    # A. Handoff
    handoff = detect_handoff_intent(user_text)
    if handoff:
        info = handoff["contact"]
        response_text = f"<b>{handoff['topic'].title()} Support</b><br>Phone: {info['phone']}<br>Email: {info['email']}"

    # B. Analytics (WITH BUTTONS)
    if not response_text:
        analytic = detect_analytic_query(user_text)
        
        if analytic == "most_expensive":
            item = get_most_expensive()
            btn = format_buy_link(item['URL'])
            response_text = f"💎 <b>Most Expensive:</b> {item['Brand']} {item['Product Name']} — ₹{item['Price']} {btn}"
            
        elif analytic == "cheapest":
            item = get_cheapest()
            btn = format_buy_link(item['URL'])
            response_text = f"💸 <b>Cheapest:</b> {item['Brand']} {item['Product Name']} — ₹{item['Price']} {btn}"
            
        elif analytic == "all_products":
            items = get_all_products()
            response_text = "<b>All Products:</b><br>"
            for i in items:
                btn = format_buy_link(i['URL'])
                response_text += f"- {i['Brand']} {i['Product Name']} — ₹{i['Price']} {btn}<br>"
                
        elif analytic == "price_filter":
            nums = re.findall(r"\d+", user_text)
            if nums:
                max_p = int(nums[0])
                items = filter_products_by_price(0, max_p)
                response_text = f"<b>Under ₹{max_p}:</b><br>"
                for i in items:
                    btn = format_buy_link(i['URL'])
                    response_text += f"- {i['Brand']} {i['Product Name']} — ₹{i['Price']} {btn}<br>"

    # C. RAG / AI (WITH BUTTONS)
    if not response_text:
        hits = query_and_rerank(user_text, n=3)
        docs = [h["doc"] for h in hits]
        
        mem_history = st.session_state["messages"][-6:]
        mem_str = "\n".join([f"{m['role']}: {m['content']}" for m in mem_history])

        # Strict Prompt to prevent LLM from making up links
        system_prompt = """
        You are a shopping assistant. 
        If you find products in the Context, mention them briefly.
        DO NOT try to create links yourself. The system will add the 'Buy Now' buttons automatically below your answer.
        """
        user_prompt = f"Question: {user_text}\nContext: {docs}\nHistory: {mem_str}"

        try:
            client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
            resp = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=250, temperature=0
            )
            response_text = html.escape(resp.choices[0].message.content.strip())
            
            # Append Top Matches with BUTTONS
            if hits:
                response_text += "<br><br><b>Recommended Matches:</b><br>"
                for m in [h["meta"] for h in hits]:
                    btn = format_buy_link(m.get('url', '#'))
                    response_text += f"- {m['brand']} {m['name']} — ₹{m['price']} {btn}<br>"
                    
        except Exception as e:
            response_text = f"AI Error: {e}"

    # 4. Save Response
    st.session_state["messages"].append({"role": "assistant", "content": response_text})

    # 5. DB Sync (Silent)
    try:
        log_conversation_db(sid, user_text, response_text)
    except Exception:
        pass 

# --- HELPER: LOAD HISTORY ---
def load_history(sid):
    try:
        rows = get_session_messages(sid)
        st.session_state["messages"] = []
        for r in rows:
            if r["user_message"]:
                st.session_state["messages"].append({"role": "user", "content": r["user_message"]})
            if r["assistant_message"]:
                st.session_state["messages"].append({"role": "assistant", "content": r["assistant_message"]})
    except Exception as e:
        st.error(f"Load Error: {e}")

# --- SIDEBAR ---
with st.sidebar:
    st.title("History")
    if st.button("➕ New Chat"):
        sid = f"session_{int(time.time())}"
        create_session_in_db(sid)
        st.session_state["current_session"] = sid
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    try:
        sessions = list_sessions()
    except:
        sessions = []
    
    for s in sessions:
        sid = s["session_id"]
        if st.button(f"📄 {sid}", key=f"load_{sid}"):
            st.session_state["current_session"] = sid
            load_history(sid)
            st.rerun()

# --- MAIN UI ---
st.title("AI Support Chatbot 💄")

# 1. Render Messages
for msg in st.session_state["messages"]:
    bg_color = "#E8F0FE" if msg["role"] == "user" else "#FFE3ED"
    st.markdown(
        f"""<div style='background:{bg_color}; color:black; padding:10px; border-radius:10px; margin-bottom:5px;'>
        {msg['content']}</div>""", 
        unsafe_allow_html=True
    )

# 2. Input Widget (Linked to Callback)
st.chat_input("Ask about products or prices...", key="user_input", on_submit=process_input)
