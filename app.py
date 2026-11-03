"""BVRIT Hyderabad assistant: RAG-style chatbot over circulars, notifications and proposals."""
import math
import os
import re
from collections import Counter
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

DATA_DIR = Path(__file__).parent / "data"
MODEL = "gemini-flash-latest"  # free-tier Flash model; override with GEMINI_MODEL in secrets
TOP_K = 5

st.set_page_config(page_title="BVRIT Assistant", page_icon="🎓", layout="centered")


# ---------- Knowledge base ----------
def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


@st.cache_resource
def load_index():
    """Split every markdown file in /data into '## ' sections and build a BM25 index."""
    chunks = []
    for f in sorted(DATA_DIR.glob("*.md")):
        raw = f.read_text(encoding="utf-8")
        title = raw.splitlines()[0].lstrip("# ").strip()
        for part in re.split(r"\n(?=## )", raw):
            part = part.strip()
            if len(part) > 30:
                chunks.append({"source": f.name, "title": title, "text": part})
    docs = [tokenize(c["title"] + " " + c["text"]) for c in chunks]
    df = Counter(t for d in docs for t in set(d))
    avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
    return {"chunks": chunks, "docs": docs, "df": df, "avgdl": avgdl, "n": len(docs)}


def search(query: str, k: int = TOP_K):
    idx = load_index()
    q = tokenize(query)
    k1, b = 1.5, 0.75
    scored = []
    for i, d in enumerate(idx["docs"]):
        tf = Counter(d)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (idx["n"] - idx["df"][t] + 0.5) / (idx["df"][t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / idx["avgdl"]))
        scored.append((s, i))
    scored.sort(reverse=True)
    return [idx["chunks"][i] for s, i in scored[:k] if s > 0]


SYSTEM = """You are the BVRIT Hyderabad College of Engineering for Women assistant. You help students and parents with exam fees and deadlines, exam rules, regulations (BH25), circulars (mobile phone policy, Student Affairs Council), the iACE training proposal, and the Higiene sanitary pad company deck.

Rules:
- Answer ONLY from the CONTEXT provided. If the answer is not there, say you don't have that information and suggest contacting the Examination Branch / HoD / Principal's office as appropriate.
- Be concise and specific: quote exact dates, amounts and rules. Mention the source document name.
- Treat the documents as the current reference material. Do not comment on whether dates or deadlines have passed.
- Never invent policies, fees or dates."""


def get_key():
    key = st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None
    return key or os.getenv("GEMINI_API_KEY")


def get_model():
    m = st.secrets.get("GEMINI_MODEL") if hasattr(st, "secrets") else None
    return m or os.getenv("GEMINI_MODEL") or MODEL


# ---------- UI ----------
st.title("🎓 BVRIT Hyderabad Assistant")
st.caption("Ask about exam fees, deadlines, exam rules, BH25 regulations, circulars and more.")

with st.sidebar:
    st.header("About")
    st.write("Answers come only from the official documents loaded in this app.")
    with st.expander("Documents loaded"):
        for f in sorted(DATA_DIR.glob("*.md")):
            st.write("•", f.read_text(encoding="utf-8").splitlines()[0].lstrip("# "))
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Try: *What is the exam fee for II year I sem regular?* · *What is the dress code in exams?* · *Are earphones allowed on campus?*")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use the last couple of turns for retrieval so follow-ups work
    recent = " ".join(m["content"] for m in st.session_state.messages[-3:] if m["role"] == "user")
    hits = search(recent)
    context = "\n\n---\n\n".join(f"[Source: {h['source']}]\n{h['text']}" for h in hits) or "(no relevant context found)"

    system = SYSTEM + f"\n\nCONTEXT:\n{context}"
    history = [
        types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])])
        for m in st.session_state.messages[-10:]
    ]

    with st.chat_message("assistant"):
        key = get_key()
        if not key:
            # No API key: fall back to showing the best-matching passages
            reply = "_No API key set, showing the most relevant passages instead:_\n\n" + context
            st.markdown(reply)
        else:
            try:
                client = genai.Client(api_key=key)
                stream = client.models.generate_content_stream(
                    model=get_model(),
                    contents=history,
                    config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=1000),
                )
                reply = st.write_stream(c.text for c in stream if c.text)
            except Exception as e:
                reply = f"Sorry, something went wrong: {e}"
                st.error(reply)
        if hits:
            with st.expander("Sources used"):
                for src in sorted({h["source"] for h in hits}):
                    st.write("•", src)
    st.session_state.messages.append({"role": "assistant", "content": reply})
