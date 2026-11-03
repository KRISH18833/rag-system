import streamlit as st
from query import rag_answer, plain_answer, retrieve

@st.cache_data(show_spinner=False)
def cached_rag_answer(question):
    return rag_answer(question)

@st.cache_data(show_spinner=False)
def cached_plain_answer(question):
    return plain_answer(question)

st.set_page_config(page_title="Cloud Notes RAG", page_icon="☁️", layout="wide")

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Full-page gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.37);
    color: #fff;
}
.glass-sm {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 14px;
    padding: 1rem 1.4rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    color: #fff;
    height: 100%;
}

/* Typography */
h1, h2, h3, h4, p, label, span, div { color: #fff !important; }

/* Input */
[data-testid="stTextInput"] > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 12px !important;
    color: #fff !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stTextInput"] > div > div > input::placeholder { color: rgba(255,255,255,0.45) !important; }
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: rgba(130,100,255,0.7) !important;
    box-shadow: 0 0 0 2px rgba(130,100,255,0.25) !important;
}

/* Buttons */
.stButton > button {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(130,100,255,0.35) !important;
    border-color: rgba(130,100,255,0.6) !important;
}

/* Radio / segmented selector */
[data-testid="stRadio"] > div { gap: 0.5rem; }
[data-testid="stRadio"] label {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 10px;
    padding: 0.3rem 0.9rem !important;
    cursor: pointer;
    transition: background 0.2s;
}
[data-testid="stRadio"] label:hover { background: rgba(130,100,255,0.25); }

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
}

/* Source chip */
.chip {
    display: inline-block;
    background: rgba(130,100,255,0.3);
    border: 1px solid rgba(130,100,255,0.5);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    margin: 3px 4px 3px 0;
    color: #e0d7ff;
}

/* Error banner */
.err-box {
    background: rgba(255,80,80,0.15);
    border: 1px solid rgba(255,80,80,0.4);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #ffbbbb !important;
}

/* Chunk card */
.chunk-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.8rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
    color: #ddd;
}
.chunk-meta { color: rgba(180,170,255,0.9); font-size: 0.8rem; margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "question" not in st.session_state:
    st.session_state.question = ""

EXAMPLE_QUESTIONS = [
    "In the W&B versioning example, which command loads version 2 of the model, and what file name is used?",
    "In the I/O interference example, what mean bandwidth does the STREAM benchmark need versus disk writes?",
    "How is elasticity expressed as an equation in the notes?",
]

# ── Header card ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="glass" style="text-align:center; padding: 2rem 2rem 1.4rem;">
  <h1 style="font-size:2.4rem; margin:0 0 0.3rem;">☁️ Cloud Notes RAG</h1>
  <p style="color:rgba(255,255,255,0.65) !important; font-size:1.05rem; margin:0;">
    Retrieval-Augmented Generation over Cloud Computing for ML course notes &nbsp;·&nbsp; Chroma + Gemini
  </p>
</div>
""", unsafe_allow_html=True)

# ── Example question buttons ───────────────────────────────────────────────────
st.markdown("<p style='color:rgba(255,255,255,0.55);font-size:0.85rem;margin-bottom:0.4rem;'>Try an example:</p>", unsafe_allow_html=True)
ecols = st.columns(3)
labels = ["W&B version 2", "STREAM benchmark", "Elasticity equation"]
for col, label, q in zip(ecols, labels, EXAMPLE_QUESTIONS):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.question = q

# ── Input row ─────────────────────────────────────────────────────────────────
icol, bcol = st.columns([5, 1])
with icol:
    question = st.text_input(
        "question_input",
        value=st.session_state.question,
        placeholder="Ask anything about your Cloud Computing notes…",
        label_visibility="collapsed",
    )
with bcol:
    ask = st.button("Ask ✦", use_container_width=True)

# ── Mode selector ──────────────────────────────────────────────────────────────
mode = st.radio("Mode", ["RAG", "Plain LLM", "Compare"], horizontal=True, label_visibility="collapsed")

st.markdown("<hr style='border-color:rgba(255,255,255,0.1);margin:0.5rem 0 1rem;'>", unsafe_allow_html=True)

# ── Helper: source chips ───────────────────────────────────────────────────────
def render_chips(sources):
    chips = "".join(f'<span class="chip">{s}</span>' for s in sources)
    st.markdown(f"<div style='margin-top:0.7rem;'>{chips}</div>", unsafe_allow_html=True)

# ── Helper: retrieved chunks expander ─────────────────────────────────────────
def render_chunks(chunks):
    with st.expander("📄 Retrieved chunks"):
        for i, c in enumerate(chunks, 1):
            text_preview = c["text"][:400].replace("\n", " ")
            st.markdown(f"""
            <div class="chunk-card">
              <div class="chunk-meta">#{i} &nbsp;|&nbsp; {c['source']} &nbsp;·&nbsp; page {c['page']} &nbsp;·&nbsp; distance {c['distance']:.4f}</div>
              {text_preview}{'…' if len(c['text']) > 400 else ''}
            </div>
            """, unsafe_allow_html=True)

# ── Main answer area ───────────────────────────────────────────────────────────
if ask and question.strip():
    st.session_state.question = question

    if mode == "RAG":
        with st.spinner("Retrieving and generating…"):
            try:
                answer, sources = cached_rag_answer(question)
                chunks = retrieve(question)
                st.markdown(f'<div class="glass">{answer}</div>', unsafe_allow_html=True)
                render_chips(sources)
                render_chunks(chunks)
            except Exception as e:
                st.markdown(f'<div class="err-box">⚠️ Something went wrong: <code>{e}</code><br>The API may be busy — please retry in a moment.</div>', unsafe_allow_html=True)

    elif mode == "Plain LLM":
        with st.spinner("Generating…"):
            try:
                answer = cached_plain_answer(question)
                st.markdown(f'<div class="glass">{answer}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="err-box">⚠️ Something went wrong: <code>{e}</code><br>The API may be busy — please retry in a moment.</div>', unsafe_allow_html=True)

    elif mode == "Compare":
        with st.spinner("Running both — this takes a moment…"):
            try:
                plain = cached_plain_answer(question)
                rag, sources = cached_rag_answer(question)
                chunks = retrieve(question)

                lcol, rcol = st.columns(2)
                with lcol:
                    st.markdown(f"""
                    <div class="glass-sm">
                      <h4 style="margin:0 0 0.8rem; color:rgba(255,180,100,1) !important;">🌐 Without RAG</h4>
                      {plain}
                    </div>
                    """, unsafe_allow_html=True)

                with rcol:
                    chips = "".join(f'<span class="chip">{s}</span>' for s in sources)
                    st.markdown(f"""
                    <div class="glass-sm">
                      <h4 style="margin:0 0 0.8rem; color:rgba(130,220,150,1) !important;">✦ With RAG</h4>
                      {rag}
                      <div style="margin-top:0.9rem;">{chips}</div>
                    </div>
                    """, unsafe_allow_html=True)

                render_chunks(chunks)

            except Exception as e:
                st.markdown(f'<div class="err-box">⚠️ Something went wrong: <code>{e}</code><br>The API may be busy — please retry in a moment.</div>', unsafe_allow_html=True)

elif ask and not question.strip():
    st.markdown('<div class="err-box">Please enter a question first.</div>', unsafe_allow_html=True)

