import hashlib
import os

import streamlit as st

import rag

st.set_page_config(page_title="PDF Question Answering Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Question Answering Chatbot (RAG)")
st.caption("Upload a PDF and ask questions. Answers come only from the document, with page citations.")


def get_secret_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY", "")


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    api_key = get_secret_key()
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password",
                                help="Free key from https://aistudio.google.com/apikey")
    chat_model = st.text_input("Chat model", value=rag.CHAT_MODEL)
    top_k = st.slider("Passages to retrieve", 2, 8, 4)
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if st.button("Clear chat"):
        st.session_state.messages = []

if not api_key:
    st.info("Enter your Gemini API key in the sidebar to start.")
    st.stop()

client = rag.make_gemini_client(api_key)


def embed_fn(texts, task_type, progress=None):
    return rag.gemini_embed(client, texts, task_type, progress)


# ---------- Build the index once per uploaded file ----------
if uploaded is None:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = hashlib.md5(file_bytes).hexdigest()

if st.session_state.get("file_hash") != file_hash:
    try:
        bar = st.progress(0.0, text="Reading the PDF and building the search index...")

        def on_progress(done, total):
            bar.progress(done / total, text=f"Embedding passages: {done}/{total} "
                                            "(free tier is rate-limited, so large PDFs take a minute)")

        st.session_state.index = rag.build_index(
            file_bytes, lambda t, k: embed_fn(t, k, on_progress), cache_dir=".cache"
        )
        bar.empty()
        st.session_state.file_hash = file_hash
        st.session_state.messages = []
    except Exception as exc:
        st.error(f"Could not process this PDF: {exc}")
        st.stop()

index = st.session_state.index
st.success(f"Indexed {len(index.chunks)} passages from {uploaded.name}")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for page, score, text in msg["sources"]:
                    st.markdown(f"**Page {page}** (similarity {score:.2f})")
                    st.caption(text[:400] + ("..." if len(text) > 400 else ""))

question = st.chat_input("Ask a question about the PDF")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the document..."):
                hits = rag.retrieve(index, question, embed_fn, k=top_k)
                answer = rag.answer_question(
                    client, question, hits, st.session_state.messages[:-1], model=chat_model
                )
            st.markdown(answer)
            sources = [(c.page, s, c.text) for c, s in hits]
            with st.expander("Sources"):
                for page, score, text in sources:
                    st.markdown(f"**Page {page}** (similarity {score:.2f})")
                    st.caption(text[:400] + ("..." if len(text) > 400 else ""))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")