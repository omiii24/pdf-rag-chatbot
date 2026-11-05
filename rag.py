"""Core RAG pipeline: PDF -> chunks -> embeddings -> vector index -> retrieval -> grounded answer.

The embedding function is passed in, so the pipeline can be unit-tested without an API key.
"""
import hashlib
import io
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from pypdf import PdfReader

try:  # FAISS is optional: fall back to plain numpy search if it is not installed
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

# Model names change often, so they can be overridden with environment variables / secrets.
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBED_DIM = 768
EMBED_BATCH = 20          # texts per API call
EMBED_PER_MIN = 80        # stay under the free-tier limit (100 embeddings per minute)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about a PDF document. "
    "Use ONLY the context excerpts provided. Each excerpt is labelled with its page number. "
    "Cite the pages you used like (p. 3). "
    "If the answer is not in the context, reply exactly: "
    "\"I couldn't find that in the document.\" Do not use outside knowledge."
)


@dataclass
class Chunk:
    text: str
    page: int


# ---------- 1. Extraction and chunking ----------
def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split each page into overlapping chunks, keeping the page number for citations."""
    chunks = []
    for page_no, text in pages:
        text = " ".join(text.split())  # normalise whitespace
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):  # try not to cut a word in half
                cut = text.rfind(" ", start + size // 2, end)
                if cut != -1:
                    end = cut
            piece = text[start:end].strip()
            if len(piece) >= 30:
                chunks.append(Chunk(piece, page_no))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


# ---------- 2. Embeddings (Gemini) ----------
def _retry_delay(msg: str) -> float:
    """Read the wait time Google asks for, e.g. 'Please retry in 46.9s' or retryDelay: '46s'."""
    m = re.search(r"retry in ([\d.]+)s", msg) or re.search(r"retryDelay'?:\s*'?(\d+)s", msg)
    return float(m.group(1)) if m else 0.0


def _with_retry(fn, tries: int = 5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # retry only on rate-limit / temporary errors
            msg = str(exc)
            temporary = any(s in msg for s in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))
            if not temporary or attempt == tries - 1:
                raise
            time.sleep(max(_retry_delay(msg) + 1, 2 ** attempt * 2))


_window = deque()  # (timestamp, number_of_texts) for the last 60 seconds


def _throttle(n: int, per_minute: int = EMBED_PER_MIN) -> None:
    """Block until sending n more texts keeps us under the per-minute embedding limit."""
    while True:
        now = time.time()
        while _window and now - _window[0][0] > 60:
            _window.popleft()
        used = sum(c for _, c in _window)
        if not _window or used + n <= per_minute:
            _window.append((now, n))
            return
        time.sleep(max(60 - (now - _window[0][0]), 0) + 0.5)


def make_gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def gemini_embed(client, texts: list[str], task_type: str, progress=None) -> np.ndarray:
    """Embed texts with Gemini. task_type: RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY.
    progress(done, total) is called after every batch (used for the progress bar)."""
    from google.genai import types
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        _throttle(len(batch))
        resp = _with_retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
        ))
        vectors.extend(e.values for e in resp.embeddings)
        if progress:
            progress(min(i + EMBED_BATCH, len(texts)), len(texts))
    return _normalise(np.array(vectors, dtype="float32"))


def _normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype("float32")


# ---------- 3. Vector index and retrieval ----------
class PdfIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors
        if faiss is not None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product = cosine (normalised)
            self.index.add(vectors)
        else:
            self.index = None

    def search(self, query_vec: np.ndarray, k: int = 4):
        k = min(k, len(self.chunks))
        query_vec = query_vec.reshape(1, -1).astype("float32")
        if self.index is not None:
            scores, ids = self.index.search(query_vec, k)
            scores, ids = scores[0], ids[0]
        else:
            sims = (self.vectors @ query_vec.T).ravel()
            ids = np.argsort(-sims)[:k]
            scores = sims[ids]
        return [(self.chunks[i], float(s)) for i, s in zip(ids, scores) if i != -1]


def _cache_key(file_bytes: bytes) -> str:
    h = hashlib.sha256(file_bytes)
    h.update(f"{EMBED_MODEL}|{EMBED_DIM}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    return h.hexdigest()[:24]


def build_index(file_bytes: bytes, embed_fn, cache_dir: str | None = None) -> PdfIndex:
    """embed_fn(texts, task_type) -> normalised np.ndarray.
    With cache_dir, embeddings are saved so the same PDF never uses API quota twice."""
    paths = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _cache_key(file_bytes)
        paths = (os.path.join(cache_dir, key + ".npy"), os.path.join(cache_dir, key + ".json"))
        if all(os.path.exists(p) for p in paths):
            vectors = np.load(paths[0])
            with open(paths[1], encoding="utf-8") as f:
                chunks = [Chunk(**c) for c in json.load(f)]
            return PdfIndex(chunks, vectors)

    pages = extract_pages(file_bytes)
    if not pages:
        raise ValueError(
            "No selectable text found in this PDF. It may be a scanned document (image only)."
        )
    chunks = chunk_pages(pages)
    vectors = embed_fn([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
    if paths:
        np.save(paths[0], vectors)
        with open(paths[1], "w", encoding="utf-8") as f:
            json.dump([{"text": c.text, "page": c.page} for c in chunks], f)
    return PdfIndex(chunks, vectors)


def retrieve(index: PdfIndex, question: str, embed_fn, k: int = 4):
    qvec = embed_fn([question], "RETRIEVAL_QUERY")[0]
    return index.search(qvec, k)


# ---------- 4. Grounded answer generation ----------
def build_prompt(question: str, hits, history=None) -> str:
    context = "\n\n".join(f"[Page {c.page}]\n{c.text}" for c, _ in hits)
    past = ""
    if history:
        lines = [f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:]]
        past = "Recent conversation:\n" + "\n".join(lines) + "\n\n"
    return f"{past}Context from the PDF:\n{context}\n\nQuestion: {question}"


def answer_question(client, question: str, hits, history=None, model: str = CHAT_MODEL) -> str:
    from google.genai import types
    prompt = build_prompt(question, hits, history)
    resp = _with_retry(lambda: client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
    ))
    return resp.text or "I couldn't find that in the document."