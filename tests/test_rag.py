"""Tests run without any API key by using a fake, deterministic embedder."""
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import rag  # noqa: E402

VOCAB = ["python", "gradient", "descent", "photosynthesis", "chlorophyll", "invoice", "payment"]


def fake_embed(texts, task_type):
    """Bag-of-keywords vectors: enough to check that retrieval finds the right chunk."""
    out = []
    for t in texts:
        low = t.lower()
        vec = np.array([low.count(w) for w in VOCAB], dtype="float32") + 0.01
        out.append(vec)
    return rag._normalise(np.array(out))


def make_pdf(pages):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_chunking_keeps_page_numbers_and_overlap():
    pages = [(1, "word " * 500), (2, "short page with enough characters to be kept here")]
    chunks = rag.chunk_pages(pages, size=300, overlap=50)
    assert chunks and all(len(c.text) <= 300 for c in chunks)
    assert {c.page for c in chunks} == {1, 2}


def test_retrieval_returns_the_right_page():
    pdf = make_pdf([
        "Gradient descent is an optimisation method used to train python models.",
        "Photosynthesis happens when chlorophyll absorbs light in plants.",
        "This invoice is due and payment must be made within thirty days.",
    ])
    index = rag.build_index(pdf, fake_embed)
    hits = rag.retrieve(index, "How does chlorophyll help photosynthesis?", fake_embed, k=1)
    assert hits[0][0].page == 2


def test_prompt_contains_pages_and_question():
    pdf = make_pdf(["Gradient descent trains python models and is widely used today."])
    index = rag.build_index(pdf, fake_embed)
    hits = rag.retrieve(index, "what is gradient descent", fake_embed, k=1)
    prompt = rag.build_prompt("what is gradient descent", hits)
    assert "[Page 1]" in prompt and "what is gradient descent" in prompt


def test_empty_pdf_raises_clear_error():
    pdf = make_pdf([""])
    try:
        rag.build_index(pdf, fake_embed)
    except ValueError as e:
        assert "No selectable text" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_retry_delay_is_read_from_google_error():
    assert rag._retry_delay("429 ... Please retry in 46.914973965s.") == 46.914973965
    assert rag._retry_delay("{'retryDelay': '46s'}") == 46.0
    assert rag._retry_delay("some other error") == 0.0


def test_throttle_waits_when_limit_reached(monkeypatch):
    clock = {"t": 1000.0}
    sleeps = []
    monkeypatch.setattr(rag.time, "time", lambda: clock["t"])
    monkeypatch.setattr(rag.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__("t", clock["t"] + s)))
    rag._window.clear()
    rag._throttle(60, per_minute=80)
    rag._throttle(15, per_minute=80)   # 75 used, fine
    assert sleeps == []
    rag._throttle(20, per_minute=80)   # would be 95 -> must wait for the window to clear
    assert sleeps and sleeps[0] > 50


def test_embeddings_are_cached_on_disk(tmp_path):
    pdf = make_pdf(["Gradient descent trains python models and is widely used today."])
    calls = []

    def counting_embed(texts, task_type):
        calls.append(len(texts))
        return fake_embed(texts, task_type)

    rag.build_index(pdf, counting_embed, cache_dir=str(tmp_path))
    rag.build_index(pdf, counting_embed, cache_dir=str(tmp_path))
    assert len(calls) == 1  # second call came from the cache