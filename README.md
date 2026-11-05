# 📄 PDF Question Answering Chatbot (RAG)

🔗 **Live Demo:** _add your Streamlit link here_

A Retrieval-Augmented Generation (RAG) app. Upload a PDF, ask questions in natural language, and get answers grounded in the document with **page-number citations**. Built with Python, Gemini (LLM and embeddings), FAISS and Streamlit.

## How it works

```
PDF upload -> text extraction (per page) -> chunking with overlap
          -> embeddings (Gemini) -> FAISS vector index
Question -> query embedding -> top-k similarity search -> relevant chunks
          -> prompt with context -> Gemini LLM -> answer + page citations
```

## Features
- Page-aware chunking, so every answer can cite its source pages
- Semantic search with FAISS (cosine similarity over normalised embeddings)
- Grounded answers: the model is instructed to answer only from the retrieved context and to say when the answer is not in the PDF, which reduces hallucination
- Chat history for follow-up questions and a "Sources" panel showing the retrieved passages
- Retry with backoff on API rate limits; clear error for scanned (image-only) PDFs
- Unit tests that run without an API key (fake embedder)

## Tech stack
Python, Streamlit, Gemini API (`google-genai`), FAISS, pypdf, NumPy, pytest

## Run locally
```bash
pip install -r requirements.txt
# get a free key at https://aistudio.google.com/apikey
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key
streamlit run app.py
```
Run the tests: `python -m pytest -q tests`

Model names can be changed in the sidebar or with the `GEMINI_MODEL` and `GEMINI_EMBED_MODEL` environment variables.

## Limitations
- Works on PDFs with selectable text (no OCR for scans)
- Answers are only as good as retrieval; very long tables or diagrams are not understood
- Free Gemini tier has rate limits

## Future improvements
- OCR for scanned PDFs, multi-PDF support, hybrid (keyword + semantic) search, re-ranking, answer-quality evaluation

## Author
Om Pawar, B.Sc. IT, University of Mumbai
