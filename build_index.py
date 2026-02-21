#!/usr/bin/env python3
"""
Build index:
- parse all emails in emails_dir
- for each email: LLM metadata summary (via OpenAI), paragraph-based chunking with sliding window (4 paragraphs, 1 overlap)
- create chunk objects with metadata
- embed all chunks with sentence-transformers all-MiniLM-L6-v2 and L2 normalize
- build FAISS IndexFlatIP and save index + metadata + vectors + BM25 corpus
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

from utils import parse_email_file, paragraph_split, sliding_window_paragraph_chunks, summarize_metadata_openai, simple_tokenize_for_bm25

def build_chunks_for_email(email_path: str, email_id: str, model_for_metadata: str = "gpt-3.5-turbo"):
    parsed = parse_email_file(email_path)
    subject = parsed['subject']
    frm = parsed['from']
    to = parsed['to']
    body = parsed['body']
    raw = parsed['raw']

    # 1) Get metadata summary via OpenAI. If not available, fallback used inside function.
    meta = summarize_metadata_openai(raw, model=model_for_metadata)

    # 2) Paragraph split
    paragraphs = paragraph_split(body)

    # 3) Sliding-window paragraph chunks with 4 paragraphs, 1 overlap (≈25%)
    window_paragraphs = 4
    overlap_paragraphs = 1
    body_chunks = sliding_window_paragraph_chunks(paragraphs, window_paragraphs, overlap_paragraphs)
    # If no paragraphs (very short body), fallback to whole body as one chunk
    if not body_chunks:
        body_chunks = [body]

    chunks = []
    for i, ctext in enumerate(body_chunks):
        chunk_id = f"{email_id}_c{i+1}"
        chunk = {
            "chunk_id": chunk_id,
            "email_file": os.path.basename(email_path),
            "subject": subject,
            "from": frm,
            "to": to,
            "metadata_summary": meta,
            "text": ctext
        }
        chunks.append(chunk)
    return chunks

def main(args):
    emails_dir = args.emails_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load all email files
    email_files = sorted([os.path.join(emails_dir, f) for f in os.listdir(emails_dir) if f.endswith('.txt')])
    print(f"Found {len(email_files)} email files")

    all_chunks = []
    # Build chunks (including metadata via OpenAI)
    for i, ef in enumerate(tqdm(email_files, desc="Parsing emails")):
        email_id = os.path.splitext(os.path.basename(ef))[0]
        chunks = build_chunks_for_email(ef, email_id)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    # Build BM25 corpus tokens
    corpus_texts = [c['text'] for c in all_chunks]
    tokenized_corpus = [simple_tokenize_for_bm25(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Compute embeddings in batches
    texts_for_embedding = [c['text'] for c in all_chunks]
    batch_size = 64
    embeddings = []
    for i in tqdm(range(0, len(texts_for_embedding), batch_size), desc="Embedding chunks"):
        batch = texts_for_embedding[i:i+batch_size]
        emb = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings).astype('float32')

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    print(f"Embeddings shape: {embeddings.shape}, dim={dim}")

    # Build FAISS IndexFlatIP (inner-product, works with normalized vectors -> cosine)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built, n = {index.ntotal}")

    # Save index and metadata
    index_path = os.path.join(out_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"Wrote FAISS index to {index_path}")

    # Save embeddings as well (for re-ranking reuse)
    np.save(os.path.join(out_dir, "vectors.npy"), embeddings)
    # Save metadata list
    with open(os.path.join(out_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Save BM25 tokenized corpus (to reconstruct/reload)
    with open(os.path.join(out_dir, "bm25_corpus.json"), 'w', encoding='utf-8') as f:
        json.dump(tokenized_corpus, f, ensure_ascii=False)

    print("Index build complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emails_dir", type=str, default="emails", help="Directory with email_*.txt files")
    parser.add_argument("--out_dir", type=str, default="faiss_index", help="Directory to write FAISS index and metadata")
    args = parser.parse_args()
    main(args)

