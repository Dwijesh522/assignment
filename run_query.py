#!/usr/bin/env python3
"""
Run queries against the built FAISS + BM25 index, then call OpenAI to generate answers.

Retrieval:
- parse optional metadata filters in query (from:, subject:)
- BM25 top_n (50) + FAISS top_n (100) → union candidates
- compute combined score = 0.6 * emb_sim + 0.4 * bm25_norm
- return top_k (default 5) chunks

Generation:
- constructs a short instruction + enumerated retrieved chunks with chunk IDs
- calls OpenAI ChatCompletion (gpt-3.5-turbo) with temperature 0.0
- instructs model to only answer from provided chunks and cite chunk IDs
"""

import os
import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import openai
from utils import simple_tokenize_for_bm25, detect_metadata_filters

def load_index_and_data(index_dir: str):
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    vectors = np.load(os.path.join(index_dir, "vectors.npy"))
    with open(os.path.join(index_dir, "metadata.json"), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    with open(os.path.join(index_dir, "bm25_corpus.json"), 'r', encoding='utf-8') as f:
        tokenized_corpus = json.load(f)
    bm25 = BM25Okapi(tokenized_corpus)
    return index, vectors, metadata, bm25

def retrieve(query: str, index, vectors: np.ndarray, metadata: list, bm25: BM25Okapi,
             embed_model: SentenceTransformer, top_k: int = 5):
    # detect filters
    filters = detect_metadata_filters(query)
    # prepare BM25
    q_tokens = simple_tokenize_for_bm25(query)
    bm_scores = bm25.get_scores(q_tokens)
    # BM25 top ids
    top_n_bm = 50
    bm_idx_sorted = np.argsort(-bm_scores)[:top_n_bm]
    bm_idx_set = set(int(i) for i in bm_idx_sorted if bm_scores[int(i)] > 0)

    # embedding query
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0].astype('float32')
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)

    # FAISS search
    top_n_faiss = 100
    D, I = index.search(np.array([q_emb]).astype('float32'), top_n_faiss)
    faiss_idx = [int(i) for i in I[0] if i != -1]

    candidate_ids = list(set(list(bm_idx_set) + faiss_idx))
    if not candidate_ids:
        # fallback: use faiss results even if no bm scores
        candidate_ids = faiss_idx[:top_k]

    # compute scores for candidates
    emb_candidates = vectors[candidate_ids]
    emb_sims = (emb_candidates @ q_emb).reshape(-1)  # since normalized -> cosine

    # normalize bm25 scores for the candidates
    bm_scores_cand = np.array([bm_scores[i] for i in candidate_ids], dtype=float)
    max_bm = bm_scores_cand.max() if bm_scores_cand.size > 0 else 0.0
    bm_norm = bm_scores_cand / (max_bm + 1e-10)

    # combined score: weights (embedding 0.6, bm25 0.4)
    combined = 0.6 * emb_sims + 0.4 * bm_norm

    # apply metadata filtering if filters provided
    if filters:
        filtered = []
        filtered_scores = []
        filtered_ids = []
        for idx, score in zip(candidate_ids, combined):
            md = metadata[idx]
            matches = True
            if 'from' in filters:
                if filters['from'].lower() not in md.get('from', '').lower():
                    matches = False
            if 'subject' in filters:
                if filters['subject'].lower() not in md.get('subject', '').lower():
                    matches = False
            if matches:
                filtered.append((idx, score))
        if filtered:
            candidate_ids = [x[0] for x in filtered]
            combined = np.array([x[1] for x in filtered])
        else:
            # if no match after filtering, keep original candidates but warn
            print("[INFO] No candidates matched metadata filters — ignoring filters for this query.")

    # select top_k by combined score
    order = np.argsort(-combined)[:top_k]
    selected = []
    for o in order:
        idx = candidate_ids[o]
        selected.append({
            "idx": int(idx),
            "chunk_id": metadata[int(idx)]["chunk_id"],
            "subject": metadata[int(idx)]["subject"],
            "from": metadata[int(idx)]["from"],
            "text": metadata[int(idx)]["text"],
            "metadata_summary": metadata[int(idx)].get("metadata_summary", {})
        })
    return selected

def format_prompt(retrieved_chunks: list, question: str) -> str:
    """
    Build a prompt with enumerated chunks and a final instruction.
    """
    header = (
        "You are an assistant that answers questions using ONLY the provided chunks below. "
        "Each chunk has an ID. When you state facts, include the chunk ID(s) that support them in square brackets.",
        "If the information is not present in the chunks, answer 'I don't know.' Do not hallucinate."
    )
    parts = []
    parts.append("Context chunks:")
    for i, c in enumerate(retrieved_chunks, start=1):
        parts.append(f"### Chunk {i} | ID: {c['chunk_id']}\nSubject: {c['subject']}\nFrom: {c['from']}\nMetadata summary: {json.dumps(c['metadata_summary'], ensure_ascii=False)}\n\n{c['text']}\n")
    parts.append("\nInstructions: Answer the question using only the information in the chunks above. Always cite chunk IDs in square brackets after the sentence where you used that chunk. If not answerable, say \"I don't know.\" Keep answer concise (3-6 sentences).")
    parts.append(f"\nQuestion: {question}")
    prompt = "\n\n".join(parts)
    return prompt

def call_openai_chat(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 300, temperature: float = 0.0):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Set it before running generation.")
    openai.api_key = api_key
    messages = [
        {"role": "system", "content": "You are a helpful assistant that only uses the provided chunks to answer."},
        {"role": "user", "content": prompt}
    ]
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp['choices'][0]['message']['content'].strip()

def interactive_loop(index_dir: str):
    print("Loading index and data...")
    index, vectors, metadata, bm25 = load_index_and_data(index_dir)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Ready. Enter queries (type 'exit' to quit).")
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            continue
        if q.lower() in ('exit', 'quit'):
            break
        retrieved = retrieve(q, index, vectors, metadata, bm25, embed_model, top_k=5)
        print(f"\nRetrieved {len(retrieved)} chunks:")
        for r in retrieved:
            print(f"- {r['chunk_id']} | Subject: {r['subject']} | From: {r['from']}")

        prompt = format_prompt(retrieved, q)
        print("\nCalling OpenAI to generate answer...")
        try:
            ans = call_openai_chat(prompt)
        except Exception as e:
            print(f"[ERROR] OpenAI call failed: {e}")
            ans = "ERROR: OpenAI generation failed. Check OPENAI_API_KEY and network."
        print("\n========== ANSWER ==========\n")
        print(ans)
        print("\n============================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=str, default="faiss_index", help="Directory where index and metadata live")
    args = parser.parse_args()
    interactive_loop(args.index_dir)

