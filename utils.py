"""
Utility functions for parsing emails, chunking, summarizing metadata via OpenAI,
and small helpers used by build_index.py and run_query.py
"""

import os
import re
import json
import math
from typing import Dict, Any, List, Tuple, Optional
import openai
from tqdm import tqdm

# Simple email parser — expects the format shown in your samples
def parse_email_file(path: str) -> Dict[str, str]:
    """
    Returns dict with keys: subject, from, to, body, raw
    """
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    # Extract Subject, From, To using regex (robust to whitespace)
    subject_match = re.search(r"^Subject:\s*(.+)$", raw, flags=re.MULTILINE)
    from_match = re.search(r"^From:\s*(.+)$", raw, flags=re.MULTILINE)
    to_match = re.search(r"^To:\s*(.+)$", raw, flags=re.MULTILINE)

    subject = subject_match.group(1).strip() if subject_match else ""
    frm = from_match.group(1).strip() if from_match else ""
    to = to_match.group(1).strip() if to_match else ""

    # Body: attempt to find first blank line after To header then take remainder
    body_split = re.split(r"\n\s*\n", raw, maxsplit=1)
    if len(body_split) > 1:
        body = body_split[1].strip()
    else:
        # fallback: remove headers area
        body = re.sub(r"^.*?\n\n", "", raw, count=1, flags=re.DOTALL).strip()

    return {"subject": subject, "from": frm, "to": to, "body": body, "raw": raw}


def paragraph_split(text: str) -> List[str]:
    """
    Split text into paragraphs. Keep paragraphs that have at least one word.
    """
    paras = [p.strip() for p in re.split(r'\n{1,}', text) if p.strip()]
    return paras


def sliding_window_paragraph_chunks(paragraphs: List[str],
                                    window_paragraphs: int = 4,
                                    overlap_paragraphs: int = 1) -> List[str]:
    """
    Build chunks as sliding windows of paragraphs.
    window_paragraphs: number of paragraphs in a chunk (we use 4)
    overlap_paragraphs: number of paragraphs to overlap (we use 1 to approximate 25%)
    """
    if not paragraphs:
        return []
    step = max(1, window_paragraphs - overlap_paragraphs)
    chunks = []
    for i in range(0, len(paragraphs), step):
        window = paragraphs[i:i + window_paragraphs]
        if not window:
            continue
        chunks.append("\n\n".join(window))
        # stop if we've reached the end
        if i + window_paragraphs >= len(paragraphs):
            break
    return chunks


# OpenAI-based metadata summarizer. Expects OPENAI_API_KEY in env.
def summarize_metadata_openai(email_text: str, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Calls OpenAI chat completion to extract a small JSON metadata summary.
    Returns a dictionary with fields: short_summary, topics, action_items, urgency, dates (free text)
    If the API call fails or API key missing, returns a fallback heuristic dict.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return summarize_metadata_fallback(email_text)

    openai.api_key = api_key

    system = (
        "You are an assistant that extracts concise metadata from an email. "
        "Return only JSON with these fields: short_summary (1-2 sentences), "
        "topics (list of short tags), action_items (list of short action items or empty list), "
        "urgency (low/medium/high/unknown), dates (list of date strings mentioned or empty), "
        "people_mentioned (list of names or emails if present)."
    )

    prompt = (
        "Email text:\n\n"
        + email_text
        + "\n\nExtract the metadata as described. Output ONLY valid JSON."
    )

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        text = resp["choices"][0]["message"]["content"].strip()

        # Try to locate JSON in response
        json_match = re.search(r"(\{[\s\S]*\})", text)
        if json_match:
            parsed = json.loads(json_match.group(1))
            return parsed
        else:
            # If not JSON, fallback
            return summarize_metadata_fallback(email_text)
    except Exception as e:
        print(f"[WARN] OpenAI metadata summarization failed: {e}")
        return summarize_metadata_fallback(email_text)


def summarize_metadata_fallback(email_text: str) -> Dict[str, Any]:
    """
    Lightweight heuristic fallback when OpenAI is not available.
    """
    lines = [l.strip() for l in email_text.splitlines() if l.strip()]
    # short summary: first two sentences of body or first line
    first = lines[0] if lines else ""
    # topics by simple keyword matching
    topics = []
    kws = {
        "training": ["train", "training", "workshop", "development"],
        "meeting": ["meeting", "schedule", "agenda"],
        "budget": ["budget", "cost", "expense"],
        "technical": ["bug", "issue", "error", "technical", "production"],
        "client": ["client", "customer"],
        "deadline": ["deadline", "extension", "due"],
        "vendor": ["vendor", "proposal", "contract"],
        "review": ["review", "performance"]
    }
    text_lower = email_text.lower()
    for t, words in kws.items():
        for w in words:
            if w in text_lower:
                topics.append(t)
                break

    action_items = []
    # naive: look for sentences containing "please", "let me", "request", "please let me"
    for sent in re.split(r'(?<=[.!?])\s+', email_text):
        if re.search(r'\b(please|let me|request|schedule|register|attach)\b', sent, flags=re.I):
            action_items.append(sent.strip())
            if len(action_items) >= 3:
                break

    urgency = "unknown"
    if re.search(r'\burgent\b|\bASAP\b', email_text, flags=re.I):
        urgency = "high"
    elif re.search(r'\bnext week\b|\btomorrow\b|\bsoon\b', email_text, flags=re.I):
        urgency = "medium"

    people = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)|([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', email_text)
    people_flat = []
    for p in people:
        # p is tuple; pick non-empty
        item = next((x for x in p if x), None)
        if item and item not in people_flat:
            people_flat.append(item)

    return {
        "short_summary": first[:300],
        "topics": topics,
        "action_items": action_items,
        "urgency": urgency,
        "dates": [],
        "people_mentioned": people_flat
    }


def simple_tokenize_for_bm25(text: str) -> List[str]:
    # Lowercase, split on non-word chars
    tokens = re.findall(r"\w+", text.lower())
    return tokens


def detect_metadata_filters(query: str) -> Dict[str, str]:
    """
    Simple parsing for filters in queries:
    Examples:
      'from:tara woods training' or 'sender:tara.woods@enterprise.com'
      'subject:budget approval' or 'subject:Training'
    Returns dict like {'from': 'tara woods', 'subject': 'training'}
    """
    filters = {}
    # from:
    fm = re.search(r'(?:from|sender)\s*:\s*([^\n,;]+)', query, flags=re.I)
    if fm:
        filters['from'] = fm.group(1).strip()
    sb = re.search(r'subject\s*:\s*([^\n,;]+)', query, flags=re.I)
    if sb:
        filters['subject'] = sb.group(1).strip()
    return filters

