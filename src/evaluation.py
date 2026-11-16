"""Evidence matching shared by every retriever and the agent evaluation.

Definitions (same as Day 6 `evaluation.py`, so numbers stay comparable):
- norm(): NFC, lowercase, every run of whitespace collapsed to one space. PDF
  extraction leaves line breaks inside sentences, so exact matching would count
  a correct chunk as a miss only because of a newline.
- Evidence hit: a chunk from a gold (source_id, page) whose normalized text
  contains the FULL normalized evidence of that gold entry. `gold` entries are
  alternatives: one is enough.
- Hit@1 / Evidence Hit@3: questions with a hit at rank 1 / within top-3, divided
  by the number of answerable questions.
- Context has evidence: the final context sent to the model contains a full
  gold evidence, either inside one chunk or across chunks of the gold page
  joined in page order (the context may come from several searches).
"""
import re
import unicodedata
from typing import Any


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).lower().split())


def is_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    text = norm(chunk.get("text") or "")
    return any(chunk.get("source_id") == g["source_id"] and chunk.get("page") == g["page"]
               and norm(g["evidence"]) in text for g in gold)


def _chunk_order(chunk_id: str) -> tuple:
    # Day 6 IDs look like "<doc>:p015:D03"; sort numerically on the trailing part.
    return tuple(int(n) for n in re.findall(r"\d+", chunk_id.rsplit(":", 1)[-1]))


def context_has_evidence(context_chunks: list[dict[str, Any]], gold: list[dict[str, Any]]) -> bool:
    if any(is_hit(c, gold) for c in context_chunks):
        return True
    for g in gold:
        same_page = [c for c in context_chunks if c.get("source_id") == g["source_id"] and c.get("page") == g["page"]]
        if len(same_page) >= 2:
            joined = " ".join(c["text"] for c in sorted(same_page, key=lambda c: _chunk_order(c["id"])))
            if norm(g["evidence"]) in norm(joined):
                return True
    return False


def gold_report(chunks: list[dict[str, Any]], questions: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Check labels against one index: which evidence fits one chunk, which is split, which is missing."""
    report = {"single_chunk": [], "split_across_chunks": [], "missing": []}
    for q in questions:
        if any(is_hit(c, q["gold"]) for c in chunks):
            report["single_chunk"].append(q["id"])
        elif context_has_evidence(chunks, q["gold"]):
            report["split_across_chunks"].append(q["id"])
        else:
            report["missing"].append(q["id"])
    return report
