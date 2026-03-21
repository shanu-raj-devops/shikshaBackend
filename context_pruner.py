import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

MAX_CONTEXT_CHARS = 4000
TOP_K_CHAPTERS = 3

def prune_context(query, index, chapters, top_k=TOP_K_CHAPTERS):
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    selected = []
    total_chars = 0

    for idx in indices[0]:
        if idx < len(chapters):
            chapter = chapters[idx]
            content = chapter["content"][:MAX_CONTEXT_CHARS]
            if total_chars + len(content) <= MAX_CONTEXT_CHARS * top_k:
                selected.append({
                    "title": chapter["title"],
                    "content": content,
                    "pages": chapter["pages"]
                })
                total_chars += len(content)

    return selected


def build_prompt(query, pruned_chapters):
    context_text = ""
    for ch in pruned_chapters:
        context_text += f"[{ch['title']}]\n{ch['content']}\n\n"

    prompt = f"""You are a helpful tutor for Indian school students. Use only the textbook content below to answer the question clearly and simply.

Textbook Content:
{context_text}

Student Question: {query}

Answer in simple English. If the answer is not in the textbook, say so honestly."""

    return prompt


def calculate_tokens_saved(all_chapters, pruned_chapters):
    total_chars = sum(len(ch["content"]) for ch in all_chapters)
    pruned_chars = sum(len(ch["content"]) for ch in pruned_chapters)
    reduction_percent = round((1 - pruned_chars / total_chars) * 100, 1) if total_chars > 0 else 0
    return {
        "total_chars": total_chars,
        "pruned_chars": pruned_chars,
        "reduction_percent": reduction_percent,
        "chapters_used": len(pruned_chapters),
        "total_chapters": len(all_chapters)
    }
