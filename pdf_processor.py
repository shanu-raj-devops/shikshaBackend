import fitz
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import re

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_chapters(pdf_path):
    doc = fitz.open(pdf_path)
    chapters = []
    current_chapter = {"title": "Introduction", "content": "", "pages": []}

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if re.match(r"^(Chapter|CHAPTER|Unit|UNIT)\s+\d+", line) or re.match(r"^\d+\.\s+[A-Z]", line):
                if current_chapter["content"].strip():
                    chapters.append(current_chapter)
                current_chapter = {"title": line, "content": "", "pages": []}

        current_chapter["content"] += text + "\n"
        current_chapter["pages"].append(page_num + 1)

    if current_chapter["content"].strip():
        chapters.append(current_chapter)

    doc.close()
    return chapters


def build_faiss_index(chapters, index_path, meta_path):
    texts = [ch["title"] + " " + ch["content"][:500] for ch in chapters]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chapters, f)

    return len(chapters)


def load_index(index_path, meta_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chapters = pickle.load(f)
    return index, chapters
