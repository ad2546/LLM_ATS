import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import faiss
import mysql.connector
from config import settings
import google.generativeai as genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

from utils.pdf_utils import read_pdf_content

# Configure embedding model
EMBEDDING_MODEL = os.getenv('GEMINI_EMBED_MODEL', 'embed-gecko')

# Utility to call Gemini embeddings

def embed_text(text: str) -> np.ndarray:
    """
    Returns an L2-normalized embedding vector for the given text.
    """
    resp = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="semantic_similarity"
    )
    print(f"Embedding response: {resp}")
    embedding = resp.get('embedding')
    print(f"Vector shape: {np.array(embedding).shape}")
    if embedding is None or not embedding:
        raise ValueError(f"Failed to get embedding for: {text}\nResponse: {resp}")
    vector = np.array(embedding, dtype='float32')
    if vector.shape == () or vector.shape[0] == 0:
        raise ValueError(f"Empty embedding returned for input: '{text}'")
    # Normalize for cosine similarity
    vector_2d = vector.reshape(1, -1)
    faiss.normalize_L2(vector_2d)
    vector = vector_2d.flatten()
    return vector

class CategoryIndex:
    """
    FAISS index for category name embeddings.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.id_map = []  # maps index positions to category_id

    def add(self, category_id: int, vector: np.ndarray):
        self.index.add(vector[np.newaxis, :])
        self.id_map.append(category_id)

    def search(self, vector: np.ndarray, k: int = 2):
        """
        Returns list of (category_id, score) for top k.
        """
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(vector[np.newaxis, :], k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))
        return results

# Load CategoryIndex from DB

def load_category_index() -> CategoryIndex:
    conn = mysql.connector.connect(**{
        'host': os.getenv('MYSQL_HOST', 'localhost'),
        'user': os.getenv('MYSQL_USER', 'root'),
        'password': os.getenv('MYSQL_PASSWORD', ''),
        'database': os.getenv('MYSQL_DATABASE', 'LLM_Resume')
    })
    cursor = conn.cursor()
    cursor.execute("SELECT category_id, name FROM category")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return CategoryIndex(dimension=0)
    # Embed first name to set dimension
    first_vec = embed_text(rows[0][1])
    dim = first_vec.shape[0]
    idx = CategoryIndex(dim)
    for cid, name in rows:
        vec = embed_text(name)
        idx.add(cid, vec)
    return idx

_category_index = None
def get_category_index():
    global _category_index
    if _category_index is None:
        try:
            _category_index = load_category_index()
        except Exception:
            _category_index = None
    return _category_index

# --------- Resume Index ----------
class ResumeIndex:
    """
    FAISS index for resume embeddings.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.id_map = []  # maps index positions to resume file paths

    def add(self, file_path: str, vector: np.ndarray):
        self.index.add(vector[np.newaxis, :])
        self.id_map.append(file_path)

    def search(self, vector: np.ndarray, k: int = 5):
        """
        Returns list of (file_path, score) for top k resumes.
        """
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(vector[np.newaxis, :], k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < len(self.id_map):
                results.append((self.id_map[idx], float(score)))
        return results

# Load ResumeIndex from local resumes directory

def load_resume_index() -> ResumeIndex:
    resumes_dir = os.path.join(os.path.dirname(__file__), '../resumes')
    pdf_paths = []
    for root, _, files in os.walk(resumes_dir):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(root, f))
    if not pdf_paths:
        return ResumeIndex(dimension=0)
    # Embed first resume to set dimension
    try:
        with open(pdf_paths[0], 'rb') as f:
            text = read_pdf_content(f.read())
        first_vec = embed_text(text)
        dim = first_vec.shape[0]
    except Exception:
        return ResumeIndex(dimension=0)
    idx = ResumeIndex(dim)
    for path in pdf_paths:
        try:
            with open(path, 'rb') as f:
                text = read_pdf_content(f.read())
            vec = embed_text(text)
            idx.add(path, vec)
        except Exception:
            continue
    return idx

_resume_index = None
def get_resume_index():
    global _resume_index
    if _resume_index is None:
        try:
            _resume_index = load_resume_index()
        except Exception:
            _resume_index = None
    return _resume_index
