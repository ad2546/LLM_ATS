import os
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np

load_dotenv()  # Load env variables from .env

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

EMBEDDING_MODEL = os.getenv('GEMINI_EMBED_MODEL', 'models/embedding-001')

def embed_text(text: str) -> np.ndarray:
    response = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="semantic_similarity"
    )
    embedding = response['embedding'] if 'embedding' in response else response['data'][0]['embedding']
    return np.array(embedding, dtype='float32')

# Example usage:
text = "Machine learning is transforming how we solve problems."
vector = embed_text(text)
print("Embedding shape:", vector.shape)
print("Sample vector values:", vector[:10])