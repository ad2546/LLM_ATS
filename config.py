# config.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os

import google.generativeai as genai

class Settings:
    # Gemini API key (stored in environment variable GEMINI_API_KEY)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

settings = Settings()
# Configure Gemini SDK
genai.configure(api_key=settings.GEMINI_API_KEY)

# Database connection parameters (from environment or defaults)
db_config = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "LLM_Resume")
}