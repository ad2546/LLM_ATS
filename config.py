# config.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os

class Settings:
    # DeepSeek API key (stored in environment variable DEEPSEEK_KEY)
    OPENAI_API_KEY: str = os.getenv("DEEPSEEK_KEY", "")

# Database connection parameters (from environment or defaults)
db_config = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "LLM_Resume")
}