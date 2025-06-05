import os
import re
import json
import fitz                     # PyMuPDF
import requests
import mysql.connector
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add parent dir to path so we can import Tools.logs
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Tools.logs import save_log   # expects save_log(log_type, message, process="JD_Analysis")

# ---------- Logging Setup ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Flask App ----------
app = Flask(__name__)
CORS(app)

# ---------- Configuration ----------
class Settings:
    OPENAI_API_KEY: str = os.getenv("DEEPSEEK_KEY", "")

settings = Settings()

db_config = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "LLM_Resume")
}


# ---------- Helper: Read PDF bytes → text ----------
def read_pdf_content(file_bytes: bytes) -> str:
    """
    Given raw PDF bytes, return concatenated text.
    """
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() or ""
        return text
    except Exception as e:
        msg = f"PDF parsing failed: {str(e)}"
        logger.exception(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        return ""


# ---------- Helper: Fetch all existing category names from `category` table ----------
def fetch_all_category_names() -> list:
    """
    Returns a list of all category.name values currently in the `category` table.
    """
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT `name` FROM `category`")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]


# ---------- Helper: Insert a new category if missing, return its category_id ----------
def get_or_create_category_id(category_name: str, category_type: str) -> int:
    """
    Given a category_name and category_type ('category1' or 'category2'),
    return category.category_id, inserting a new row if it doesn't exist.
    """
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT `category_id` FROM `category` WHERE `name` = %s", (category_name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        # Insert new category
        cursor.execute(
            "INSERT INTO `category` (`name`, `type`) VALUES (%s, %s)",
            (category_name, category_type)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
        conn.close()


# ---------- DeepSeek call: classify a JD into categories + extract quals/reqs ----------
def analyze_jd_with_gpt(jd_text: str) -> dict:
    """
    Calls DeepSeek to classify the JD into up to two categories, and also
    extract 'qualifications' and 'requirements'. Returns:
      {
        "categories": [... up to two names ...],
        "qualifications": "<short summary>",
        "requirements": "<short summary>"
      }
    """
    existing_names = fetch_all_category_names()
    if not existing_names:
        prompt_category_list = "    (no categories exist in the database)\n"
    else:
        prompt_category_list = ""
        for name in existing_names:
            prompt_category_list += f'    "{name}"\n'

    prompt = f"""
You are an AI model. Given a job description, choose up to two categories, plus extract qualifications and requirements.

Below is the list of valid category names (pulled from database). Pick exactly up to two that match this JD.
If fewer than two apply, return only those. If none apply, return an empty array [].

Valid categories (from database):
{prompt_category_list}

Format your answer as a JSON object with exactly these fields:
- "categories": an array of zero, one, or two distinct category names, each matching one above
- "qualifications": a short summary of academic/professional credentials (one line, comma-separated)
- "requirements": a short summary of key skills/experience needed (one line, comma-separated)

Do NOT include markdown, commentary, or extra fields—only the JSON object.

### Job Description:
{jd_text}

### Example valid output:
{{
  "categories": ["Academic Nursing","Nursing Research"],
  "qualifications": "PhD in Nursing, RN license",
  "requirements": "5+ years teaching, curriculum development"
}}

### Example if only one category:
{{
  "categories": ["Academic Nursing"],
  "qualifications": "PhD in Nursing, RN license",
  "requirements": "5+ years teaching, research publications"
}}

### Example if no match:
{{
  "categories": [],
  "qualifications": "",
  "requirements": ""
}}
"""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type":  "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": "You are a JD analyzer that returns a JSON object with fields: categories, qualifications, requirements."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        response_text = response.json()['choices'][0]['message']['content'].strip()
        logger.info(f"DeepSeek raw response: {response_text}")

        # Strip code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip("` \n")
        elif response_text.startswith("```"):