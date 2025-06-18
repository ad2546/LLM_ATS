"""
services/jd_service.py

NOTE: All JD analysis now uses Gemini embeddings + FAISS for category detection and similarity.
The function analyze_jd is the sole analysis function. analyze_jd_with_gpt is kept as an alias for compatibility.
"""
import os, sys

import requests
import google.generativeai as genai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.category_utils import get_or_create_category_id
import os
import logging
import mysql.connector
import json
from utils.db_utils import get_connection
from Tools.logs import save_log
from utils.embeddings import embed_text

logger = logging.getLogger(__name__)

from utils.embeddings import get_category_index

def analyze_jd(jd_text: str) -> dict:
    """
    Classify a job description into a single best-fit category using Gemini Pro LLM prompting.
    Returns {"categories": [name1], "qualifications": "", "requirements": ""}
    """
    try:
        # Load all category names from DB
        conn = mysql.connector.connect(**{
            "host": os.getenv('MYSQL_HOST', 'localhost'),
            "user": os.getenv('MYSQL_USER', 'root'),
            "password": os.getenv('MYSQL_PASSWORD', ''),
            "database": os.getenv('MYSQL_DATABASE', 'LLM_Resume')
        })
        cur = conn.cursor()
        cur.execute("SELECT name FROM category")
        categories = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()

        # Prepare prompt with category options
        category_options = ", ".join(categories)
        prompt = f"""
You are a job description analyzer. Given the following job description, extract the single best-fit job category (choose one category from the following options: {category_options}), qualifications, and requirements.

Respond ONLY with a JSON object in the format:
{{
  "category": category1,
  "qualifications": "...",
  "requirements": "..."
}}

Job Description:
\"\"\"
{jd_text}
\"\"\"
"""

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        content = response.text.strip()
        logger.info(f"Gemini raw response: {content}")
        import re
        try:
            result = json.loads(content)
        except Exception:
            # Try to extract JSON block from output if there's extra text
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group())
                except Exception:
                    result = {}
            else:
                result = {}
        categories = [result["category"]] if "category" in result else []
        qualifications = result.get("qualifications", "")
        requirements = result.get("requirements", "")
        return {"categories": categories, "qualifications": qualifications, "requirements": requirements}
    except Exception as e:
        logger.error(f"JD Gemini flash classification failed: {e}")
        save_log("ERROR", str(e), process="JD_Analysis")
        return {"categories": [], "qualifications": "", "requirements": ""}

#

analyze_jd_with_gpt = analyze_jd

def save_jd_to_db(jd_text: str, categories: list, qualifications: str, requirements: str):
    """
    Saves a new job_description row with:
      - jd_text,
      - category_detected (comma-separated),
      - qualifications, requirements.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO `job_description`
              (`jd_text`, `category_detected`, `uploaded_at`
               , `qualifications`, `requirements`)
            VALUES (%s, %s, NOW(), %s, %s)
            """,
            (
                jd_text,
                ", ".join(categories),
                qualifications,
                requirements
            )
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    save_log(
        "INFO",
        f"JD saved (detected='{','.join(categories)}', quals='{qualifications}', reqs='{requirements}')",
        process="JD_Analysis"
    )
 
