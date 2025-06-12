# services/jd_service.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import logging
import requests

from utils.db_utils import get_connection
from Tools.logs import save_log
from config import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Fetch all category names that already exist in the `category` table
def fetch_all_category_names() -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT `name` FROM `category`")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

# Insert a new category row if it does not exist; return its category_id
def get_or_create_category_id(category_name: str, category_type: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT `category_id` FROM `category` WHERE `name` = %s", (category_name,))
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor.execute(
            "INSERT INTO `category` (`name`, `type`) VALUES (%s, %s)",
            (category_name, category_type)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
        conn.close()

def analyze_jd_with_gpt(jd_text: str) -> dict:
    """
    Calls DeepSeek to classify the JD into up to two categories, and extract qualifications/requirements.
    Returns a dict:
      {
        "categories": [... up to two category names ...],
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
                "content": "You are a JD analyzer that returns exactly the JSON object described."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()

        response_text = response.json()['choices'][0]['message']['content'].strip()
        logger.info(f"DeepSeek raw response for JD classification: {response_text}")

        # Strip code fences if present
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip("` \n")
        elif response_text.startswith("```"):
            response_text = response_text[3:].strip("` \n")

        result = json.loads(response_text)
        if not isinstance(result, dict):
            raise ValueError("DeepSeek did not return a JSON object")

        for field in ("categories", "qualifications", "requirements"):
            if field not in result:
                raise ValueError(f"Missing field '{field}' in DeepSeek response")

        # Deduplicate categories and limit to 1
        unique_cats = []
        for name in result["categories"]:
            if name not in unique_cats:
                unique_cats.append(name)
            if len(unique_cats) == 1:
                break
        result["categories"] = unique_cats[:1]

        return result

    except Exception as e:
        msg = f"Error calling DeepSeek or parsing response: {str(e)}"
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        return {"categories": [], "qualifications": "", "requirements": ""}


def save_jd_to_db(jd_text: str, categories: list, qualifications: str, requirements: str):
    """
    Saves a new job_description row with:
      - jd_text,
      - category_detected (comma-separated),
      - category1_id,
      - qualifications, requirements.
    """
    c1_id = None

    if len(categories) >= 1:
        c1_id = get_or_create_category_id(categories[0], "category1")

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO `job_description`
              (`jd_text`, `category_detected`, `uploaded_at`,
               `category1_id`, `qualifications`, `requirements`)
            VALUES (%s, %s, NOW(), %s, %s, %s)
            """,
            (
                jd_text,
                ", ".join(categories),
                c1_id,
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