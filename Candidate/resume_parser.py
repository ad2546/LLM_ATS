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
from Tools.logs import save_log

# --------------- Logging Setup ---------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------- Flask App ---------------
app = Flask(__name__)
CORS(app)

# --------------- Configuration ---------------
class Settings:
    OPENAI_API_KEY: str = os.getenv("DEEPSEEK_KEY", "")

settings = Settings()

db_config = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "LLM_Resume")
}

# --------------- Helper: Read PDF bytes → text ---------------
def read_pdf_content(file_bytes: bytes) -> str:
    """
    Given raw PDF bytes, return the concatenated text of all pages.
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
        save_log("ERROR", msg)
        return ""

# --------------- Helper: Fetch all existing category names from `category` table ---------------
def fetch_all_category_names() -> list:
    """
    Returns a Python list of all category names currently in the `category` table.
    """
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM category")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [row[0] for row in rows]

# --------------- Helper: Insert a new category if missing, return its category_id ---------------
def get_or_create_category_id(category_name: str, category_type: str) -> int:
    """
    Given a category_name and a category_type ("category1" or "category2"),
    returns its category_id, inserting a new row if it doesn't exist.
    """
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    try:
        # Check if it already exists (we ignore type on SELECT; we just care about name)
        cursor.execute(
            "SELECT category_id FROM category WHERE name = %s",
            (category_name,)
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        # Otherwise insert a new category row with the given type
        cursor.execute(
            "INSERT INTO category (name, type) VALUES (%s, %s)",
            (category_name, category_type)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
        conn.close()

# --------------- DeepSeek call: classify a JD into up to two categories ---------------
def analyze_jd_with_gpt(jd_text: str) -> list:
    """
    Fetches up to two category names from DeepSeek.  The prompt dynamically
    includes all existing category names from the database.
    Returns a Python list (length 0–2) of category names.
    """
    # 1) Read all existing category names from `category` table
    existing_names = fetch_all_category_names()
    if not existing_names:
        prompt_category_list = "    (no categories exist in the database)\n"
    else:
        prompt_category_list = ""
        for name in existing_names:
            prompt_category_list += f'    "{name}"\n'

    # 2) Build the DeepSeek prompt
    prompt = f"""
You are an AI model that classifies job descriptions into zero, one or two categories.

Below is the *current* list of valid category names (pulled from our database).  Please choose up to two names from this list.  If you feel the job description does not match any, return an empty JSON array [].

Valid categories (from database):
{prompt_category_list}

Format your answer as a JSON array of up to two strings, each exactly matching one of the above names.
Do NOT include markdown, commentary, or additional fields—just the JSON array.

### Job Description:
{jd_text}

### Example output if you pick two:
[
    "Academic Nursing",
    "Administrative/Leadership"
]
    
Example if no match:
[]
"""

    # 3) Call DeepSeek
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            { "role": "system", "content": "You are a job description classifier.  Return exactly a JSON array of up to two names." },
            { "role": "user",   "content": prompt }
        ],
        "temperature": 0.3
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        response_text = response.json()['choices'][0]['message']['content'].strip()

        logger.info(f"DeepSeek raw response: {response_text}")
        # Strip triple back‐ticks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:].strip("` \n")
        elif response_text.startswith("```"):
            response_text = response_text[3:].strip("` \n")

        categories = json.loads(response_text)
        if not isinstance(categories, list):
            raise ValueError("DeepSeek did not return a JSON array")

        # Return at most two categories
        return categories[:2]

    except Exception as e:
        msg = f"Error calling DeepSeek or parsing response: {str(e)}"
        logger.error(msg)
        save_log("ERROR", msg)
        return []


# --------------- Save JD into DB (with two category slots) ---------------
def save_jd_to_db(jd_text: str, categories: list):
    """
    categories is a Python list of up to two category‐names, e.g.
      ["Academic Nursing","Administrative/Leadership"].

    We interpret index 0 as type "category1", index 1 as "category2".
    Insert any categories not already in the `category` table.
    Finally, insert a row into `job_description(...)` with the two foreign‐keys.
    """
    c1_id = None
    c2_id = None

    if len(categories) >= 1:
        c1_id = get_or_create_category_id(categories[0], "category1")
    if len(categories) >= 2:
        c2_id = get_or_create_category_id(categories[1], "category2")

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO job_description (jd_text, category1_id, category2_id)
            VALUES (%s, %s, %s)
            """,
            (jd_text, c1_id, c2_id)
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()

    save_log("INFO", f"JD saved (categories: {categories}) as category1_id={c1_id}, category2_id={c2_id}")


# --------------- Extract minimal candidate details from resume text ---------------
def extract_candidate_details(resume_text: str) -> dict:
    """
    Very basic extraction of name, email, phone, LinkedIn.  In production,
    replace with a robust NLP pipeline.
    """
    details = {
        "name": None,
        "email": None,
        "phone": None,
        "linkedin": None,
        "location": None,
        "years_experience": None,
        "education_level": None,
        "job_title": None,
        "skills": []
    }

    # 1) email
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text)
    if m:
        details["email"] = m.group(0).lower()

    # 2) phone (US style, very basic)
    m2 = re.search(r"(\+?\d{1,2}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", resume_text)
    if m2:
        details["phone"] = m2.group(0)

    # 3) name: first line with Title Case words
    for line in resume_text.split("\n"):
        ln = line.strip()
        if not ln:
            continue
        words = ln.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            details["name"] = ln
            break

    # 4) LinkedIn
    m3 = re.search(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+", resume_text)
    if m3:
        details["linkedin"] = m3.group(0)

    # (You can expand regex/NLP for location/experience/education/job_title/skills.)
    return details

# --------------- Dummy scoring: score a resume vs. one category ---------------
def score_resume_for_category(category_name: str, resume_text: str) -> float:
    """
    Placeholder logic:  score = min(len(resume_text)/1000, 1.0) * 100.
    In production, replace with a DeepSeek call or similar.
    """
    try:
        raw_score = min(len(resume_text) / 1000.0, 1.0)
        return raw_score * 100.0
    except Exception as e:
        msg = f"Error scoring for category '{category_name}': {str(e)}"
        logger.exception(msg)
        save_log("ERROR", msg)
        return 0.0

# --------------- Helper: Write a log record ---------------
def log_to_db(process: str, message: str, log_type: str="info"):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (log_type, process, message) VALUES (%s,%s,%s)",
            (log_type, process, message)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to write to logs table: {e}")

# --------------- API: /upload (upload JD text and save categories) ---------------
@app.route('/upload', methods=['POST'])
def upload_jd():
    logger.info("Received JD upload request.")
    save_log("INFO", "JD upload received")

    jd_text = request.form.get('jd_text')
    if not jd_text:
        msg = "No job description text provided"
        save_log("ERROR", msg)
        return jsonify({'error': msg}), 400

    try:
        categories = analyze_jd_with_gpt(jd_text)   # e.g. ["Academic Nursing","Administrative/Leadership"]
        if not categories:
            msg = "DeepSeek returned no categories"
            save_log("ERROR", msg)
            return jsonify({'error': msg}), 500

        save_jd_to_db(jd_text, categories)
        return jsonify({"categories": categories})
    except Exception as e:
        msg = f"Unhandled exception in /upload: {e}"
        logger.exception(msg)
        save_log("ERROR", msg)
        return jsonify({'error': msg}), 500

# --------------- API: /recommended (score resumes against JD categories) ---------------
@app.route('/recommended', methods=['GET'])
def recommended():
    jd_id = request.args.get('jd_id')
    if not jd_id:
        msg = "Missing jd_id parameter"
        save_log("ERROR", msg)
        return jsonify({'error': msg}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 1) Fetch category1_id & category2_id for this JD
        cursor.execute(
            "SELECT category1_id, category2_id FROM job_description WHERE jd_id=%s",
            (jd_id,)
        )
        jd_row = cursor.fetchone()
        if not jd_row:
            msg = f"Job description {jd_id} not found"
            save_log("ERROR", msg)
            cursor.close()
            conn.close()
            return jsonify({'error': msg}), 404

        cat_ids = []
        if jd_row["category1_id"]:
            cat_ids.append(jd_row["category1_id"])
        if jd_row["category2_id"]:
            cat_ids.append(jd_row["category2_id"])

        # 2) Convert category_id → category_name
        category_names = []
        for cid in cat_ids:
            cursor.execute("SELECT name FROM category WHERE category_id=%s", (cid,))
            r2 = cursor.fetchone()
            if r2:
                category_names.append(r2["name"])

        # 3) Locate ./resumes directory
        resumes_dir = os.path.join(os.path.dirname(__file__), 'resumes')
        if not os.path.isdir(resumes_dir):
            msg = f"Resumes folder not found at {resumes_dir}"
            save_log("ERROR", msg)
            cursor.close()
            conn.close()
            return jsonify({'error': msg}), 500

        # 4) Build a map of existing candidates (email → candidate_id)
        cursor.execute("SELECT candidate_id, candidate_email FROM candidate")
        existing_map = {row["candidate_email"].lower(): row["candidate_id"] for row in cursor.fetchall()}

        summary = []

        # 5) Loop over each resume.pdf
        for fname in os.listdir(resumes_dir):
            if not fname.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(resumes_dir, fname)

            # 5a) Read PDF bytes + text
            try:
                with open(full_path, "rb") as f:
                    pdf_bytes = f.read()
                resume_text = read_pdf_content(pdf_bytes)
            except Exception as e:
                msg = f"Failed to read {fname}: {e}"
                logger.error(msg)
                save_log("ERROR", msg)
                continue

            # 5b) Extract candidate details
            cand_info = extract_candidate_details(resume_text)
            email = (cand_info.get("email") or "").lower().strip()

            # 5c) Get or create `candidate` row
            candidate_id = None
            if email and email in existing_map:
                candidate_id = existing_map[email]
            else:
                cursor.execute(
                    """
                    INSERT INTO candidate (
                      candidate_name,
                      candidate_email,
                      candidate_location,
                      candidate_phone,
                      candidate_resume,
                      candidate_linkedin,
                      candidate_years_experience,
                      candidate_education_level,
                      candidate_job_title,
                      candidate_skills
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        cand_info.get("name"),
                        email,
                        cand_info.get("location"),
                        cand_info.get("phone"),
                        full_path,
                        cand_info.get("linkedin"),
                        cand_info.get("years_experience"),
                        cand_info.get("education_level"),
                        cand_info.get("job_title"),
                        ", ".join(cand_info.get("skills", []))
                    )
                )
                conn.commit()
                candidate_id = cursor.lastrowid
                existing_map[email] = candidate_id

            # 5d) For each category, compute a score & upsert `category_score`
            for cname in category_names:
                score = score_resume_for_category(cname, resume_text)

                # Lookup category_id by name
                cursor.execute("SELECT category_id FROM category WHERE name=%s", (cname,))
                c_row = cursor.fetchone()
                if not c_row:
                    continue
                cat_id = c_row["category_id"]

                # Check if row exists in category_score
                cursor.execute(
                    """
                    SELECT score_id FROM category_score
                    WHERE candidate_id=%s AND category_id=%s AND jd_id=%s
                    """,
                    (candidate_id, cat_id, jd_id)
                )
                existing = cursor.fetchone()
                if existing:
                    # Update existing
                    cursor.execute(
                        "UPDATE category_score SET score=%s WHERE score_id=%s",
                        (score, existing["score_id"])
                    )
                else:
                    # Insert new
                    cursor.execute(
                        """
                        INSERT INTO category_score
                        (candidate_id, category_id, jd_id, score)
                        VALUES (%s,%s,%s,%s)
                        """,
                        (candidate_id, cat_id, jd_id, score)
                    )
                conn.commit()

                summary.append({
                    "candidate_id": candidate_id,
                    "candidate_email": email,
                    "category": cname,
                    "score": score,
                    "resume_file": fname
                })

        cursor.close()
        conn.close()

        save_log("INFO", f"Completed /recommended for jd_id={jd_id}")
        return jsonify(summary)

    except Exception as e:
        msg = f"Unhandled exception in /recommended: {e}"
        logger.exception(msg)
        save_log("ERROR", msg)
        return jsonify({'error': msg}), 500

# --------------- Healthcheck ---------------
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

# --------------- Main ---------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)