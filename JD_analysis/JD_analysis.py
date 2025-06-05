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
            response_text = response_text[3:].strip("` \n")

        result = json.loads(response_text)

        if not isinstance(result, dict):
            raise ValueError("DeepSeek did not return a JSON object")

        # Ensure required keys exist
        for field in ("categories", "qualifications", "requirements"):
            if field not in result:
                raise ValueError(f"Missing field '{field}' in DeepSeek response")

        # Deduplicate and limit to 2 categories
        unique_cats = []
        for name in result["categories"]:
            if name not in unique_cats:
                unique_cats.append(name)
            if len(unique_cats) == 2:
                break
        result["categories"] = unique_cats

        return result

    except Exception as e:
        msg = f"Error calling DeepSeek or parsing response: {str(e)}"
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        # Return empty fallback
        return {"categories": [], "qualifications": "", "requirements": ""}


# ---------- Save JD into DB (with two category slots) ----------
def save_jd_to_db(jd_text: str, categories: list, qualifications: str, requirements: str):
    """
    categories: list of up to two category names.
    qualifications: short summary string.
    requirements: short summary string.
    Insert or get category1_id/category2_id, then store all fields in job_description.
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
            INSERT INTO `job_description`
              (`jd_text`, `category_detected`, `uploaded_at`,
               `category1_id`, `category2_id`, `qualifications`, `requirements`)
            VALUES (%s, %s, NOW(), %s, %s, %s, %s)
            """,
            (
                jd_text,
                ", ".join(categories),  # store combined text in category_detected
                c1_id,
                c2_id,
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


# ---------- Extract minimal candidate details from resume text ----------
def extract_candidate_details(resume_text: str) -> dict:
    """
    Very basic extraction of name, email, phone, LinkedIn, etc.
    """
    details = {
        "name": None,
        "email": None,
        "phone": None,
        "linkedin_url": None,
        "current_location": None,
        "years_of_experience": None,
        "education_level": None,
        "last_position_title": None,
        "skills": []
    }

    # Extract email
    m = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text)
    if m:
        details["email"] = m.group(0).lower()

    # Extract phone
    m2 = re.search(r"(\+?\d{1,2}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", resume_text)
    if m2:
        details["phone"] = m2.group(0)

    # Extract name as first Title‐case line
    for line in resume_text.split("\n"):
        ln = line.strip()
        if not ln:
            continue
        words = ln.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w):
            details["name"] = ln
            break

    # Extract LinkedIn
    m3 = re.search(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+", resume_text)
    if m3:
        details["linkedin_url"] = m3.group(0)

    return details


# ---------- REAL DeepSeek scoring: score a resume vs. both categories ----------
def score_resume_for_categories(category_names: list, jd_text: str, resume_text: str) -> dict:
    """
    Calls DeepSeek once to score 'resume_text' against both categories.
    Returns a dict:
    {
      "categories": [
        {
          "category": "<cat1>",
          "score": <0-100>,
          "weight": <0-1>,
          "justification": "<…>"
        },
        {
          "category": "<cat2>",
          "score": <0-100>,
          "weight": <0-1>,
          "justification": "<…>"
        }
      ],
      "final_score": <0-100>
    }
    """
    cats_list = '", "'.join(category_names)
    prompt = f"""
You are an AI that scores a candidate’s resume against **two categories** within a single job description.  
Prioritize “experience” in the resume with the highest weight, then “education,” then “skills.”

Categories to score:
- "{category_names[0]}"
- "{category_names[1]}"

Full Job Description:
{jd_text}

Candidate Resume:
{resume_text}

Please respond ONLY in the following JSON format (no extra commentary or markdown):

{{
  "categories": [
    {{
      "category": "{category_names[0]}",
      "score": 0.0,
      "weight": 0.0,
      "justification": ""
    }},
    {{
      "category": "{category_names[1]}",
      "score": 0.0,
      "weight": 0.0,
      "justification": ""
    }}
  ],
  "final_score": 0.0
}}

- Each “score” must be between 0 and 100.
- Each “weight” must be between 0.0 and 1.0.
- “justification” should be a brief (1–2 sentences) rationale.
- “final_score” is the weighted sum of the two category scores.
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
                "content": "You are a resume scoring assistant. Return exactly the JSON object requested."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        raw = r.json()['choices'][0]['message']['content'].strip()
        logger.info(f"DeepSeek combined scoring raw response: {raw}")

        # Strip code fences if present
        if raw.startswith("```json"):
            raw = raw[7:].strip("` \n")
        elif raw.startswith("```"):
            raw = raw[3:].strip("` \n")

        parsed = json.loads(raw)
        # Validate structure
        if "categories" not in parsed or "final_score" not in parsed:
            raise ValueError("Missing keys in combined scoring response")

        # Coerce numeric types
        for entry in parsed["categories"]:
            entry["score"] = float(entry["score"])
            entry["weight"] = float(entry["weight"])
        parsed["final_score"] = float(parsed["final_score"])

        return parsed

    except Exception as e:
        msg = f"Error calling DeepSeek combined scoring: {str(e)}"
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        # Fallback: zero everything
        return {
            "categories": [
                {"category": category_names[0], "score": 0.0, "weight": 0.0, "justification": f"Error: {str(e)}"},
                {"category": category_names[1], "score": 0.0, "weight": 0.0, "justification": f"Error: {str(e)}"}
            ],
            "final_score": 0.0
        }


# ---------- Helper: Write a log record ----------
def log_to_db(process: str, message: str, log_type: str="info"):
    """
    Insert a row into logs (log_type, process, message).
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO `logs` (`log_type`, `process`, `message`) VALUES (%s, %s, %s)",
            (log_type, process, message)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to write to logs table: {e}")


# ---------- API: /upload (Job Description upload) ----------
@app.route('/upload', methods=['POST'])
def upload_jd():
    logger.info("Received JD upload request.")
    save_log("INFO", "JD upload received", process="JD_Analysis")

    if 'file' not in request.files:
        msg = "No job description PDF provided"
        save_log("ERROR", msg, process="JD_Analysis")
        return jsonify({'error': msg}), 400

    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'):
        msg = "Only PDF files are accepted for job description"
        save_log("ERROR", msg, process="JD_Analysis")
        return jsonify({'error': msg}), 400

    try:
        # 1) Read PDF bytes → extract text
        file_bytes = file.read()
        jd_text = read_pdf_content(file_bytes)
        if not jd_text.strip():
            msg = "Job description PDF parsing returned no text"
            save_log("ERROR", msg, process="JD_Analysis")
            return jsonify({'error': msg}), 400

        # 2) Call DeepSeek to get categories, qualifications, requirements
        jd_info = analyze_jd_with_gpt(jd_text)
        categories     = jd_info.get("categories", [])
        qualifications = jd_info.get("qualifications", "")
        requirements   = jd_info.get("requirements", "")

        # 3) Save JD into DB
        save_jd_to_db(jd_text, categories, qualifications, requirements)

        # 4) Return JSON to caller
        return jsonify({
            "categories":     categories,
            "qualifications": qualifications,
            "requirements":   requirements
        })

    except Exception as e:
        msg = f"Unhandled exception in /upload: {e}"
        logger.exception(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        return jsonify({'error': msg}), 500


# ---------- API: /recommended (Score all resumes against that JD) ----------
@app.route('/recommended', methods=['GET'])
def recommended():
    jd_id = request.args.get('jd_id')
    if not jd_id:
        msg = "Missing jd_id parameter"
        save_log("ERROR", msg, process="JD_Analysis")
        return jsonify({'error': msg}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 1) Fetch jd_text, category1_id & category2_id for this JD
        cursor.execute(
            "SELECT `jd_text`, `category1_id`, `category2_id` "
            "FROM `job_description` WHERE `jd_id` = %s",
            (jd_id,)
        )
        jd_row = cursor.fetchone()
        if not jd_row:
            msg = f"Job description {jd_id} not found"
            save_log("ERROR", msg, process="JD_Analysis")
            cursor.close()
            conn.close()
            return jsonify({'error': msg}), 404

        jd_text = jd_row["jd_text"]
        cat_ids = []
        if jd_row["category1_id"]:
            cat_ids.append(jd_row["category1_id"])
        if jd_row["category2_id"]:
            cat_ids.append(jd_row["category2_id"])

        # 2) Convert category_id → category_name
        category_names = []
        for cid in cat_ids:
            cursor.execute("SELECT `name` FROM `category` WHERE `category_id` = %s", (cid,))
            r2 = cursor.fetchone()
            if r2:
                category_names.append(r2["name"])

        # If fewer than 2 categories, we still want exactly two slots (second can be empty)
        if len(category_names) == 1:
            category_names.append("")  # empty second category

        # 3) Locate ./resumes directory
        resumes_dir = os.path.join(os.path.dirname(__file__), '../resumes')
        if not os.path.isdir(resumes_dir):
            msg = f"Resumes folder not found at {resumes_dir}"
            save_log("ERROR", msg, process="JD_Analysis")
            cursor.close()
            conn.close()
            return jsonify({'error': msg}), 500

        # 4) Build a map of existing candidates (email → candidate_id)
        cursor.execute("SELECT `candidate_id`, `email` FROM `candidate`")
        existing_map = {row["email"].lower(): row["candidate_id"] for row in cursor.fetchall()}

        summary = []

        # 5) Loop over each resume PDF in ./resumes
        for fname in os.listdir(resumes_dir):
            if not fname.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(resumes_dir, fname)

            # 5a) Read PDF bytes + extract text
            try:
                with open(full_path, "rb") as rf:
                    pdf_bytes = rf.read()
                resume_text = read_pdf_content(pdf_bytes)
            except Exception as e:
                msg = f"Failed to read resume '{fname}': {e}"
                logger.error(msg)
                save_log("ERROR", msg, process="JD_Analysis")
                continue

            # 5b) Extract candidate details
            cand_info = extract_candidate_details(resume_text)
            email = (cand_info.get("email") or "").lower().strip()

            # 5c) Get or create `candidate` row
            candidate_id = None
            if email and email in existing_map:
                candidate_id = existing_map[email]
            else:
                # Insert new candidate
                cursor.execute(
                    """
                    INSERT INTO `candidate`
                      (`name`, `email`, `phone`, `linkedin_url`,
                       `current_location`, `years_of_experience`, `education_level`,
                       `last_position_title`, `skills`, `resume_path`, `created_at`)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (
                        cand_info.get("name"),
                        email,
                        cand_info.get("phone"),
                        cand_info.get("linkedin_url"),
                        cand_info.get("current_location"),
                        cand_info.get("years_of_experience"),
                        cand_info.get("education_level"),
                        cand_info.get("last_position_title"),
                        ", ".join(cand_info.get("skills", [])),
                        full_path
                    )
                )
                conn.commit()
                candidate_id = cursor.lastrowid
                existing_map[email] = candidate_id

            # 5d) Score the resume once against both categories
            combined_scores = score_resume_for_categories(category_names, jd_text, resume_text)

            # 5e) Insert each category score row into `category_score`
            for cat_entry in combined_scores["categories"]:
                cname   = cat_entry["category"]
                sc      = cat_entry["score"]
                wt      = cat_entry["weight"]
                just    = cat_entry["justification"]

                # Skip empty category names
                if not cname:
                    continue

                cursor.execute(
                    """
                    INSERT INTO `category_score`
                      (`candidate_id`, `category_name`, `score`, `weight`, `justification`, `updated_at`)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    """,
                    (candidate_id, cname, sc, wt, just)
                )
                conn.commit()

            # 5f) Build a single summary JSON object
            summary.append({
                "candidate_email": email,
                "candidate_id":    candidate_id,
                "resume_file":     fname,
                "categories":      combined_scores["categories"],
                "final_score":     combined_scores["final_score"]
            })

        cursor.close()
        conn.close()

        save_log("INFO", f"Completed scoring recommendation for jd_id={jd_id}", process="JD_Analysis")
        return jsonify(summary)

    except Exception as e:
        msg = f"Unhandled exception in /recommended: {e}"
        logger.exception(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        return jsonify({'error': msg}), 500


# ---------- Healthcheck Endpoint ----------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})


# ---------- Main ----------
if __name__ == "__main__":
    app.run(debug=True, port=5000)