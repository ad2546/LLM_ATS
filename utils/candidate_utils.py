# utils/candidate_utils.py

import os
import sys
import re
import json
import logging
import requests
import pymysql
import mysql.connector
import google.generativeai as genai

# Set your Gemini API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# Ensure project root is on sys.path so Tools.logs can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Tools.logs import save_log   # save_log(log_type, message, process="Candidate_Parsing")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration (expects environment variables)
db_config = {
    "host":     os.getenv("MYSQL_HOST", "localhost"),
    "user":     os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "LLM_Resume")
}


def _regex_extract_basic(resume_text: str) -> dict:
    """
    First-pass extraction using regex. Returns a dict with any fields found; missing fields remain None or empty.
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

    # 1) Email
    m_email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", resume_text)
    if m_email:
        details["email"] = m_email.group(0).lower()

    # 2) Phone (US‐style)
    m_phone = re.search(
        r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", resume_text
    )
    if m_phone:
        details["phone"] = m_phone.group(0)

    # 3) LinkedIn URL (plain) or href="..."
    m_link = re.search(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+", resume_text)
    if m_link:
        details["linkedin_url"] = m_link.group(0)
    else:
        # Try to catch href variants: <a href="https://linkedin.com/in/...">
        m_href = re.search(r'href=["\'](https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_]+)["\']', resume_text, flags=re.IGNORECASE)
        if m_href:
            details["linkedin_url"] = m_href.group(1)

    # 4) Name: look for first non‐empty line in Title Case
    for line in resume_text.splitlines():
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w[0].isalpha()):
            details["name"] = line
            break

    # 5) Years of Experience: e.g. “X years experience” or “X+ years”
    m_yoe = re.search(r"(\d{1,2})\+?\s+years? (of )?experience", resume_text, flags=re.IGNORECASE)
    if m_yoe:
        try:
            details["years_of_experience"] = int(m_yoe.group(1))
        except:
            pass

    # 6) Education Level: look for common degree abbreviations
    m_edu = re.search(
        r"(Ph\.?D\.?|Doctorate|M\.?S\.?|MSc|MBA|B\.?S\.?|BA|BSc|"
        r"M\.?A\.?|Master of Science|Master of Arts|Bachelor of Science|Bachelor of Arts)",
        resume_text, flags=re.IGNORECASE
    )
    if m_edu:
        details["education_level"] = m_edu.group(0)

    # 7) Last Position Title: look for lines containing “Experience” then next non‐empty line
    sections = resume_text.splitlines()
    for idx, line in enumerate(sections):
        if re.search(r"\bExperience\b|\bWork History\b|\bProfessional Experience\b", line, flags=re.IGNORECASE):
            for next_line in sections[idx+1:]:
                nl = next_line.strip()
                if nl:
                    details["last_position_title"] = nl
                    break
            break

    # 8) Current Location: look for “City, State” pattern near top 10 lines
    for line in sections[:10]:
        m_loc = re.search(r"[A-Za-z]+,\s*[A-Z]{2}", line)
        if m_loc:
            details["current_location"] = m_loc.group(0)
            break

    # 9) Skills: look for a “Skills” section, then comma‐separated keywords
    skills = []
    for idx, line in enumerate(sections):
        if re.search(r"\bSkills\b|\bTechnical Skills\b", line, flags=re.IGNORECASE):
            for sub_line in sections[idx+1:]:
                if not sub_line.strip():
                    break
                if re.search(r"\bExperience\b|\bWork History\b", sub_line, flags=re.IGNORECASE):
                    break
                parts = re.split(r"[,;]", sub_line)
                for p in parts:
                    p = p.strip()
                    if p and len(p) < 40:
                        skills.append(p)
            break
    details["skills"] = list(dict.fromkeys(skills))  # dedupe

    return details


def extract_candidate_details(resume_text: str) -> dict:
    """
    Combined extraction: first try regex (_regex_extract_basic). If any of the nine fields
    is still None (or empty list for skills), fall back to Gemini LLM for those missing pieces.
    """
    # 1) First‐pass regex extraction
    parsed = _regex_extract_basic(resume_text)

    # Build a list of fields that remain missing
    missing_fields = []
    for key in ("name", "email", "phone", "linkedin_url",
                "current_location", "years_of_experience",
                "education_level", "last_position_title", "skills"):
        val = parsed.get(key)
        if val is None or (key == "skills" and not val):
            missing_fields.append(key)

    if not missing_fields:
        # All fields found by regex, return immediately
        return parsed

    # 2) Build a Gemini prompt asking only for missing fields
    ask_fields = ", ".join(f'"{f}"' for f in missing_fields)
    prompt = f"""
You are an AI assistant specialized in parsing resumes. Extract only the following fields (if present) in valid JSON format: {ask_fields}

Return a JSON object with keys exactly matching:
{json.dumps(missing_fields, indent=2)}

Each missing field should be one of:
- "name": Full name (First Last), or null.
- "email": Email address, or null.
- "phone": Phone number, or null.
- "linkedin_url": LinkedIn URL, or null.
- "current_location": City, State or null.
- "years_of_experience": integer or null.
- "education_level": Highest degree (e.g., "PhD in Nursing") or null.
- "last_position_title": Most recent job title or null.
- "skills": array of up to 10 skill strings or [].

Resume Text:
\"\"\"
{resume_text}
\"\"\"

Respond with only a JSON object containing exactly those keys (no extra commentary).
"""

    try:
        # Use Gemini 2.0 Flash model for fast and cheap inference
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```json"):
            raw = raw[7:].strip("` \n")
        elif raw.startswith("```"):
            raw = raw[3:].strip("` \n")

        llm_data = json.loads(raw)

        # Fill in only the missing fields from llm_data
        for key in missing_fields:
            if key in llm_data and llm_data[key] not in (None, "", []):
                parsed[key] = llm_data[key]

        # Ensure skills is a list
        if not isinstance(parsed["skills"], list):
            parsed["skills"] = []

    except Exception as e:
        msg = f"Gemini request failed while extracting missing fields: {e}"
        logger.error(msg)
        save_log("ERROR", msg, process="Candidate_Parsing")

    return parsed


def upsert_candidate(cand_info: dict, resume_path: str) -> int:
    """
    Inserts a new candidate or updates existing based on email.
    Returns the candidate_id.
    """
    email = (cand_info.get("email") or "").lower().strip()
    if not email:
        raise ValueError("Cannot upsert candidate without an email.")

    candidate_name = cand_info.get("name")
    candidate_phone = cand_info.get("phone")
    candidate_location = cand_info.get("current_location")
    candidate_year = cand_info.get("years_of_experience")
    candidate_job = cand_info.get("last_position_title")
    candidate_resume = resume_path
    candidate_company = cand_info.get("company", None)

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    try:
        sql = """
        INSERT INTO `candidate`
            (`candidate_name`, `candidate_email`, `candidate_phone`, `candidate_location`,
             `candidate_year`, `candidate_job`, `candidate_resume`, `created_at`, `candidate_updated_time`, `candidate_company`)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, NOW(), %s)
        ON DUPLICATE KEY UPDATE
            `candidate_name` = VALUES(`candidate_name`),
            `candidate_phone` = VALUES(`candidate_phone`),
            `candidate_location` = VALUES(`candidate_location`),
            `candidate_year` = VALUES(`candidate_year`),
            `candidate_job` = VALUES(`candidate_job`),
            `candidate_resume` = VALUES(`candidate_resume`),
            `candidate_updated_time` = NOW(),
            `candidate_company` = VALUES(`candidate_company`);
        """
        vals = (
            candidate_name,
            email,
            candidate_phone,
            candidate_location,
            candidate_year,
            candidate_job,
            candidate_resume,
            candidate_company
        )
        cursor.execute(sql, vals)
        conn.commit()

        # Fetch candidate_id for downstream use
        cursor.execute("SELECT candidate_id FROM candidate WHERE candidate_email = %s", (email,))
        candidate_id = cursor.fetchone()[0]
        return candidate_id

    except Exception as e:
        msg = f"Error upserting candidate record: {e}"
        logger.error(msg)
        save_log("ERROR", msg, process="Candidate_Upsert")
        raise

    finally:
        cursor.close()
        conn.close()


def save_score_to_jd_score(
    db_connection,
    jd_id,
    candidate_id,
    candidate_email,
    candidate_name,
    category_score,
    final_score,
    qualifications_score,
    requirements_score,
    reason,
    resume_filename,
    resume_path
):
    try:
        with db_connection.cursor() as cursor:
            insert_sql = """
                INSERT INTO jd_score (
                    jd_id, candidate_id, candidate_email, candidate_name,
                    category_score, final_score, qualifications_score, requirements_score,
                    reason, resume_filename, resume_path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_sql, (
                jd_id, candidate_id, candidate_email, candidate_name,
                category_score, final_score, qualifications_score, requirements_score,
                reason, resume_filename, resume_path
            ))
            db_connection.commit()
        print("Scoring result saved to jd_score.")
    except Exception as e:
        print("Error saving score to jd_score:", e)