# utils/candidate_utils.py

import os
import sys
import re
import json
import logging
import requests
import mysql.connector

# Ensure project root is on sys.path so Tools.logs can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Tools.logs import save_log   # save_log(log_type, message, process="Candidate_Parsing")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepSeek API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_KEY", "")

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
    is still None (or empty list for skills), fall back to DeepSeek LLM for those missing pieces.
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

    # 2) Build a DeepSeek prompt asking only for missing fields
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
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a resume parsing assistant. Return exactly the requested JSON."},
                {"role": "user",   "content": prompt}
            ],
            "temperature": 0.0
        }

        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()

        # Strip code fences if present
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

    except requests.exceptions.RequestException as e:
        msg = f"DeepSeek request failed while extracting missing fields: {e}"
        logger.error(msg)
        save_log("ERROR", msg, process="Candidate_Parsing")

    except json.JSONDecodeError as e:
        msg = f"JSON parsing error from DeepSeek (missing fields): {e}"
        logger.error(msg)
        save_log("ERROR", msg, process="Candidate_Parsing")

    except Exception as e:
        msg = f"Unexpected error extracting candidate details via DeepSeek: {e}"
        logger.error(msg)
        save_log("ERROR", msg, process="Candidate_Parsing")

    return parsed


def upsert_candidate(cand_info: dict, resume_path: str) -> int:
    """
    Inserts a new candidate or updates existing based on email.
    cand_info is the dict returned by extract_candidate_details(...).
    resume_path is the full file path to the stored resume PDF.

    Returns the candidate_id.
    """
    email = (cand_info.get("email") or "").lower().strip()
    if not email:
        raise ValueError("Cannot upsert candidate without an email.")

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        # Check if email already exists
        cursor.execute("SELECT `candidate_id` FROM `candidate` WHERE `email` = %s", (email,))
        row = cursor.fetchone()

        if row:
            # Candidate exists → UPDATE only those fields that are not None
            candidate_id = row[0]
            cursor.execute(
                """
                UPDATE `candidate`
                   SET `name`                 = COALESCE(%s, `name`),
                       `phone`                = COALESCE(%s, `phone`),
                       `linkedin_url`         = COALESCE(%s, `linkedin_url`),
                       `current_location`     = COALESCE(%s, `current_location`),
                       `years_of_experience`  = COALESCE(%s, `years_of_experience`),
                       `education_level`      = COALESCE(%s, `education_level`),
                       `last_position_title`  = COALESCE(%s, `last_position_title`),
                       `skills`               = COALESCE(%s, `skills`),
                       `resume_path`          = %s,
                       `updated_at`           = NOW()
                 WHERE `candidate_id` = %s
                """,
                (
                    cand_info.get("name"),
                    cand_info.get("phone"),
                    cand_info.get("linkedin_url"),
                    cand_info.get("current_location"),
                    cand_info.get("years_of_experience"),
                    cand_info.get("education_level"),
                    cand_info.get("last_position_title"),
                    ", ".join(cand_info.get("skills", [])) or None,
                    resume_path,
                    candidate_id
                )
            )
            conn.commit()
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
                    resume_path
                )
            )
            conn.commit()
            candidate_id = cursor.lastrowid

        return candidate_id

    except Exception as e:
        msg = f"Error upserting candidate record: {e}"
        logger.error(msg)
        save_log("ERROR", msg, process="Candidate_Upsert")
        raise

    finally:
        cursor.close()
        conn.close()