# services/score_service.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import logging
import requests
from typing import Tuple
from collections import Counter

from config import Settings
from Tools.logs import save_log

logger = logging.getLogger(__name__)
settings = Settings()

# Maximum tokens DeepSeek can accept
MAX_TOKENS = 65536

def approximate_token_count(text: str) -> int:
    print(len(text.split()))
    """Roughly count tokens by splitting on whitespace."""
    return len(text.split())

def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    Keep only the first max_tokens “words”—a rough stand-in for tokens.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return text
    truncated = " ".join(words[:max_tokens])
    return truncated

def prepare_jd_and_resume(jd_text: str, resume_text: str) -> Tuple[str, str]:
    """
    Force‐truncate JD+resume so that total tokens ≤ MAX_TOKENS.
    Strategy:
      1) If JD alone > 30k tokens, truncate JD to 30k.
      2) Otherwise, allocate remaining tokens to resume.
    """
    # Sanitize null characters
    jd_text = jd_text.replace("\x00", "")
    resume_text = resume_text.replace("\x00", "")

    jd_tokens = approximate_token_count(jd_text)
    # If JD is extremely long, cut it to 30k tokens
    if jd_tokens > 60000:
        logger.warning(f"Truncating full JD from {jd_tokens} tokens → 30000 tokens.")
        save_log("WARNING", f"JD truncated from {jd_tokens} to 30000 tokens", process="JD_Analysis")
        jd_text = truncate_text_to_tokens(jd_text, 60000)
        jd_tokens = 60000

    # Now remaining budget for resume:
    budget = MAX_TOKENS - jd_tokens
    if budget < 0:
        budget = 0

    resume_tokens = approximate_token_count(resume_text)
    if resume_tokens > budget:
        logger.warning(f"Resume too large ({resume_tokens} tokens) – summarizing using top keywords.")
        save_log("WARNING", f"Resume too large: summarizing instead of truncating", process="JD_Analysis")
        words = [word.strip(".,;:!?()[]{}\"'").lower() for word in resume_text.split()]
        common_words = Counter(words).most_common(100)
        resume_text = ", ".join([w for w, _ in common_words])

    return jd_text, resume_text

def score_resume_for_categories(category_names: list, jd_text: str, resume_text: str) -> dict:
    """
    Calls DeepSeek once to score 'resume_text' against a category.
    If the raw resume is too large (> MAX_TOKENS), log and skip sending to DeepSeek.
    Returns a dict:
      {
        "category": {"category": "<cat>", "score": <0–100>, "weight": <0.0–1.0>, "justification": "..."},
        "final_score": <0–10>
      }
    """
    if not category_names:
        category_names = [""]

    # Check raw resume size before any truncation
    raw_resume_tokens = approximate_token_count(resume_text)
    if raw_resume_tokens > MAX_TOKENS:
        msg = f"Resume too large ({raw_resume_tokens} tokens) – skipping DeepSeek scoring."
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        # Return zeroed fallback for category
        return score_resume_nlp_based(category_names[0], jd_text, resume_text)

    # 1) Truncate JD + resume to fit under MAX_TOKENS
    jd_trimmed, resume_trimmed = prepare_jd_and_resume(jd_text, resume_text)

    prompt = f"""
You are an AI that scores a candidate’s resume against a single category within a job description.
Prioritize “experience” with highest weight, then “education,” then “skills.”

Category:
"{category_names[0]}"

Full Job Description:
{jd_trimmed}

Candidate Resume:
{resume_trimmed}

Respond ONLY as JSON, exactly in this form (no extra comments).
IMPORTANT: All scores must be floating-point numbers between 0 and 10 (not fractions or percentages like 0.7 or 70%). Example: 6.5, 8.0, 4.3

{{
  "category": {{
    "category": "Data Science",
    "score": 7.5,  # out of 10
    "weight": 0.8,
    "justification": "..."
  }},
  "final_score": 7.5  # out of 10
}}
"""

    total_chars = len(prompt)
    logger.error(f"Estimated total characters sent to DeepSeek: {total_chars}")

    if total_chars > 150000:
        msg = f"Aborting DeepSeek request – prompt too large: {total_chars} characters (limit is 150,000)"
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        return score_resume_nlp_based(category_names[0], jd_text, resume_text)

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

    logger.error(f"DeepSeek request to {url}:\nHeaders: {headers}\nPayload:\n{json.dumps(payload, indent=2)}")

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

        # Validate
        if "category" not in parsed or "final_score" not in parsed:
            raise ValueError("Missing keys in JSON response")

        # Convert types
        entry = parsed["category"]
        entry["score"] = float(entry["score"])
        entry["weight"] = float(entry["weight"])
        parsed["final_score"] = round(entry["score"] * entry["weight"], 2)

        return parsed

    except Exception as e:
        if 'r' in locals():
            try:
                logger.error(f"DeepSeek error response:\n{r.text}")
                save_log("ERROR", r.text, process="DeepSeek_Scoring")
            except Exception:
                pass
        msg = f"Error calling DeepSeek combined scoring: {str(e)}"
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        # Return zeroed fallback
        return {
            "category": {"category": category_names[0], "score": 0.0, "weight": 0.0,
                         "justification": f"Error: {str(e)}"},
            "final_score": 0.0
        }

def score_resume_nlp_based(category_name: str, jd_text: str, resume_text: str) -> dict:
    """
    Fallback scoring method for large resumes. Performs keyword summarization
    and asks LLM to score based on top tokens.
    """
    from collections import Counter

    # Sanitize null characters
    jd_text = jd_text.replace("\x00", "")
    resume_text = resume_text.replace("\x00", "")

    # Tokenize and extract top keywords (simplified)
    words = [word.strip(".,;:!?()[]{}\"'").lower() for word in resume_text.split()]
    common_words = Counter(words).most_common(100)
    summary_keywords = ", ".join([w for w, _ in common_words])

    prompt = f"""
You are a resume scorer. Given a category, job description, and a list of resume keywords, assign a 0–10 score.

Category: "{category_name}"

Job Description:
{jd_text}

Resume Keywords:
{summary_keywords}

IMPORTANT: All scores must be floating-point numbers between 0 and 10. Example: 6.5, 8.0, 4.3

Respond in this format:

{{
  "category": {{
    "category": "{category_name}",
    "score": 0.0,
    "weight": 0.0,
    "justification": "..."
  }},
  "final_score": 0.0
}}
"""

    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a resume scoring assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
        )
        r.raise_for_status()
        raw = r.json()['choices'][0]['message']['content'].strip()

        if raw.startswith("```json"):
            raw = raw[7:].strip("` \n")
        elif raw.startswith("```"):
            raw = raw[3:].strip("` \n")

        parsed = json.loads(raw)
        if "category" not in parsed or "final_score" not in parsed:
            raise ValueError("Missing keys in fallback response")

        entry = parsed["category"]
        entry["score"] = float(entry["score"])
        entry["weight"] = float(entry["weight"])
        parsed["final_score"] = float(parsed["final_score"])

        return parsed
    except Exception as e:
        msg = f"NLP fallback scoring failed: {e}"
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        return {
            "category": {
                "category": category_name,
                "score": 0.0,
                "weight": 0.0,
                "justification": f"NLP fallback failed: {e}"
            },
            "final_score": 0.0
        }