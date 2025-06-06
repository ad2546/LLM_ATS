# services/score_service.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import logging
import requests
from typing import Tuple

from config import Settings
from Tools.logs import save_log

logger = logging.getLogger(__name__)
settings = Settings()

# Maximum tokens DeepSeek can accept
MAX_TOKENS = 65536

def approximate_token_count(text: str) -> int:
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
        logger.warning(f"Truncating resume from {resume_tokens} tokens → {budget} tokens.")
        save_log("WARNING", f"Resume truncated from {resume_tokens} to {budget} tokens", process="JD_Analysis")
        resume_text = truncate_text_to_tokens(resume_text, budget)

    return jd_text, resume_text

def score_resume_for_categories(category_names: list, jd_text: str, resume_text: str) -> dict:
    """
    Calls DeepSeek once to score 'resume_text' against both categories.
    If the raw resume is too large (> MAX_TOKENS), log and skip sending to DeepSeek.
    Returns a dict:
      {
        "categories": [
          {"category": "<cat1>", "score": <0–100>, "weight": <0.0–1.0>, "justification": "..."},
          {"category": "<cat2>", "score": <0–100>, "weight": <0.0–1.0>, "justification": "..."}
        ],
        "final_score": <0–100>
      }
    """
    # Ensure exactly two slots
    if len(category_names) == 1:
        category_names.append("")  # second slot empty
    elif len(category_names) == 0:
        category_names = ["", ""]

    # Check raw resume size before any truncation
    raw_resume_tokens = approximate_token_count(resume_text)
    if raw_resume_tokens > MAX_TOKENS:
        msg = f"Resume too large ({raw_resume_tokens} tokens) – skipping DeepSeek scoring."
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        # Return zeroed fallback for both categories
        return {
            "categories": [
                {"category": category_names[0], "score": 0.0, "weight": 0.0, "justification": "Resume too large to process."},
                {"category": category_names[1], "score": 0.0, "weight": 0.0, "justification": "Resume too large to process."}
            ],
            "final_score": 0.0
        }

    # 1) Truncate JD + resume to fit under MAX_TOKENS
    jd_trimmed, resume_trimmed = prepare_jd_and_resume(jd_text, resume_text)

    prompt = f"""
You are an AI that scores a candidate’s resume against two categories within a single job description.
Prioritize “experience” with highest weight, then “education,” then “skills.”

Categories:
1. "{category_names[0]}"
2. "{category_names[1]}"

Full Job Description:
{jd_trimmed}

Candidate Resume:
{resume_trimmed}

Respond ONLY as JSON, exactly in this form (no extra comments):

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

        # Validate
        if "categories" not in parsed or "final_score" not in parsed:
            raise ValueError("Missing keys in JSON response")

        # Convert types
        for entry in parsed["categories"]:
            entry["score"] = float(entry["score"])
            entry["weight"] = float(entry["weight"])
        parsed["final_score"] = float(parsed["final_score"])

        return parsed

    except Exception as e:
        msg = f"Error calling DeepSeek combined scoring: {str(e)}"
        logger.error(msg)
        save_log("ERROR", msg, process="JD_Analysis")
        # Return zeroed fallback
        return {
            "categories": [
                {"category": category_names[0], "score": 0.0, "weight": 0.0,
                 "justification": f"Error: {str(e)}"},
                {"category": category_names[1], "score": 0.0, "weight": 0.0,
                 "justification": f"Error: {str(e)}"}
            ],
            "final_score": 0.0
        }