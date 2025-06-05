# services/score_service.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import logging
import requests

from config import Settings
from Tools.logs import save_log

logger = logging.getLogger(__name__)
settings = Settings()

def score_resume_for_categories(category_names: list, jd_text: str, resume_text: str) -> dict:
    """
    Calls DeepSeek a single time to score 'resume_text' against two categories.
    Returns a dict:
    {
      "categories": [
        {
          "category": "<cat1>",
          "score": <0-100>,
          "weight": <0.0-1.0>,
          "justification": "<…>"
        },
        {
          "category": "<cat2>",
          "score": <0-100>,
          "weight": <0.0-1.0>,
          "justification": "<…>"
        }
      ],
      "final_score": <0-100>
    }
    """
    # Ensure we always have exactly two slots
    if len(category_names) == 1:
        category_names.append("")  # second slot empty
    elif len(category_names) == 0:
        category_names = ["", ""]

    prompt = f"""
You are an AI that scores a candidate’s resume against two categories within a single job description.
Prioritize “experience” in the resume with the highest weight, then “education,” then “skills.”

Categories to score:
1. "{category_names[0]}"
2. "{category_names[1]}"

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
- “justification” should be a brief (1–2 sentences) rationale for that category.
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
        return {
            "categories": [
                {"category": category_names[0], "score": 0.0, "weight": 0.0, "justification": f"Error: {str(e)}"},
                {"category": category_names[1], "score": 0.0, "weight": 0.0, "justification": f"Error: {str(e)}"}
            ],
            "final_score": 0.0
        }