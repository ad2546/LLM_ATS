# services/score_service.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
from utils.embeddings import embed_text, get_resume_index
from utils.pdf_utils import read_pdf_content
from utils.candidate_utils import extract_candidate_details
from Tools.logs import save_log
from utils.candidate_utils import save_score_to_jd_score
logger = logging.getLogger(__name__)
from utils.db_utils import get_connection

# Utility to normalize resume section (handle list, str, None)
def normalize_section(section):
    """Ensures section is a string, even if a list or None."""
    if isinstance(section, list):
        return " ".join(str(item) for item in section if item)
    elif isinstance(section, str):
        return section
    elif section is None:
        return ""
    else:
        return str(section)


def extract_keywords(text):
    """Extract keywords from text by splitting on common delimiters and stripping whitespace."""
    if not text:
        return []
    import re
    # Split by commas, semicolons, pipes, slashes, periods, dashes, or newlines
    tokens = re.split(r'[,\n;|/.\-]+', text)
    # Clean and filter out empty tokens
    keywords = [token.strip().lower() for token in tokens if token.strip()]
    return keywords


import json
import google.generativeai as genai

def score_resume_with_gemini_flash(jd_category, jd_requirements, jd_qualifications, resume_text):
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
Given the following job description details and a candidate's resume, score how well the candidate matches each section on a scale from 0 to 10 (0 = no match, 10 = perfect match). Give only numbers and a short reason.

Job Category:
{jd_category}

Job Requirements:
{jd_requirements}

Job Qualifications:
{jd_qualifications}

Resume:
{resume_text}

Return the result in JSON like this:
{{
  "category_score": number,
  "requirements_score": number,
  "qualifications_score": number,
  "final_score": number,
  "reason": "Short summary why"
}}
"""
    response = model.generate_content(prompt)
    text_response = response.text
    try:
        result = json.loads(text_response)
    except Exception:
        import re
        match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
        else:
            raise ValueError("Could not parse Gemini output as JSON.")
    return result


def score_all_resumes_in_folder(
        jd_text: str,
        folder_path: str,
        jd_category: str,
        jd_qualifications: str,
        jd_requirements: str
    ) -> list:
    """
    Scores all resumes in the given folder against the JD components using Gemini 1.5 Flash via Vertex AI.
    """
    import glob
    results = []
    resume_dir = os.path.abspath(os.path.join(os.getcwd(), "resumes", folder_path))
    pdf_files = glob.glob(os.path.join(resume_dir, "*.pdf"))
    for path in pdf_files:
        abs_path = os.path.abspath(path)
        try:
            with open(abs_path, 'rb') as f:
                pdf_bytes = f.read()
            text = read_pdf_content(pdf_bytes)
            info = extract_candidate_details(text)  # Should return dict with 'experience', 'projects', 'skills', etc

            # Concatenate all main sections for full resume text
            full_resume_text = " ".join([
                normalize_section(info.get("experience", "")),
                normalize_section(info.get("projects", "")),
                normalize_section(info.get("skills", "")),
                normalize_section(info.get("summary", "")),
                normalize_section(info.get("education", "")),
                normalize_section(info.get("certifications", "")),
                normalize_section(info.get("other", "")),
            ]).strip()

            gemini_result = score_resume_with_gemini_flash(
                jd_category=jd_category,
                jd_requirements=jd_requirements,
                jd_qualifications=jd_qualifications,
                resume_text=full_resume_text
            )

            results.append({
                'candidate_email': info.get('email'),
                'candidate_name': info.get('name') or info.get('email'),
                'resume_path': abs_path,
                'resume_filename': os.path.basename(abs_path),
                'category_score': gemini_result.get('category_score'),
                'requirements_score': gemini_result.get('requirements_score'),
                'qualifications_score': gemini_result.get('qualifications_score'),
                'final_score': gemini_result.get('final_score'),
                'reason': gemini_result.get('reason')
            })

            conn = get_connection()
            cursor = conn.cursor(dictionary=True)
           
        except Exception as e:
            logger.error(f"Failed to process resume '{abs_path}': {e}")
            save_log("ERROR", f"Resume load error: {e}", process="JD_Analysis")
    results.sort(key=lambda x: x['final_score'] if x['final_score'] is not None else 0, reverse=True)
    return results





def recommend_resumes_by_embedding(jd_text: str, top_k: int = 5) -> list:
    """
    Embed a job description and return top_k resumes most similar via FAISS.

    Returns a list of dicts:
      [{
         'candidate_email': str,
         'resume_path': str,
         'score': float
       }, ...]
    """
    try:
        # Embed the JD text
        vec = embed_text(jd_text)
        # Search the resume index
        results = get_resume_index().search(vec, k=top_k)
        recommendations = []
        for path, score in results:
            try:
                # Load and extract resume text
                with open(path, 'rb') as f:
                    pdf_bytes = f.read()
                text = read_pdf_content(pdf_bytes)
                info = extract_candidate_details(text)
                recommendations.append({
                    'candidate_email': info.get('email'),
                    'resume_path': path,
                    'score': score
                })
            except Exception as e:
                logger.error(f"Failed to process resume '{path}': {e}")
                save_log("ERROR", f"Resume load error: {e}", process="JD_Analysis")
        return recommendations
    except Exception as e:
        logger.error(f"Embedding recommendation failed: {e}")
        save_log("ERROR", str(e), process="JD_Analysis")
        return []
