# routes/score_routes.py
import os, sys, logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Blueprint, request, jsonify
from utils.db_utils import get_connection
from services.score_service import recommend_resumes_by_embedding
from Tools.logs import save_log
from utils.candidate_utils import extract_candidate_details, upsert_candidate
from utils.pdf_utils import read_pdf_content  # Adjust the import if your util is named differently

logger = logging.getLogger(__name__)
score_bp = Blueprint('score_bp', __name__)

@score_bp.route('/recommended', methods=['GET'])
def recommended():
    jd_id = request.args.get('jd_id')
    resume_folder = request.args.get('resume_folder')
    if not jd_id:
        msg = "Missing jd_id parameter"
        save_log("ERROR", msg, process="Score_Recommendation")
        return jsonify({'error': msg}), 400
    if not resume_folder:
        msg = "Missing resume_folder parameter"
        save_log("ERROR", msg, process="Score_Recommendation")
        return jsonify({'error': msg}), 400
    try:
        # Fetch JD details from DB
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT jd_text, category_detected, qualifications, requirements
            FROM job_description
            WHERE jd_id = %s
        """, (jd_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            msg = f"Job description {jd_id} not found"
            save_log("ERROR", msg, process="Score_Recommendation")
            return jsonify({'error': msg}), 404

        jd_text = row['jd_text']
        category = row.get('category_detected', '') or ''
        qualifications = row.get('qualifications', '') or ''
        requirements = row.get('requirements', '') or ''

        from services.score_service import score_all_resumes_in_folder
        recommendations = score_all_resumes_in_folder(
            jd_text, resume_folder, category, qualifications, requirements
        )
        # --- DB insert block ---
        conn = get_connection()
        cursor = conn.cursor()
        for rec in recommendations:
            candidate_email = rec.get("candidate_email", "")
            candidate_id = None

            # Try to upsert candidate by parsing resume if possible
            if candidate_email and rec.get("resume_path"):
                try:
                    with open(rec["resume_path"], "rb") as f:
                        file_bytes = f.read()
                    resume_text = read_pdf_content(file_bytes)
                    cand_info = extract_candidate_details(resume_text)
                    candidate_id = upsert_candidate(cand_info, rec["resume_path"])
                except Exception as e:
                    logger.error(f"Failed candidate upsert: {e}")
                    save_log("ERROR", f"Failed candidate upsert: {e}", process="Score_Recommendation")
            else:
                # Fallback: lookup by email
                try:
                    cursor.execute("SELECT candidate_id FROM candidate WHERE candidate_email=%s", (candidate_email,))
                    c_row = cursor.fetchone()
                    if c_row:
                        candidate_id = c_row[0]
                except Exception as e:
                    logger.error(f"Failed candidate lookup: {e}")

            cursor.execute("""
                INSERT INTO jd_score (
                    jd_id, candidate_id, category_score,
                    qualifications_score, requirements_score, final_score, reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    category_score=VALUES(category_score),
                    qualifications_score=VALUES(qualifications_score),
                    requirements_score=VALUES(requirements_score),
                    final_score=VALUES(final_score),
                    reason=VALUES(reason)
            """, (
                jd_id,
                candidate_id,
                rec.get("category_score"),
                rec.get("qualifications_score"),
                rec.get("requirements_score"),
                rec.get("final_score"),
                rec.get("reason", ""),
            ))
        conn.commit()
        cursor.close()
        conn.close()
        # --- END DB insert block ---
        save_log("INFO", f"Completed embedding recommendation for jd_id={jd_id}", process="Score_Recommendation")
        fit_summaries = [r.get('fit_summary', '') for r in recommendations if 'fit_summary' in r][:3]
        summary = " | ".join(fit_summaries) if fit_summaries else "No resumes scored for this job description."
        return jsonify({
            "job_id": jd_id,
            "resume_folder": resume_folder,
            "results": recommendations,
            "count": len(recommendations),
            "summary": summary
        })

    except Exception as e:
        msg = f"Unhandled exception in /recommended: {e}"
        logger.exception(msg)
        save_log("ERROR", msg, process="Score_Recommendation")
        return jsonify({'error': msg}), 500
