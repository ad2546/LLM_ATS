# routes/score_routes.py
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Blueprint, request, jsonify
import os, logging, mysql.connector
from utils.pdf_utils import read_pdf_content
from utils.candidate_utils import extract_candidate_details
from utils.db_utils import get_connection
from services.score_service import score_resume_for_categories
from Tools.logs import save_log

logger = logging.getLogger(__name__)
score_bp = Blueprint('score_bp', __name__)

@score_bp.route('/recommended', methods=['GET'])
def recommended():
    jd_id = request.args.get('jd_id')
    resume_subfolder = request.args.get('resume_folder', '').strip()
    if not jd_id:
        msg = "Missing jd_id parameter"
        save_log("ERROR", msg, process="JD_Analysis")
        return jsonify({'error': msg}), 400

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        # 1) Fetch jd_text and category1_id only
        cursor.execute(
            "SELECT `jd_text`,`category1_id` FROM `job_description` WHERE `jd_id` = %s",
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
        cat_ids = [jd_row["category1_id"]] if jd_row["category1_id"] else []

        # 2) Convert category ID → category name
        category_names = []
        for cid in cat_ids:
            cursor.execute("SELECT `name` FROM `category` WHERE `category_id` = %s", (cid,))
            row = cursor.fetchone()
            if row:
                category_names.append(row["name"])

        # 3) Point to your ./resumes folder, optionally with subfolder
        base_dir = os.path.join(os.path.dirname(__file__), '../resumes')
        resumes_dir = os.path.join(base_dir, resume_subfolder) if resume_subfolder else base_dir
        if not os.path.isdir(resumes_dir):
            msg = f"Resumes folder not found at {resumes_dir} (subfolder: '{resume_subfolder}')"
            save_log("ERROR", msg, process="JD_Analysis")
            cursor.close()
            conn.close()
            return jsonify({'error': msg}), 500

        # 4) Build a map of existing candidates (email → candidate_id)
        cursor.execute("SELECT `candidate_id`, `email` FROM `candidate`")
        existing_map = {row["email"].lower(): row["candidate_id"] for row in cursor.fetchall()}

        summary = []

        # 5) Loop over each resume PDF — one HTTP call per file
        for fname in os.listdir(resumes_dir):
            if not fname.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(resumes_dir, fname)

            # 5a) Read PDF and extract plain text
            try:
                with open(full_path, "rb") as rf:
                    pdf_bytes = rf.read()
                resume_text = read_pdf_content(pdf_bytes)
            except Exception as e:
                msg = f"Failed to read resume '{fname}': {e}"
                logger.error(msg)
                save_log("ERROR", msg, process="JD_Analysis")
                continue

            # 5b) Extract candidate basics (email → key for upsert)
            cand_info = extract_candidate_details(resume_text)
            email = (cand_info.get("email") or "").lower().strip()

            # 5c) Insert or reuse existing candidate row
            candidate_id = None
            if email and email in existing_map:
                candidate_id = existing_map[email]
            else:
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

            # 5d) Score this single resume against the single category
            combined_scores = score_resume_for_categories(category_names, jd_text, resume_text)

            # 5e) Persist the single category score row
            cat_entry = combined_scores["category"]
            cname   = cat_entry["category"]
            sc      = cat_entry["score"]
            wt      = cat_entry["weight"]
            just    = cat_entry["justification"]

            if cname:
                cursor.execute(
                    """
                    INSERT INTO `category_score`
                      (`candidate_id`,`jd_id`,`category_name`,`score`,`weight`,`justification`,`updated_at`)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    """,
                    (candidate_id, jd_id, cname, sc, wt, just)
                )
                conn.commit()

            # 5f) Collect summary for this resume
            summary.append({
                "candidate_email": email,
                "candidate_id":    candidate_id,
                "resume_file":     fname,
                "category":       combined_scores["category"],
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