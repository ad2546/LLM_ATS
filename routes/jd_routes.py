# routes/jd_routes.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging

from flask import Blueprint, request, jsonify
from utils.pdf_utils import read_pdf_content
from services.jd_service import analyze_jd_with_gpt, save_jd_to_db
from Tools.logs import save_log

logger = logging.getLogger(__name__)
jd_bp = Blueprint('jd_bp', __name__)

@jd_bp.route('/upload', methods=['POST'])
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
        file_bytes = file.read()
        jd_text = read_pdf_content(file_bytes)
        if not jd_text.strip():
            msg = "Job description PDF parsing returned no text"
            save_log("ERROR", msg, process="JD_Analysis")
            return jsonify({'error': msg}), 400

        # 2) Call DeepSeek to classify JD
        jd_info = analyze_jd_with_gpt(jd_text)
        categories = jd_info.get("categories", [])
        qualifications = jd_info.get("qualifications", "")
        requirements = jd_info.get("requirements", "")

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