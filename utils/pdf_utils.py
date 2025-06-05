# utils/pdf_utils.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
import fitz  # PyMuPDF
from Tools.logs import save_log

logger = logging.getLogger(__name__)

def read_pdf_content(file_bytes: bytes) -> str:
    """
    Given raw PDF bytes, return the concatenated text of all pages.
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