# Tools/logs.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import logging
import mysql.connector
from config import db_config

logger = logging.getLogger(__name__)

def save_log(log_type: str, message: str, process: str="JD_Analysis"):
    """
    Inserts a row into the 'logs' table with (log_type, process, message).
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO `logs` (`log_type`, `process`, `message`) VALUES (%s, %s, %s)",
            (log_type, process, message)
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"[LOGGED] {process} - {log_type}: {message}")
    except Exception as e:
        # If logging to DB fails, at least log to console
        logger.exception(f"Failed to write to logs table: {e}")