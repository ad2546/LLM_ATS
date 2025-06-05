# utils/db_utils.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import mysql.connector
from mysql.connector import Error
from config import db_config

def get_connection():
    """
    Returns a new MySQL connection using parameters from config.db_config.
    Caller is responsible for closing the connection.
    """
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except Error as e:
        # In a production system, log this error
        raise