import mysql.connector
import os

def get_or_create_category_id(name: str, type_field: str):
    conn = mysql.connector.connect(
        host=os.getenv('MYSQL_HOST', 'localhost'),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', ''),
        database=os.getenv('MYSQL_DATABASE', 'LLM_Resume')
    )
    cursor = conn.cursor()
    # Try to find category
    cursor.execute("SELECT category_id FROM category WHERE name=%s", (name,))
    row = cursor.fetchone()
    if row:
        cat_id = row[0]
    else:
        cursor.execute("INSERT INTO category (name, type_field) VALUES (%s, %s)", (name, type_field))
        conn.commit()
        cat_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return cat_id