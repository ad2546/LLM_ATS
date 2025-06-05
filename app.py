# app.py

import logging
from flask import Flask
from flask_cors import CORS

from routes.jd_routes import jd_bp
from routes.score_routes import score_bp

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Flask App ----------
app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(jd_bp, url_prefix='')
app.register_blueprint(score_bp, url_prefix='')

@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    app.run(debug=True, port=5000)