import os

from flask import Flask, jsonify
from flask_cors import CORS

from .firecastbot import firecastbot_bp

app = Flask(__name__)
CORS(app, max_age=600)
# Hard ceiling on incoming request bodies (protects all upload endpoints)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB
app.register_blueprint(firecastbot_bp)

API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "6969"))


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT)
