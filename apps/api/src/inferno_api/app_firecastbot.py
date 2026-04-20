from flask import Flask, jsonify
from flask_cors import CORS
import os

from .firecastbot import firecastbot_bp

app = Flask(__name__)
CORS(app)
app.register_blueprint(firecastbot_bp)

API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "6969"))


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host=API_HOST, port=API_PORT)
