import os
import joblib
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from model_architecture import TKARiskModel

# Initialize Flask app
app = Flask(__name__)

# Allow frontend to call API from any domain (Next.js, React, mobile, localhost)
CORS(app)

# Load trained hospital-grade model using absolute path
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "model.pkl"
model = joblib.load(str(model_path))

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "TKA Clinical Decision Support System",
        "version": "3.0.0",
        "status": "Online",
        "framework": "Flask WSGI (Native PythonAnywhere)",
        "docs": "Send POST request to /predict-risk with patient JSON data"
    })

@app.route("/predict-risk", methods=["POST"])
def predict_risk():
    patient_dict = request.get_json(force=True, silent=True) or {}
    analysis = model.analyze_patient(patient_dict)
    return jsonify(analysis)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
