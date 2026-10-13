import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_architecture import TKARiskModel

# Load the trained hospital-grade ensemble model
model = joblib.load("model.pkl")

# Initialize FastAPI app
app = FastAPI(
    title="TKA Clinical Risk Predictor API",
    description="Hospital-Grade Pre-operative Risk & Decision Support System for Total Knee Arthroplasty (TKA)",
    version="3.0.0"
)

# Allow frontend / clinic EHR to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input patient schema (28 pre-operative clinical parameters)
class PatientData(BaseModel):
    oks_t0_score: float
    oks_t0_pain: float
    oks_t0_night_pain: float
    oks_t0_washing: float
    oks_t0_transport: float
    oks_t0_walking: float
    oks_t0_standing: float
    oks_t0_limping: float
    oks_t0_kneeling: float
    oks_t0_work: float
    oks_t0_confidence: float
    oks_t0_shopping: float
    oks_t0_stairs: float
    t0_mobility: float
    t0_self_care: float
    t0_activity: float
    t0_discomfort: float
    t0_anxiety: float
    t0_eq_vas: float
    age_band: int
    gender: int
    heart_disease: int
    high_bp: int
    diabetes: int
    stroke: int
    lung_disease: int
    t0_disability: float
    t0_previous_surgery: float

@app.get("/")
def root():
    return {
        "service": "TKA Clinical Decision Support System",
        "version": "3.0.0",
        "status": "Online",
        "ensemble": "LightGBM + XGBoost with Orthopedic Clinical Feature Engineering"
    }

@app.post("/predict-risk")
def predict_risk(patient: PatientData):
    # Convert patient Pydantic model to dictionary
    patient_dict = patient.dict()
    
    # Run full clinical diagnostic analysis
    analysis = model.analyze_patient(patient_dict)
    return analysis
