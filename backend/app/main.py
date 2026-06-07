import os
import logging
from typing import Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =====================================================================
# 1. LOGGING & SYSTEM CONFIGURATION
# =====================================================================
# Set up professional server logging to monitor requests and track errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("student-performance-api")

# Initialize the FastAPI core application instance
app = FastAPI(
    title="Student Performance Predictive Engine",
    description="Asynchronous MLOps inference API utilizing regularized behavioral modeling.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) so your upcoming React UI 
# can securely communicate with this API even if they run on different ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, swap "*" out for your explicit UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the absolute system path to our serialized joblib model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "behavioral_ridge_v1.joblib")
model_pipeline = None

# =====================================================================
# 2. LIFESPAN EVENT HOOKS (The $500K MLOps Pattern)
# =====================================================================
# A junior engineer loads the model *inside* the route function, meaning the file 
# is re-read from disk on EVERY single request—killing performance.
# We load the model ONCE globally when the server boots up so it stays warm in memory.
@app.on_event("startup")
def load_ml_model():
    global model_pipeline
    logger.info("⏳ Initializing system startup: Loading serialized ML pipeline...")
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Critical Error: Model binary missing at path: {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        logger.info("📦 ML Pipeline safely decompressed and mounted into server RAM.")
    except Exception as e:
        logger.critical(f"❌ Failed to load model pipeline: {str(e)}")
        raise e

# =====================================================================
# 3. PYDANTIC INPUT DATA VALIDATION MATRIX
# =====================================================================
# This class defines the exact shape, type, and bounds of data the API accepts.
# If a client sends invalid types, FastAPI automatically catches it and returns 
# a clean 422 Unprocessable Entity error before it ever reaches our model.
class StudentFeaturesInput(BaseModel):
    school: str = Field(..., description="Student's school ('GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)")
    sex: str = Field(..., description="Student's sex ('F' - female or 'M' - male)")
    age: int = Field(..., ge=15, le=22, description="Age of student (15 to 22)")
    address: str = Field(..., description="Student's home address type ('U' - urban or 'R' - rural)")
    famsize: str = Field(..., description="Family size ('LE3' - less or equal to 3 or 'GT3' - greater than 3)")
    pstatus: str = Field(..., description="Parent's cohabitation status ('T' - living together or 'A' - apart)")
    medu: int = Field(..., ge=0, le=4, description="Mother's education level (0 to 4)")
    fedu: int = Field(..., ge=0, le=4, description="Father's education level (0 to 4)")
    mjob: str = Field(..., description="Mother's job category")
    fjob: str = Field(..., description="Father's job category")
    reason: str = Field(..., description="Reason to choose this school")
    guardian: str = Field(..., description="Student's guardian ('mother', 'father' or 'other')")
    traveltime: int = Field(..., ge=1, le=4, description="Home to school travel time (1 to 4)")
    studytime: int = Field(..., ge=1, le=4, description="Weekly study time (1 to 4)")
    failures: int = Field(..., ge=0, le=4, description="Number of past class failures (0 to 4)")
    schoolsup: str = Field(..., description="Extra educational support ('yes' or 'no')")
    famsup: str = Field(..., description="Family educational support ('yes' or 'no')")
    paid: str = Field(..., description="Extra paid classes within the course subject ('yes' or 'no')")
    activities: str = Field(..., description="Extra-curricular activities ('yes' or 'no')")
    nursery: str = Field(..., description="Attended nursery school ('yes' or 'no')")
    higher: str = Field(..., description="Wants to take higher education ('yes' or 'no')")
    internet: str = Field(..., description="Internet access at home ('yes' or 'no')")
    romantic: str = Field(..., description="With a romantic relationship ('yes' or 'no')")
    famrel: int = Field(..., ge=1, le=5, description="Quality of family relationships (1 to 5)")
    freetime: int = Field(..., ge=1, le=5, description="Free time after school (1 to 5)")
    goout: int = Field(..., ge=1, le=5, description="Going out with friends (1 to 5)")
    dalc: int = Field(..., ge=1, le=5, description="Workday alcohol consumption (1 to 5)")
    walc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption (1 to 5)")
    health: int = Field(..., ge=1, le=5, description="Current health status (1 to 5)")
    absences: int = Field(..., ge=0, le=93, description="Number of school absences (0 to 93)")

    # Provide a mock sample for interactive API documentation UI (/docs)
    class Config:
        json_schema_extra = {
            "example": {
                "school": "GP", "sex": "F", "age": 18, "address": "U", "famsize": "GT3", "pstatus": "T",
                "medu": 4, "fedu": 3, "mjob": "health", "fjob": "services", "reason": "home", "guardian": "mother",
                "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "yes", "famsup": "no", "paid": "no",
                "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
                "famrel": 4, "freetime": 3, "goout": 4, "dalc": 1, "walc": 1, "health": 3, "absences": 4
            }
        }

# =====================================================================
# 4. ROUTE ROUTERS (Endpoints)
# =====================================================================

@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simpler cluster heart-beat check. Used by load balancers and Nginx 
    to verify that our microservice container is up and breathing.
    """
    return {"status": "healthy", "model_loaded": model_pipeline is not None}


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_student_grade(payload: StudentFeaturesInput):
    """
    Asynchronous Inference Gateway. Converts incoming Pydantic payload 
    into a pandas DataFrame row, pipes it through the serialized pipeline,
    and yields the predicted final continuous mark (G3).
    """
    if model_pipeline is None:
        logger.critical("Inference request denied: ML model is not mounted.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is currently unavailable on the server backend."
        )
    
    try:
        # 1. Cast validated Pydantic parameters into standard dictionary format
        raw_features: Dict[str, Any] = payload.model_dump()
        
        # Direct key mapping to match the exact training column names expected by scikit-learn
        key_mapping = {
            "medu": "Medu",
            "fedu": "Fedu",
            "mjob": "Mjob",
            "fjob": "Fjob",
            "pstatus": "Pstatus",
            "dalc": "Dalc",
            "walc": "Walc"
        }
        # Force all incoming keys to lowercase to strictly match training data headers
        normalized_features = {key_mapping.get(k, k): v for k, v in raw_features.items()}
        
        # 2. Convert dict into a Single-Row Pandas DataFrame.
        # Scikit-learn Pipelines demand explicit 2D structures with precise column headers
        input_df = pd.DataFrame([normalized_features])
        
        # 3. Trigger asynchronous multi-thread safe predictive calculations
        logger.info(f"Processing inference for student input matrix. Age: {payload.age}, Failures: {payload.failures}")
        prediction = model_pipeline.predict(input_df)
        
        # 4. Extract scalar output and enforce boundaries. 
        # A student grade cannot logically exceed 20 or dip past 0 marks.
        final_calculated_score = float(prediction[0])
        bounded_score = max(0.0, min(20.0, final_calculated_score))
        
        # 5. Dispatch payload response matrix back to the client
        return {
            "prediction_version": "v1_behavioral_ridge",
            "predicted_g3_score": round(bounded_score, 2),
            "status": "success"
        }
        
    except Exception as error:
        logger.error(f"❌ Error compiling real-time prediction matrix: {str(error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(error)}"
        )