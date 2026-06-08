import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from backend.app.core import config

# Initialize the router instance to be included into main app stack later
router = APIRouter()

# --- INPUT DATA VALIDATION SCHEMAS (Pydantic Matrix) ---
class StudentFeaturesInput(BaseModel):
    school: str = Field(..., example="GP")
    sex: str = Field(..., example="F")
    age: int = Field(..., ge=15, le=22, example=18)
    address: str = Field(..., example="U")
    famsize: str = Field(..., example="GT3")
    pstatus: str = Field(..., example="T")
    medu: int = Field(..., ge=0, le=4, example=4)
    fedu: int = Field(..., ge=0, le=4, example=3)
    mjob: str = Field(..., example="health")
    fjob: str = Field(..., example="services")
    reason: str = Field(..., example="home")
    guardian: str = Field(..., example="mother")
    traveltime: int = Field(..., ge=1, le=4, example=1)
    studytime: int = Field(..., ge=1, le=4, example=2)
    failures: int = Field(..., ge=0, le=4, example=0)
    schoolsup: str = Field(..., example="yes")
    famsup: str = Field(..., example="no")
    paid: str = Field(..., example="no")
    activities: str = Field(..., example="no")
    nursery: str = Field(..., example="yes")
    higher: str = Field(..., example="yes")
    internet: str = Field(..., example="yes")
    romantic: str = Field(..., example="no")
    famrel: int = Field(..., ge=1, le=5, example=4)
    freetime: int = Field(..., ge=1, le=5, example=3)
    goout: int = Field(..., ge=1, le=5, example=4)
    dalc: int = Field(..., ge=1, le=5, example=1)
    walc: int = Field(..., ge=1, le=5, example=1)
    health: int = Field(..., ge=1, le=5, example=3)
    absences: int = Field(..., ge=0, le=93, example=4)

class ChatMessageInput(BaseModel):
    session_id: str = Field(..., description="Unique tracking identifier.")
    message: str = Field(..., description="User query payload.")

# --- ROUTE OPERATIONS ---
@router.post("/predict", status_code=status.HTTP_200_OK)
async def run_predictive_inference(payload: StudentFeaturesInput):
    if config.model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline down.")
    try:
        raw_features = payload.model_dump()
        
        # Realignment mapping to handle uppercase scikit-learn header schemas
        key_mapping = {
            "medu": "Medu", "fedu": "Fedu", "mjob": "Mjob", "fjob": "Fjob",
            "pstatus": "Pstatus", "dalc": "Dalc", "walc": "Walc"
        }
        final_features = {key_mapping.get(k, k): v for k, v in raw_features.items()}
        
        input_df = pd.DataFrame([final_features])
        prediction = config.model_pipeline.predict(input_df)
        
        bounded_score = max(0.0, min(20.0, float(prediction[0])))
        return {
            "prediction_version": "v1_behavioral_ridge",
            "predicted_g3_score": round(bounded_score, 2),
            "risk_level": "HIGH" if bounded_score < 10 else "MODERATE" if bounded_score < 14 else "LOW"
        }
    except Exception as e:
        config.logger.error(f"Inference failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing failure.")

@router.post("/chat", status_code=status.HTTP_200_OK)
async def execute_stateful_chat(payload: ChatMessageInput):
    session_id = payload.session_id
    user_utterance = payload.message

    # Wrap payload parameters into standardized role mappings
    user_message_frame = {"role": "user", "content": user_utterance}
    
    # 1. WRITE to storage via the interface contract (.set)
    config.memory_store.set(session_id, user_message_frame)
    
    # 2. READ current updated thread history state back via the interface (.get)
    history_thread = config.memory_store.get(session_id)
    context_depth = len(history_thread)
    
    # Generate system responses
    response_content = (
        f"Understood. Analyzing student parameters across abstract chat interface thread. "
        f"The isolated contract layer safely preserves {context_depth} runtime context frames."
    )
    
    assistant_message_frame = {"role": "assistant", "content": response_content}
    
    # 3. WRITE the final server execution block back to memory storage
    config.memory_store.set(session_id, assistant_message_frame)
    
    return {
        "session_id": session_id,
        "history": config.memory_store.get(session_id),
        "status": "synchronized"
    }