import pandas as pd
from fastapi import HTTPException
from backend.app.core import config

def run_prediction(payload_dict: dict) -> dict:
    """
    Handles data manipulation, feature realignment, and model 
    inference completely decoupled from the HTTP routing protocol.
    """
    if config.model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is uninitialized.")
        
    try:
        # Realignment mapping to handle uppercase scikit-learn header schemas
        key_mapping = {
            "medu": "Medu", "fedu": "Fedu", "mjob": "Mjob", "fjob": "Fjob",
            "pstatus": "Pstatus", "dalc": "Dalc", "walc": "Walc"
        }
        final_features = {key_mapping.get(k, k): v for k, v in payload_dict.items()}
        
        # Convert to 2D DataFrame row for Scikit-Learn compliance
        input_df = pd.DataFrame([final_features])
        prediction = config.model_pipeline.predict(input_df)
        
        bounded_score = max(0.0, min(20.0, float(prediction[0])))
        
        return {
            "prediction_version": "v1_behavioral_ridge",
            "predicted_g3_score": round(bounded_score, 2),
            "risk_level": "HIGH" if bounded_score < 10 else "MODERATE" if bounded_score < 14 else "LOW"
        }
    except Exception as e:
        config.logger.error(f"Inference Engine failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal ML processing failure.")