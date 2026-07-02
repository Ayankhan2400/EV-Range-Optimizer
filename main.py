from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. Initialize the FastAPI web server instance
app = FastAPI(
    title="AuraCharge AI Core Engine", 
    description="Production API endpoint serving the Random Forest EV Range Regressor",
    version="1.0.0"
)

# 2. Safely load our pre-trained AI brain asset into server memory
try:
    model = joblib.load('ev_model.pkl')
    print("🧠 Success: AI Model successfully loaded into server memory!")
except Exception as e:
    print(f"❌ Error: Could not find or load 'ev_model.pkl'. Details: {e}")
    model = None

# 3. Define the strict Data Schema for incoming mobile requests
class EVTelemetryInput(BaseModel):
    speed_kmh: float
    temperature_c: float
    elevation_change_m: float
    driving_style_score: float

# 4. Create a basic Welcome Endpoint (Health Check)
@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "Welcome to the AuraCharge AI Telemetry Server. Use the /predict endpoint for AI inference."
    }

# 5. Create the Core AI Inference Endpoint
@app.post("/predict")
def predict_ev_drain(data: EVTelemetryInput):
    # Ensure the model is completely loaded before attempting analysis
    if not model:
        raise HTTPException(status_code=500, detail="Machine Learning model asset is missing on the server.")
    
    try:
        # Convert incoming JSON data into a formal structured Pandas DataFrame
        input_df = pd.DataFrame([{
            'Speed_kmh': data.speed_kmh,
            'Temperature_C': data.temperature_c,
            'Elevation_Change_m': data.elevation_change_m,
            'Driving_Style_Score': data.driving_style_score
        }])
        
        # Execute the Random Forest prediction calculation
        predicted_wh_km = model.predict(input_df)[0]
        
        # Return a structured JSON response back over the web
        return {
            "success": True,
            "predicted_drain_rate_wh_km": round(predicted_wh_km, 2),
            "units": "Watt-hours per kilometer"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference pipeline execution failure: {str(e)}")