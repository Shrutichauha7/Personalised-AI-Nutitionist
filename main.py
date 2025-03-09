from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load trained model & scaler
model = joblib.load("models/nutrition_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_enc = joblib.load("models/label_encoder.pkl")

@app.get("/")
def home():
    return {"message": "Welcome to AI Nutritionist API"}

@app.get("/predict_macronutrients")
def predict_macronutrients(age: int, gender: str, height: int, weight: int, activity_level: str):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == "male" else 0
    activity_mapping = {"sedentary": 0, "lightly_active": 1, "active": 2, "very_active": 3}
    activity_encoded = activity_mapping.get(activity_level.lower(), 0)

    # Prepare input
    input_data = np.array([[age, gender_encoded, height, weight, activity_encoded]])
    input_scaled = scaler.transform(input_data)

    # Predict Macronutrient Distribution
    prediction = model.predict_proba(input_scaled)

    return {
        "carbs_probability": float(prediction[0][0]),
        "protein_probability": float(prediction[0][1]),
        "fat_probability": float(prediction[0][2]),
    }
