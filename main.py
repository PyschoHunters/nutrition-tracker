from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with the specific origin (e.g., "http://localhost:3000" for local development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model once when the app starts
model = joblib.load("calories_burnt.sav")

class PredictionInput(BaseModel):
    gender: str
    age: int
    height: float
    weight: float
    duration: float
    heartRate: float
    bodyTemp: float

@app.get("/")
def root():
    return {"message": "Calorie Burn Prediction API is live 🎯"}

@app.post("/predict")
def predict(data: PredictionInput):
    # Convert gender to numeric
    gender_code = 0 if data.gender.lower() == "male" else 1

    # Create input array for prediction
    input_array = np.array([[gender_code, data.age, data.height, data.weight,
                             data.duration, data.heartRate, data.bodyTemp]])

    # Make prediction
    prediction = float(round(model.predict(input_array)[0], 2))

    return {"calories_burnt": prediction}
