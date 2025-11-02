from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel
from typing import List

app = FastAPI()
model = joblib.load("crop_price_model.pkl")

class Crop(BaseModel):
    crop_name: str
    location: str
    temperature: float
    rainfall: float
    season: str
    historical_price: float
    quantity: float
    cost: float

@app.post("/predict")
def predict(crops: List[Crop]):
    df = pd.DataFrame([crop.dict() for crop in crops])
    features = ["crop_name", "location", "temperature", "rainfall", "season", "historical_price"]
    df["predicted_price"] = model.predict(df[features])
    df["revenue"] = df["predicted_price"] * df["quantity"]
    df["profit"] = df["revenue"] - df["cost"]
    return df[["crop_name", "quantity", "predicted_price", "revenue", "profit"]].to_dict(orient="records")
