from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, including OPTIONS
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("housing_price_model.pkl")


# Define input schema
class HouseFeatures(BaseModel):
    SquareFeet: float
    Bedrooms: int
    Bathrooms: int
    YearBuilt: int
    Neighborhood: str

# Define binning function for YearBuilt
def bin_year_built(year):
    if year <= 1950:
        return 1
    elif year <= 1980:
        return 2
    elif year <= 2000:
        return 3
    else:
        return 4

@app.post("/predict/")
def predict_price(features: HouseFeatures):
    # Convert boolean to integer

    suburb = 1 if features.Neighborhood == "Suburb" else 0
    urban = 1 if features.Neighborhood == "Urban" else 0
    print(f"suburb {suburb}\n")
    print(f"urban {urban}\n")

    # Replicate feature engineering
    squarefeet_bedrooms = features.SquareFeet * features.Bedrooms
    squarefeet_bathrooms = features.SquareFeet * features.Bathrooms
    squarefeet_squared = features.SquareFeet ** 2
    yearbuilt_squared = features.YearBuilt ** 2
    log_squarefeet = np.log1p(features.SquareFeet)
    yearbuilt_bin = bin_year_built(features.YearBuilt)

    # Create feature array in the same order as training data
    input_features = np.array([
        features.SquareFeet, features.Bedrooms, features.Bathrooms, features.YearBuilt,
        suburb, urban,
        squarefeet_bedrooms, squarefeet_bathrooms, squarefeet_squared, yearbuilt_squared,
        log_squarefeet, yearbuilt_bin
    ]).reshape(1, -1)

    print(f"Input Features Shape: {input_features.shape}")

    # Ensure feature count matches trained model
    expected_features = model.n_features_in_
    if input_features.shape[1] != expected_features:
        raise HTTPException(status_code=400, detail=f"Feature mismatch: Expected {expected_features}, got {input_features.shape[1]}")

    # Make prediction (model expects log price, so we exponentiate)
    predicted_log_price = model.predict(input_features)[0]
    predicted_price = np.expm1(predicted_log_price)  # Reverse log transformation

    return {"predicted_price": predicted_price}
