# External Imports
from fastapi import status, HTTPException
import pickle
import pandas as pd
import numpy as np

# Internal Imports

# Load the model from file
with open("./Normalizer_Random Forest.pkl", "rb") as f:
    final_model = pickle.load(f)

# Predict Controller
# Predict Function
def predict_from_data(data: dict):
    # Use the loaded model to make predictions
    # Assuming final_model is a trained model
    # Prepare the data for prediction
    data = pd.DataFrame([data.dict()])
    # Make the prediction
    prediction = final_model.predict(data)
    # Return the prediction
    return {
        "prediction": prediction[0].tolist()
    }