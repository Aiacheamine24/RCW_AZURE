# Predict Routes
# External Imports
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Internal Imports

# Create Router
router = APIRouter(
    tags=["predict"],
    responses={404: {"description": "Not found"}},
)

# Create Route
from controllers.predict.predict import predict_from_data
from controllers.predict.predict_validation import PredictRequest
@router.post("/", status_code=200)
async def make_prediction(data: PredictRequest):
    print(data)
    return  predict_from_data(data)
    
