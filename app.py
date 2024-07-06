# APP File For MY FastAPI Application
# External Imports
from fastapi import FastAPI
import dotenv
import os

# Internal Imports

# Load environment variables
dotenv.load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), './config/config.env')
)

ENV = os.getenv('ENV')

# Create FastAPI Application
app = FastAPI(
    title="My FastAPI Application",
    description="This is a FastAPI Application",
    version="0.1.0",
    debug=True if ENV == "development" else False
)

# Create Routes
# Predict Route
from routes import predict
app.include_router(predict.router, prefix="/predict")