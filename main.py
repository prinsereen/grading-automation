from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    jaccard: float
    spam: int
    grammar: int
    sentiment: int
    violence: int
    sexual: int
    scholarly: int

# Load the trained linear regression model
model = pickle.load(open('linear_regression_model.pkl', 'rb'))

# Create the FastAPI app
app = FastAPI()

# Define the API endpoint for prediction
@app.post('/grade')
def make_prediction(input_data: ModelInput):
    # Extract the input features
    input_features = [input_data.jaccard, input_data.spam, input_data.grammar, input_data.sentiment,
                      input_data.violence, input_data.sexual, input_data.scholarly]

    # Make the prediction
    prediction = model.predict([input_features])

    # Return the prediction as a response
    return {'prediction': float(prediction[0])}

