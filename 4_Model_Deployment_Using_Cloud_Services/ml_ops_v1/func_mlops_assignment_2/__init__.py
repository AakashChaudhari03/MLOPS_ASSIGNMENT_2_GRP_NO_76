import json
import logging
import os
import azure.functions as func
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Processing a request.')

    if req.method == 'GET':
        return func.HttpResponse("Flask app running on Azure Functions!")

    if req.method == 'POST':
        try:
            # Get the input data from the request body
            req_body = req.get_json()
            sepal_length = float(req_body['sepal_length'])
            sepal_width = float(req_body['sepal_width'])
            petal_length = float(req_body['petal_length'])
            petal_width = float(req_body['petal_width'])

            # Prepare the input data for prediction
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(features)

            # Return the prediction result
            return func.HttpResponse(
                json.dumps({'prediction': int(prediction[0])}),
                mimetype="application/json"
            )
        except ValueError:
            return func.HttpResponse(
                json.dumps({'error': 'Invalid input'}),
                status_code=400,
                mimetype="application/json"
            )