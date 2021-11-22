from flask import Flask, jsonify, request
import numpy as np 
import pandas as pd
import pickle
#Swagger is an api that automatically generates UI for frontend based on certain keywords
from flasgger import Swagger

#creating instance of flask app
app = Flask(__name__)
#Initialising the flask app into Swagger to generate the UI
Swagger(app)

pickle_load = open('./classifier.pkl', 'rb')
classifier = pickle.load(pickle_load)

@app.route('/')
def home():
    return "WELCOME  THIS IS A BANK NOTE ML PREDICTION MODEL"

@app.route('/predict', methods=["Get"])
def predict_note_authentication():

    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
        - name:variance
        - in: query
        - type: number
        - required: true
        - name:skewness
        - in: query
        - type: number
        - required: true
        - name:curtosis
        - in: query
        - type: number
        - required: true
        - name:entropy
        - in: query
        - type: number
        - required: true
    responses:
        200:
            description: The output values
        

    """

    variance =request.args.get('variance')
    skewness =request.args.get('skewness')
    curtosis =request.args.get('curtosis')
    entropy =request.args.get('entropy')

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)

    return "The prediction value is "+ str(prediction)

@app.route('/predict_file', methods=["POST"])
def predict_file_test():

    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
        - name:file
        - in: formData
        - type: file
        - required: true

    responses:
        200:
            description: The output values
        
    """

    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    pred = classifier.predict(df_test)

    return str(list(pred))



if __name__ == '__main__':
    app.run(debug=True, host=5000)