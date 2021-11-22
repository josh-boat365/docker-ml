from flask import Flask, request
import numpy as np 
import pandas as pd
import pickle
# creating instance of flask app
app = Flask(__name__)

pickle_load = open('./classifier.pkl', 'rb')
classifier = pickle.load(pickle_load)

@app.route('/')
def home():
    return "WELCOMOE"

@app.route('/predict')
def predict_note_authentication():
    variance =request.args.get('variance')
    skewness =request.args.get('skewness')
    curtosis =request.args.get('curtosis')
    entropy =request.args.get('entropy')

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    return "The prediction value is "+ str(prediction)

@app.route('/predict_file', methods=["POST"])
def predict_file_test():
    df_test = pd.read_csv(request.files.get("file"))
    pred = classifier.predict(df_test)

    return "The prediction values for the CSV TEST FILE is "+ str(list(pred))










if __name__ == '__main__':
    app.run(debug=True, host=5000)