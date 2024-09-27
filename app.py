# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('model/linear_regression_model.pkl')

scaler = joblib.load('model/scaler.pkl')

columns = ['Present_Price', 'Driven_kms', 'Car_Age',
           'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
           'Selling_type_Used',
           'Transmission_Automatic',
           'Owner_1', 'Owner_2']  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
           
            Present_Price = float(request.form['Present_Price'])
            Driven_kms = float(request.form['Driven_kms'])
            Car_Age = float(request.form['Car_Age'])

            Fuel_Type = request.form['Fuel_Type']
            Selling_type = request.form['Selling_type']
            Transmission = request.form['Transmission']
            Owner = int(request.form['Owner'])  

            Fuel_Type_Diesel = 0
            Fuel_Type_Petrol = 0
            Selling_type_Used = 0
            Transmission_Automatic = 0
            Owner_1 = 0
            Owner_2 = 0

            if Fuel_Type == 'Diesel':
                Fuel_Type_Diesel = 1
            elif Fuel_Type == 'Petrol':
                Fuel_Type_Petrol = 1
            

            if Selling_type == 'Individual':
                Selling_type_Used = 1
        

            if Transmission == 'Automatic':
                Transmission_Automatic = 1
            
            if Owner == 1:
                Owner_2 = 1
            elif Owner == 0:
                Owner_1 = 1

            input_features = [Present_Price, Driven_kms, Car_Age,
                              Fuel_Type_Diesel, Fuel_Type_Petrol,
                              Selling_type_Used,
                              Transmission_Automatic,
                              Owner_1, Owner_2]

            features = np.array([input_features])

            features[:, 0:3] = scaler.transform(features[:, 0:3])

            prediction = model.predict(features)
            output = round(prediction[0], 2)

            return render_template('result.html', prediction_text=f'Estimated Car Price : ${output}')
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

