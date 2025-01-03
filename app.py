from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model/linear_regression_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect input data from the form
            Present_Price = float(request.form['Present_Price'])
            Driven_kms = float(request.form['Driven_kms'])
            Car_Age = int(request.form['Car_Age'])

            Fuel_Type = int(request.form['Fuel_Type'])  
            Transmission = int(request.form['Transmission']) 
            Owner = int(request.form['Owner'])  
            Selling_type = int(request.form['Selling_type'])  

           
            input_features = np.array([[Present_Price, Driven_kms,
                                         Fuel_Type, Selling_type, Transmission, Owner, Car_Age]])

            # Apply scaling
            input_features[:, [0, 1, 6]] = scaler.transform(input_features[:, [0, 1, 6]])

            # Make prediction
            predicted_price = model.predict(input_features)
            output = round(predicted_price[0], 2)

            return render_template('result.html', prediction_text=f'Estimated Car Price: ${output}')
        except Exception as e:
           
            return render_template('index.html', error=f"Error: {str(e)}")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
