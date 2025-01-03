import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  
import numpy as np

df = pd.read_csv('car_data.csv')

print(df.head())
print(df.info())
print(df.describe())

df['Car_Age'] = 2025 - df['Year']

df = df.drop(['Car_Name', 'Year'], axis=1)

# handel dummies

df['Transmission']=df['Transmission'].map({'Manual':0,'Automatic':1})

df['Selling_type']=df['Selling_type'].map({'Dealer':0,'Individual':1})

df['Fuel_Type']=df['Fuel_Type'].map({'Petrol':0,'Diesel':1,'CNG':2})

# standardization 

scaler = StandardScaler()
numerical_features = ['Present_Price', 'Driven_kms', 'Car_Age']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_lr_pred = lr_model.predict(X_test)

mse_lr = mean_squared_error(y_test, y_lr_pred)
r2_lr = r2_score(y_test, y_lr_pred)

print(f'Linear Regression - Mean Squared Error: {mse_lr}')
print(f'Linear Regression - R^2 Score: {r2_lr}')


joblib.dump(lr_model, 'model/linear_regression_model.pkl')
print("Model saved to 'model/linear_regression_model.pkl'")

joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved to 'model/scaler.pkl'")

plt.scatter(y_test, y_lr_pred, label='Linear Regression Predictions', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', lw=2, label='Perfect Fit')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.legend()
plt.show()

