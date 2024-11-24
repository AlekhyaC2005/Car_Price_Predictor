import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

# Load the cleaned car data
car = pd.read_csv('Cleaned_car_data.csv')

# Prepare data for training
X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = car['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=247)

# OneHotEncoder and column transformer setup
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company', 'fuel_type']])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
    remainder='passthrough'
)

# Linear regression model
lr = LinearRegression()

# Pipeline creation
pipe = make_pipeline(column_trans, lr)

# Model training
pipe.fit(X_train, y_train)

# R2 Score check (optional, for debugging purposes)
y_pred = pipe.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2:.2f}")  # Can be removed in production

# Streamlit App
st.title("Car Price Predictor")
st.write("This app predicts the price of a car you want to sell. Fill in the details below:")

# Dynamic dropdowns from the dataset
company = st.selectbox("Select the company:", car["company"].unique())
model = st.selectbox("Select the model:", car[car["company"] == company]["name"].unique())
year = st.selectbox("Select Year of Purchase:", sorted(car["year"].unique()))
fuel_type = st.selectbox("Select the Fuel Type:", car["fuel_type"].unique())
kms_driven = st.text_input("Enter the Number of Kilometers the car has travelled:", "0")

# Predict button
if st.button("Predict Price"):
    try:
        # Validate kilometers driven
        kms_driven = int(kms_driven)

        # Create input for the model
        input_data = pd.DataFrame([[model, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Make prediction
        predicted_price = pipe.predict(input_data)[0]

        # Display the prediction
        st.success(f"The predicted price of the car is ₹ {predicted_price:,.2f}")

    except ValueError:
        st.error("Please enter a valid number for kilometers driven.")
