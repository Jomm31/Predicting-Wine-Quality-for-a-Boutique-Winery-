import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("random_forest_model.joblib")

st.title("🍷 Wine Quality Classifier")

st.write("Enter the wine sample characteristics below:")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.001)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.4f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

if st.button("Predict Quality"):
    input_data = pd.DataFrame([{
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }])

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    predicted_class = "Good Quality" if prediction[0] == 1 else "Not Good Quality"
    confidence_score = prediction_proba[0][prediction[0]]

    st.subheader("Prediction Result")
    st.write(f"**The wine is predicted to be:** {predicted_class}")
    st.write(f"**Confidence Score:** {confidence_score:.2f}")
