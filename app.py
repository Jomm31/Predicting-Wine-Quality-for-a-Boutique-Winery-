import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("random_forest_multi_class_model.joblib")

# Define the imputation values used during training
# These should ideally be saved alongside the model or calculated from the training data
# For this example, we'll use the mean and median calculated previously
# You would need to replace these with the actual calculated values from your training data
median_citric_acid = 0.26 # Replace with actual calculated median
mean_density = 0.996783 # Replace with actual calculated mean
mean_pH = 3.310454 # Replace with actual calculated mean

st.title("Wine Quality Predictor (Multi-class)")

st.write("""
Enter the chemical attributes of the wine to predict its quality rating (3-8).
""")

# Create input fields for each feature
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, format="%f")
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, format="%f")
citric_acid = st.number_input("Citric Acid", min_value=0.0, format="%f")
residual_sugar = st.number_input("Residual Sugar", min_value=0.0, format="%f")
chlorides = st.number_input("Chlorides", min_value=0.0, format="%f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=0.0, format="%f")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=0.0, format="%f")
density = st.number_input("Density", min_value=0.0, format="%f")
pH = st.number_input("pH", min_value=0.0, format="%f")
sulphates = st.number_input("Sulphates", min_value=0.0, format="%f")
alcohol = st.number_input("Alcohol", min_value=0.0, format="%f")

# Create a button to make predictions
if st.button("Predict Quality"):
    # Create a pandas DataFrame from the input values
    data = {
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    }
    input_df = pd.DataFrame(data)

    # Handle missing values using the same imputation strategy as training
    input_df['citric acid'] = input_df['citric acid'].fillna(median_citric_acid)
    input_df['density'] = input_df['density'].fillna(mean_density)
    input_df['pH'] = input_df['pH'].fillna(mean_pH)


    # Make predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Get the predicted class and its probability
    predicted_class = prediction[0]
    confidence = prediction_proba[0][predicted_class - 3] # Adjust index for quality ratings 3-8

    st.success(f"Predicted Wine Quality: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")
