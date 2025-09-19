import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest model
model = joblib.load("random_forest_model.joblib")

st.title("🍷 Wine Quality Classifier")

st.write("""
This application predicts whether a wine sample is **Good Quality** or **Not Good Quality**.  

👉 The company defines **“Good Quality” wine** as one with a quality rating of **7 or higher**,  
and anything below that as **“Not Good.”**
""")

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

# Prediction button
if st.button("🔮 Predict Quality"):
    input_data = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])

    # Prediction (binary classification)
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    predicted_class = "Good Quality 🍷" if prediction[0] == 1 else "Not Good Quality ❌"
    confidence_score = prediction_proba[0][prediction[0]]

    # OPTIONAL: if your model was trained as regression/classification with rating (0–10), replace this with predict()
    try:
        predicted_rating = model.predict(input_data)[0]  
    except:
        predicted_rating = confidence_score * 10  # fallback: scale confidence score to 0–10

    # Display results
    st.subheader("Prediction Result")
    st.success(f"Predicted Class: **{predicted_class}**")
    st.info(f"Confidence Score: **{confidence_score:.2f}**")
    st.write(f"Predicted Quality Rating (0–10): **{predicted_rating:.1f}/10**")
